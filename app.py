from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any
from dotenv import load_dotenv
import logging
import traceback
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins

# Initialize OpenAI model
try:
    logging.info("Initializing OpenAI model")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    logging.debug("OpenAI model initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI: {str(e)}")
    logging.debug(f"Stack trace: {traceback.format_exc()}")
    llm = None

# API endpoints
PROPOSAL_API_URL = "https://api-int.uat-riskcovry.com/motor/v2/plans/selected_plan_information?quote_id={}"
PLAN_LISTING_API_URL = "https://api-int.uat-riskcovry.com/motor/fetch_quote_list?quote_id={}"

def fetch_api_data(quote_id: str) -> tuple[Dict, List]:
    headers = {
        "Content-Type": "application/json"
    }
    try:
        logging.debug(f"Fetching Proposal API: {PROPOSAL_API_URL.format(quote_id)}")
        proposal_response = requests.get(PROPOSAL_API_URL.format(quote_id), headers=headers, timeout=10)
        proposal_response.raise_for_status()
        proposal_data = proposal_response.json()

        logging.debug(f"Fetching Plan Listing API: {PLAN_LISTING_API_URL.format(quote_id)}")
        plan_listing_response = requests.get(PLAN_LISTING_API_URL.format(quote_id), headers=headers, timeout=10)
        plan_listing_response.raise_for_status()
        plan_listing_data = plan_listing_response.json()

        return proposal_data, plan_listing_data
    except requests.RequestException as e:
        logging.error(f"API fetch failed: {str(e)}")
        raise ValueError(f"API fetch failed: {str(e)}")

def validate_field(expected: Any, actual: Any, field: str, context: Dict = None) -> Dict:
    # Normalize expected and actual to check for logical equivalence
    null_values = [None, "", "None", "nan", "NaN"]
    
    # Normalize expected
    if isinstance(expected, float) and np.isnan(expected):
        expected_normalized = None
    elif isinstance(expected, str) and expected.lower() in ["nan", "none"]:
        expected_normalized = None
    else:
        expected_normalized = expected

    # Normalize actual
    if isinstance(actual, float) and np.isnan(actual):
        actual_normalized = None
    elif isinstance(actual, str) and actual.lower() in ["nan", "none"]:
        actual_normalized = None
    else:
        actual_normalized = actual

    # Check if both are null-like values
    if (expected_normalized is None or str(expected_normalized).lower() in null_values) and \
       (actual_normalized is None or str(actual_normalized).lower() in null_values):
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "status": "Pass",
            "reason": "Both expected and actual are null/empty/undefined"
        }

    if not llm:
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "status": "Fail",
            "reason": "OpenAI model not initialized"
        }

    prompt = PromptTemplate(
        input_variables=["field", "expected", "actual", "context"],
        template="""
Compare the expected and actual values for the insurance field '{field}'.

Expected:
{expected}

Actual:
{actual}

Context:
{context}

Instructions:
- Treat 'nan', 'NaN', 'None', null, and empty string as equivalent null values.
- Check for exact value match, case-insensitive if string.
- Respect data types (e.g., int ≠ str) unless both are null-like.
- If `expected` is "min" or "max", and context contains min/max range, validate accordingly.
- Explain mismatches clearly, e.g., wrong IDV range, missing discount, mismatched previous policy data.

Respond with JSON only:
{{
  "field": "{field}",
  "expected": "{expected}",
  "actual": "{actual}",
  "status": "Pass" | "Fail" | "Pending",
  "reason": "Clear and short explanation"
}}
"""
    )

    try:
        response = llm.invoke(prompt.format(
            field=field,
            expected=str(expected_normalized if expected_normalized is not None else "null"),
            actual=str(actual_normalized if actual_normalized is not None else "null"),
            context=json.dumps(context or {})
        ))
        return json.loads(response.content)
    except Exception as e:
        logging.error(f"Validation error for field {field}: {str(e)}")
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "status": "Fail",
            "reason": f"Validation error: {str(e)}"
        }

def validate_quote(test_data: Dict, proposal_data: Dict, plan_listing_data: Dict) -> List[Dict]:
    """Validate all required fields."""
    logging.info("Starting quote validation")
    results = []
    quote = next((q for q in plan_listing_data.get("quotes", []) if q["id"] == proposal_data["quotation_id"]), None)
    if not quote:
        result = {
            "field": "quote",
            "expected": "valid quote",
            "actual": "none",
            "status": "Fail",
            "reason": "Quote ID not found in plan listing."
        }
        logging.error(f"Validation failed: {result}")
        return [result]

    # NCB applicability
    ncb_context = {
        "is_new_business": proposal_data["vehicle"].get("business_type") == "new_business",
        "is_ownership_transferred": proposal_data["vehicle"].get("is_ownership_transferred", False),
        "previous_policy_expiry_date": proposal_data["vehicle"].get("previous_policy_expiry_date"),
        "previous_policy_type": proposal_data["vehicle"].get("previous_policy_type"),
        "product_code": test_data.get("product_code", "")
    }
    logging.debug(f"NCB context: {ncb_context}")
    ncb_applicable = (
        not ncb_context["is_new_business"] and
        not ncb_context["is_ownership_transferred"] and
        ncb_context["previous_policy_type"] != "third_party" and
        "COMPREHENSIVE" in ncb_context["product_code"] and
        (ncb_context["previous_policy_expiry_date"] is None or "recent" in str(ncb_context["previous_policy_expiry_date"]))
    )
    logging.debug(f"NCB applicable: {ncb_applicable}")

    # Validate addons
    addons_value = test_data.get("addons", "")
    logging.debug(f"Addons value: {addons_value}, type: {type(addons_value)}")
    if isinstance(addons_value, float) and np.isnan(addons_value):
        expected_addons = []
    elif isinstance(addons_value, str) and addons_value:
        expected_addons = addons_value.split(",")
    else:
        expected_addons = []
    actual_addons = [addon["cover_code"] for addon in proposal_data.get("addons", []) if addon.get("selected")]
    # Check for mandatory addons (is_removable: false)
    mandatory_addons = [addon["cover_code"] for addon in proposal_data.get("addons", []) if addon.get("is_removable", True) == False]
    logging.debug(f"Mandatory addons (is_removable: false): {mandatory_addons}")
    # If expected is empty and actual contains mandatory addons, pass with mandatory flag
    if not expected_addons and actual_addons and all(addon in mandatory_addons for addon in actual_addons):
        results.append({
            "field": "addons",
            "expected": str(addons_value),
            "actual": str(actual_addons),
            "status": "Pass",
            "reason": "Expected addons are empty; actual addons are mandatory (is_removable: false)",
            "is_mandatory": True
        })
    else:
        results.append(validate_field(expected_addons, actual_addons, "addons"))

    # Validate discounts
    discounts_value = test_data.get("discounts", "")
    logging.debug(f"Discounts value: {discounts_value}, type: {type(discounts_value)}")
    if isinstance(discounts_value, float) and np.isnan(discounts_value):
        expected_discounts = []
    elif isinstance(discounts_value, str) and discounts_value:
        expected_discounts = discounts_value.split(",")
    else:
        expected_discounts = []
    actual_discounts = [d.get("code") for d in proposal_data.get("discounts", [])]
    ncb_discount_context = {"ncb_applicable": ncb_applicable, "discounts": actual_discounts}
    if ncb_applicable and "NCB_DISCOUNT" not in actual_discounts:
        result = {
            "field": "discounts",
            "expected": "NCB_DISCOUNT",
            "actual": actual_discounts,
            "status": "Fail",
            "reason": "NCB_DISCOUNT missing when NCB is applicable."
        }
        logging.error(f"Discount validation failed: {result}")
        results.append(result)
    else:
        results.append(validate_field(expected_discounts, actual_discounts, "discounts", ncb_discount_context))

    # Validate IDV
    idv_value = test_data.get("idv", "")
    logging.debug(f"IDV value: {idv_value}, type: {type(idv_value)}")
    if isinstance(idv_value, float) and np.isnan(idv_value):
        expected_idv = ""
    else:
        expected_idv = str(idv_value)
    actual_idv = proposal_data["vehicle"].get("idv")
    idv_context = {
        "min_idv": quote.get("min_sum_insured") if quote else None,
        "max_idv": quote.get("max_sum_insured") if quote else None
    }
    null_values = [None, "", "None", "nan", "NaN"]
    if (isinstance(idv_value, float) and np.isnan(idv_value)) or str(idv_value).lower() in null_values:
        if actual_idv is not None and idv_context["min_idv"] is not None and idv_context["max_idv"] is not None:
            try:
                actual_idv_float = float(actual_idv)
                min_idv = float(idv_context["min_idv"])
                max_idv = float(idv_context["max_idv"])
                if min_idv <= actual_idv_float <= max_idv:
                    results.append({
                        "field": "idv",
                        "expected": str(idv_value),
                        "actual": str(actual_idv),
                        "status": "Pass",
                        "reason": f"Expected IDV is empty; actual IDV {actual_idv} is within range [{min_idv}, {max_idv}]",
                        "range_validated": True
                    })
                else:
                    results.append({
                        "field": "idv",
                        "expected": str(idv_value),
                        "actual": str(actual_idv),
                        "status": "Fail",
                        "reason": f"Expected IDV is empty; actual IDV {actual_idv} is outside range [{min_idv}, {max_idv}]"
                    })
            except (ValueError, TypeError):
                results.append(validate_field(expected_idv, actual_idv, "idv", idv_context))
        else:
            results.append(validate_field(expected_idv, actual_idv, "idv", idv_context))
    elif expected_idv.lower() == "min":
        results.append(validate_field(idv_context["min_idv"], actual_idv, "idv", idv_context))
    elif expected_idv.lower() == "max":
        results.append(validate_field(idv_context["max_idv"], actual_idv, "idv", idv_context))
    else:
        results.append(validate_field(expected_idv, actual_idv, "idv", idv_context))

    # Validate NCB
    expected_ncb = test_data.get("previous_ncb", "")
    actual_ncb = proposal_data["vehicle"].get("previous_policy_ncb")
    results.append(validate_field(expected_ncb, actual_ncb, "previous_ncb", {"ncb_applicable": ncb_applicable}))

    # Validate previous policy fields
    previous_fields = [
        "previous_expiry_date", "previous_insurer", "previous_tp_expiry_date",
        "previous_tp_insurer", "claim_taken"
    ]
    for field in previous_fields:
        api_field = field if field == "claim_taken" else f"previous_policy_{field.replace('previous_', '')}"
        if field == "previous_tp_insurer":
            api_field = "previous_tp_policy_carrier_name"
        elif field == "previous_tp_expiry_date":
            api_field = "previous_tp_policy_expiry_date"
        expected = test_data.get(field, "")
        actual = proposal_data["vehicle"].get(api_field)
        results.append(validate_field(expected, actual, field))

    # Validate proposal number
    results.append(validate_field("non-empty", proposal_data.get("proposal_number"), "proposal_number"))

    # Validate inspection
    expected_inspection = test_data.get("is_inspection_required", "No") == "Yes"
    actual_inspection = proposal_data["vehicle"].get("is_break_in")
    results.append(validate_field(expected_inspection, actual_inspection, "is_break_in"))

    logging.info(f"Validation completed with {len(results)} results")
    return results

@app.route('/')
def index():
    logging.info("Serving index page")
    return send_file('index.html')

@app.route('/script.js')
def serve_script():
    logging.info("Serving script.js")
    return send_file('script.js')

@app.route('/validate', methods=['POST'])
def validate():
    logging.info("Received /validate request")
    quote_id = request.form.get('quote_id')
    if not quote_id:
        logging.error("Quote ID missing in request")
        return jsonify({"error": "Quote ID is required."}), 400

    try:
        # Read test data from CSV
        try:
            logging.info("Reading CSV file: ./test_data.csv")
            df = pd.read_csv("./test_data.csv")
            logging.debug(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
            test_data = df.iloc[0].to_dict()
            logging.debug(f"Test data: {test_data}")
        except FileNotFoundError:
            logging.error("test_data.csv not found")
            return jsonify({"error": "CSV file not found: ./test_data.csv"}), 400
        except Exception as e:
            logging.error(f"CSV reading error: {str(e)}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

        # Fetch API data
        logging.info(f"Fetchinng API data for quote_id: {quote_id}")
        proposal_data, plan_listing_data = fetch_api_data(quote_id)
        
        # Validate
        logging.info("Starting validation")
        results = validate_quote(test_data, proposal_data, plan_listing_data)
        
        response = {
            "testcase_id": test_data.get("testcase_id", "Unknown"),
            "quote_id": quote_id,
            "validation_results": results
        }
        logging.info("Validation completed successfully")
        logging.debug(f"Response: {json.dumps(response)[:500]}")
        return jsonify(response)
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        logging.debug(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.debug(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(debug=True, host="0.0.0.0", port=port)