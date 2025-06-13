# Author: GeorgeET15
# Description: I created this Flask API during my internship at Riskcovry to validate insurance quote data. It compares
# test case data from CSV files against responses from external APIs. I designed it to support batch API data fetching,
# caching, and field-specific validation for insurance parameters like add-ons, discounts, IDV, and policy details. The
# application integrates with OpenAI's language model for complex validations and uses concurrent processing for efficiency.
# I included robust error handling, detailed logging, and secure temporary file management for CSV processing to ensure
# reliability and performance.

# Import required libraries for Flask API, data processing, API requests, and OpenAI integration
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
import numpy as np
import urllib.parse
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from cachetools import TTLCache   # type: ignore
import time
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta

# Configure logging with DEBUG level to capture detailed stack traces and error tracking
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file for secure configuration
load_dotenv()

# Initialize Flask application with CORS enabled to allow cross-origin requests from frontends
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize OpenAI model for complex field validations using API key from environment
try:
    logging.info("Initializing OpenAI model")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    logging.debug("OpenAI model initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI: {str(e)}", exc_info=True)
    llm = None

# Define external API endpoints for fetching proposal, plan listing, and thank-you page data
PROPOSAL_API_URL = "https://api-int.uat-riskcovry.com/motor/v2/plans/selected_plan_information?quote_id={}"
PLAN_LISTING_API_URL = "https://api-int.uat-riskcovry.com/motor/fetch_quote_list?quote_id={}"
THANK_YOU_API_URL = "https://api-int.uat-riskcovry.com/policies/get_policy"

# Initialize TTL cache to store API responses for 1 hour to minimize redundant requests
cache = TTLCache(maxsize=1000, ttl=3600)

# Store paths of uploaded input and output CSV files as global variables
INPUT_CSV_PATH = None
OUTPUT_CSV_PATH = None

def fetch_api_data(quote_id: str) -> tuple[Dict, List]:
    """Fetch proposal and plan listing data for a given quote ID from external APIs.

    I implemented this function to retrieve and cache data efficiently, reducing API call overhead.

    Args:
        quote_id (str): Unique identifier for the quote.

    Returns:
        tuple[Dict, List]: Proposal data and plan listing data.

    Raises:
        ValueError: If API requests fail.
    """
    if quote_id in cache:
        logging.debug(f"Cache hit for quote_id: {quote_id}")
        return cache[quote_id]
    
    headers = {"Content-Type": "application/json"}
    start_time = time.time()
    try:
        # Fetch proposal data
        logging.debug(f"Fetching Proposal API: {PROPOSAL_API_URL.format(quote_id)}")
        proposal_response = requests.get(PROPOSAL_API_URL.format(quote_id), headers=headers, timeout=5)
        proposal_response.raise_for_status()
        proposal_data = proposal_response.json()

        # Fetch plan listing data
        logging.debug(f"Fetching Plan Listing API: {PLAN_LISTING_API_URL.format(quote_id)}")
        plan_listing_response = requests.get(PLAN_LISTING_API_URL.format(quote_id), headers=headers, timeout=5)
        plan_listing_response.raise_for_status()
        plan_listing_data = plan_listing_response.json()

        # Cache the results to avoid redundant API calls
        cache[quote_id] = (proposal_data, plan_listing_data)
        logging.info(f"API fetch for quote_id {quote_id} took {time.time() - start_time:.2f} seconds")
        return proposal_data, plan_listing_data
    except requests.RequestException as e:
        logging.error(f"API fetch failed for quote_id {quote_id}: {str(e)}")
        logging.debug(f"API response: {e.response.text if e.response else 'No response'}")
        logging.info(f"API fetch for quote_id {quote_id} took {time.time() - start_time:.2f} seconds")
        raise ValueError(f"API fetch failed: {str(e)}")

def fetch_api_data_batch(quote_ids: List[str]) -> Dict[str, tuple[Dict, List]]:
    """Fetch API data for multiple quote IDs concurrently.

    I added concurrent processing with ThreadPoolExecutor to improve performance for batch requests.

    Args:
        quote_ids (List[str]): List of quote IDs to fetch data for.

    Returns:
        Dict[str, tuple[Dict, List]]: Dictionary mapping quote IDs to their proposal and plan listing data.
    """
    results = {}
    
    def fetch_single(quote_id):
        """Helper function to fetch data for a single quote ID."""
        try:
            return quote_id, fetch_api_data(quote_id)
        except ValueError as e:
            return quote_id, None
    
    start_time = time.time()
    # Use ThreadPoolExecutor for concurrent API requests with a maximum of 10 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_quote = {executor.submit(fetch_single, qid): qid for qid in quote_ids}
        for future in as_completed(future_to_quote):
            quote_id, result = future.result()
            if result:
                results[quote_id] = result
            else:
                logging.error(f"Failed to fetch data for quote_id: {quote_id}")
    
    logging.info(f"Batch API fetch for {len(quote_ids)} quote_ids took {time.time() - start_time:.2f} seconds")
    return results

def extract_quote_id(thank_url: str) -> tuple[str, str]:
    """Extract quote ID from a thank-you URL or fetch it via the thank-you API.

    I implemented this to handle both URL-based and API-based quote ID extraction for flexibility.

    Args:
        thank_url (str): URL containing the quote ID or reference ID.

    Returns:
        tuple[str, str]: Quote ID and error message (if any).
    """
    if not thank_url or not isinstance(thank_url, str) or pd.isna(thank_url):
        logging.debug(f"Invalid ThankURL: {thank_url}")
        return None, "Invalid or empty ThankURL"
    
    parsed_url = urllib.parse.urlparse(thank_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    if "quote_id" in query_params:
        return query_params["quote_id"][0], None
    
    match = re.search(r'thank-you/([^/]+)/', thank_url)
    if match:
        ref_id = match.group(1)
        try:
            response = requests.get(f"{THANK_YOU_API_URL}?ref={ref_id}", headers={"Content-Type": "application/json"}, timeout=5)
            response.raise_for_status()
            data = response.json()
            quote_id = data.get("quotation_search", {}).get("session_token")
            if quote_id:
                return quote_id, None
            return None, "No session_token found in Thank You Page API response"
        except requests.RequestException as e:
            return None, f"Thank You Page API error: {str(e)}"
    
    return None, "No quote_id or thank-you ID found in ThankURL"

def validate_field(expected: Any, actual: Any, field: str, context: Dict = None) -> Dict:
    """Validate a single field by comparing expected and actual values.

    I designed this function to handle various field types (dates, strings, lists, etc.) with special cases for insurance fields.

    Args:
        expected (Any): Expected value from test data.
        actual (Any): Actual value from API response.
        field (str): Name of the field being validated.
        context (Dict, optional): Additional context for validation (e.g., min/max IDV, date offsets).

    Returns:
        Dict: Validation result with field, expected, actual, status, and reason.
    """
    null_values = [None, "", "None", "null", "nan", "NaN"]
    context = context or {}
    
    # Normalize inputs to handle empty values consistently
    expected_normalized = None if expected in null_values or (isinstance(expected, str) and expected.lower() in null_values) or (isinstance(expected, float) and np.isnan(expected)) else expected
    actual_normalized = None if actual in null_values or (isinstance(actual, str) and actual.lower() in null_values) or (isinstance(actual, float) and np.isnan(actual)) else actual
    
    logging.debug(f"Normalized for field {field}: expected={expected_normalized}, actual={actual_normalized}")
    
    # Handle empty equivalence for all fields
    if expected_normalized is None and actual_normalized is None:
        return {
            "field": field,
            "expected": str(expected) if expected is not None else "",
            "actual": str(actual) if actual is not None else "",
            "status": "Pass",
            "reason": f"Both expected and actual values for '{field}' are empty (e.g., '', None, null, nan), considered equivalent."
        }
    
    # Handle claim_taken special case
    if field == "claim_taken" and expected_normalized == "No" and actual_normalized is None:
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual) if actual is not None else "",
            "status": "Pass",
            "reason": f"For '{field}', expected 'No' is equivalent to actual empty value, as empty is treated as 'No' per validation rules."
        }
    
    # Handle date fields with flexible parsing and offset-based validation
    if field in ["previous_tp_expiry_date", "previous_expiry_date"]:
        if expected_normalized is None and actual_normalized is None:
            return {
                "field": field,
                "expected": str(expected) if expected is not None else "",
                "actual": str(actual) if actual is not None else "",
                "status": "Pass",
                "reason": f"Date comparison for '{field}': both expected and actual are empty, considered equivalent."
            }
        try:
            # Parse actual date with flexible format support
            actual_date = parse_date(str(actual_normalized)) if actual_normalized else None
            logging.debug(f"Parsed actual date for '{field}': '{actual_normalized}' -> {actual_date}")

            # Parse expected date, prioritizing DD/MM/YYYY for CSV input
            expected_date = None
            if expected_normalized:
                if isinstance(expected_normalized, str) and re.match(r'^\d{2}/\d{2}/\d{4}$', expected_normalized):
                    try:
                        expected_date = datetime.strptime(expected_normalized, '%d/%m/%Y')
                        logging.debug(f"Parsed expected date for '{field}' as DD/MM/YYYY: '{expected_normalized}' -> {expected_date}")
                    except ValueError as e:
                        logging.warning(f"Failed to parse expected date '{expected_normalized}' as DD/MM/YYYY: {str(e)}")
                        expected_date = parse_date(str(expected_normalized))
                        logging.debug(f"Fallback parsing for expected date: '{expected_normalized}' -> {expected_date}")
                else:
                    expected_date = parse_date(str(expected_normalized))
                    logging.debug(f"Parsed expected date for '{field}': '{expected_normalized}' -> {expected_date}")

            # Compare parsed dates
            if expected_date and actual_date and expected_date.date() == actual_date.date():
                return {
                    "field": field,
                    "expected": str(expected),
                    "actual": str(actual),
                    "status": "Pass",
                    "reason": f"Date comparison for '{field}': expected '{str(expected)}' ({expected_date.date()}) matches actual '{str(actual)}' ({actual_date.date()})."
                }

            reason_parts = []
            if expected_date or actual_date:
                reason_parts.append(
                    f"Date comparison for '{field}': expected '{str(expected)}' ({expected_date.date() if expected_date else 'None'}) "
                    f"does not match actual '{str(actual)}' ({actual_date.date() if actual_date else 'None'})."
                )

            # Computed date comparison for offset-based validation
            if context.get("created_at") and context.get("offset_days"):
                try:
                    created_at_date = parse_date(context["created_at"])
                    offset = int(context["offset_days"])
                    computed_date = created_at_date + timedelta(days=offset)
                    computed_date_str = computed_date.strftime("%d/%m/%Y")
                    if actual_date and computed_date.date() == actual_date.date():
                        reason_parts.append(
                            f"Computed date '{computed_date_str}' (created_at '{context['created_at']}' + {offset} days) "
                            f"matches actual '{str(actual)}' ({actual_date.date()})."
                        )
                        if expected_date:
                            logging.warning(
                                f"CSV date '{str(expected)}' for '{field}' does not match actual, "
                                f"but computed date '{computed_date_str}' matches."
                            )
                        return {
                            "field": field,
                            "expected": computed_date_str,
                            "actual": str(actual),
                            "status": "Pass",
                            "reason": f"{' '.join(reason_parts)} Original CSV expected: '{str(expected)}'."
                        }
                    else:
                        reason_parts.append(
                            f"Computed date '{computed_date_str}' (created_at '{context['created_at']}' + {offset} days) "
                            f"does not match actual '{str(actual)}' ({actual_date.date() if actual_date else 'None'})."
                        )
                except (ValueError, TypeError) as e:
                    logging.debug(f"Offset-based date computation skipped for '{field}': {str(e)}.")
                    reason_parts.append(
                        f"Offset-based comparison skipped: error computing date from created_at '{context['created_at']}' "
                        f"and offset '{context['offset_days']}': {str(e)}."
                    )

            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Fail",
                "reason": " ".join(reason_parts) or f"Date comparison for '{field}': no valid comparison possible."
            }
        except (ValueError, TypeError) as e:
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Fail",
                "reason": f"Date comparison failed for '{field}': expected '{expected}', actual '{actual}'. Unable to parse dates: {str(e)}."
            }
    
    # Handle previous_ncb with percentage normalization
    if field == "previous_ncb":
        if expected_normalized is None and actual_normalized is None:
            return {
                "field": field,
                "expected": str(expected) if expected is not None else "",
                "actual": str(actual) if actual is not None else "",
                "status": "Pass",
                "reason": f"NCB comparison for '{field}': both expected '{expected}' and actual '{actual}' are empty, considered equivalent."
            }
        try:
            expected_ncb = float(str(expected_normalized).strip("%")) if expected_normalized else 0.0
            actual_ncb = float(str(actual_normalized).strip("%")) if actual_normalized else 0.0
            if expected_ncb == actual_ncb:
                return {
                    "field": field,
                    "expected": str(expected),
                    "actual": str(actual),
                    "status": "Pass",
                    "reason": f"NCB comparison for '{field}': expected '{expected}' and actual '{actual}' both normalize to {expected_ncb}."
                }
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Fail",
                "reason": f"NCB comparison for '{field}': expected '{expected}' (normalized: {expected_ncb}), actual '{actual}' (normalized: {actual_ncb}). Values differ."
                }
        except (ValueError, TypeError) as e:
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Fail",
                "reason": f"NCB comparison failed for '{field}': expected '{expected}', actual '{actual}'. Unable to normalize values: {str(e)}."
            }
    
    # Handle IDV with min/max range validation
    if field == "idv" and str(expected_normalized).lower() in ["min", "max"]:
        range_key = "min_idv" if str(expected_normalized).lower() == "min" else "max_idv"
        range_value = context.get(range_key)
        try:
            actual_float = float(actual_normalized) if actual_normalized else None
            range_float = float(range_value) if range_value else None
            if actual_float is None and range_float is None:
                return {
                    "field": field,
                    "expected": str(range_value) if range_value is not None else "",
                    "actual": str(actual_normalized) if actual_normalized is not None else "",
                    "status": "Pass",
                    "reason": f"IDV comparison for '{field}': both expected and actual are empty, considered equivalent."
                }
            if actual_float is not None and range_float is not None:
                status = "Pass" if actual_float == range_float else "Fail"
                reason = (
                    f"Expected IDV set to '{str(expected_normalized).lower()}', which corresponds to {range_float} from API. "
                    f"Actual IDV is {actual_float}. "
                    f"{'Values match.' if status == 'Pass' else f'Values differ; expected {range_float}, got {actual_float}.'}"
                )
                return {
                    "field": field,
                    "expected": str(range_value),
                    "actual": str(actual_normalized),
                    "status": status,
                    "reason": reason
                }
            return {
                "field": field,
                "expected": str(range_value),
                "actual": str(actual_normalized),
                "status": "Fail",
                "reason": f"Invalid IDV comparison: expected '{range_value}', actual '{actual_normalized}'. One value is empty."
            }
        except (ValueError, TypeError) as e:
            return {
                "field": field,
                "expected": str(range_value),
                "actual": str(actual_normalized),
                "status": "Fail",
                "reason": f"Invalid IDV comparison due to type error: expected '{range_value}', actual '{actual_normalized}'. Error: {str(e)}."
            }
    
    # Handle proposal_number non-empty validation
    if field == "proposal_number" and expected_normalized == "non-empty":
        if actual_normalized and isinstance(actual_normalized, str) and actual_normalized.strip():
            return {
                "field": field,
                "expected": "non-empty",
                "actual": str(actual),
                "status": "Pass",
                "reason": f"Proposal number '{actual}' is non-empty as expected."
            }
        return {
            "field": field,
            "expected": "non-empty",
            "actual": str(actual) if actual is not None else "",
            "status": "Fail",
            "reason": f"Proposal number is empty or null, which does not meet the expectation of being non-empty."
        }
    
    # Handle case-insensitive string comparisons
    if isinstance(expected_normalized, str) and isinstance(actual_normalized, str):
        if expected_normalized.lower() == actual_normalized.lower():
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Pass",
                "reason": f"String comparison for '{field}': expected '{expected}' and actual '{actual}' match (case-insensitive)."
            }
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "status": "Fail",
            "reason": f"String comparison for '{field}': expected '{expected}' and actual '{actual}' differ (case-insensitive)."
        }
    
    # Handle array comparisons for addons and discounts
    if field in ["addons", "discounts"]:
        expected_list = expected_normalized if isinstance(expected_normalized, list) else []
        actual_list = actual_normalized if isinstance(actual_normalized, list) else []
        if not expected_list and not actual_list:
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Pass",
                "reason": f"Both expected and actual {field} are empty lists, indicating no {field} are present."
            }
        missing = [item for item in expected_list if item not in actual_list]
        extra = [item for item in actual_list if item not in expected_list]
        if not missing and not extra:
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Pass",
                "reason": f"Expected {field} {expected_list} match actual {field} {actual_list}."
            }
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "status": "Fail",
            "reason": f"{field.capitalize()} mismatch: missing {missing}, extra {extra}."
        }
    
    # Handle boolean comparisons for is_break_in
    if field == "is_break_in":
        expected_bool = (
            bool(expected_normalized)
            if isinstance(expected_normalized, (bool, str, int))
            else False
        )
        actual_bool = (
            bool(actual_normalized)
            if isinstance(actual_normalized, (bool, str, int))
            else False
        )
        if expected_bool == actual_bool:
            return {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "status": "Pass",
                "reason": f"Expected is_break_in {expected_bool} matches actual {actual_bool}."
            }
        return {
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "status": "Fail",
            "reason": f"Expected is_break_in {expected_bool}, but actual is {actual_bool}."
        }
    
    # Fallback to OpenAI for complex validations
    if not llm:
        return {
            "field": field,
            "expected": str(expected) if expected is not None else "",
            "actual": str(actual) if actual is not None else "",
            "status": "Fail",
            "reason": f"OpenAI model not initialized for '{field}' validation. Cannot process complex comparison."
        }
    
    # Define prompt template for OpenAI validation
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
- Return a valid JSON object with keys: "field", "expected", "actual", "status", "reason".
- Treat '', None, null, nan, NaN as equivalent empty values; represent as "null" in JSON.
- For arrays, compare as lists, ensuring all expected elements are in actual.
- Status must be one of "Pass", "Fail", or "Pending".
- Reason must be a concise string (1-2 sentences) explaining the comparison result.
- Escape special characters in "expected", "actual", and "reason" for valid JSON.
- Do not nest JSON in the reason field; it must be a plain string.
- Example output:
  {{
    "field": "example_field",
    "expected": "value",
    "actual": "value",
    "status": "Pass",
    "reason": "Expected value matches actual value."
  }}

Return:
{{
  "field": "{field}",
  "expected": "{expected}",
  "actual": "{actual}",
  "status": "Pass" | "Fail" | "Pending",
  "reason": "Detailed explanation of comparison."
}}
"""
    )
    
    start_time = time.time()
    try:
        # Invoke OpenAI model for complex validation
        response = llm.invoke(prompt.format(
            field=field,
            expected=str(expected_normalized if expected_normalized is not None else "null"),
            actual=str(actual_normalized if actual_normalized is not None else "null"),
            context=json.dumps(context or {})
        ))
        logging.debug(f"Raw LLM response for field {field}: {response.content}")
        logging.info(f"LLM validation for field {field} took {time.time() - start_time:.2f} seconds")
        
        if not response.content or response.content.isspace():
            logging.error(f"Empty LLM response for field {field}")
            return {
                "field": field,
                "expected": str(expected) if expected is not None else "",
                "actual": str(actual) if actual is not None else "",
                "status": "Fail",
                "reason": "Empty response from OpenAI model."
            }
        
        try:
            result = json.loads(response.content.strip())
            if not isinstance(result, dict):
                logging.error(f"Invalid LLM response format for field {field}: {response.content}")
                return {
                    "field": field,
                    "expected": str(expected) if expected is not None else "",
                    "actual": str(actual) if actual is not None else "",
                    "status": "Fail",
                    "reason": f"Invalid LLM response format: not a dictionary. Raw response: {response.content}"
                }
            
            # Validate required keys in LLM response
            required_keys = {"field", "expected", "actual", "status", "reason"}
            if not all(key in result for key in required_keys):
                logging.error(f"Missing required keys in LLM response for field {field}: {result}")
                return {
                    "field": field,
                    "expected": str(expected) if expected is not None else "",
                    "actual": str(actual) if actual is not None else "",
                    "status": "Fail",
                    "reason": f"Invalid LLM response: missing required keys {required_keys - set(result.keys())}. Raw response: {response.content}"
                }
            
            # Validate status value
            if result["status"] not in ["Pass", "Fail", "Pending"]:
                logging.error(f"Invalid status in LLM response for field {field}: {result['status']}")
                return {
                    "field": field,
                    "expected": str(expected) if expected is not None else "",
                    "actual": str(actual) if actual is not None else "",
                    "status": "Fail",
                    "reason": f"Invalid LLM response: status '{result['status']}' is not one of Pass, Fail, Pending. Raw response: {response.content}"
                }
            
            # Ensure reason is a string
            if not isinstance(result["reason"], str):
                logging.error(f"Invalid reason type in LLM response for field {field}: {type(result['reason'])}")
                return {
                    "field": field,
                    "expected": str(expected) if expected is not None else "",
                    "actual": str(actual) if actual is not None else "",
                    "status": "Fail",
                    "reason": f"Invalid LLM response: reason must be a string. Raw response: {response.content}"
                }
            
            return result
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON parse error for field {field}: {str(json_err)}")
            logging.debug(f"Raw LLM response: {response.content}")
            return {
                "field": field,
                "expected": str(expected) if expected is not None else "",
                "actual": str(actual) if actual is not None else "",
                "status": "Fail",
                "reason": f"Failed to parse LLM response as JSON: {response.content}"
            }
    except Exception as e:
        logging.error(f"LLM validation error for field {field}: {str(e)}", exc_info=True)
        return {
            "field": field,
            "expected": str(expected) if expected is not None else "",
            "actual": str(actual) if actual is not None else "",
            "status": "Fail",
            "reason": f"LLM validation failed: {str(e)}."
        }

def validate_quote(test_data: Dict, proposal_data: Dict, plan_listing_data: Dict) -> List[Dict]:
    """Validate a single quote by comparing test data against API data.

    I implemented this function to handle validation for multiple insurance fields, including add-ons, discounts, and dates.

    Args:
        test_data (Dict): Test case data from CSV.
        proposal_data (Dict): Data from the proposal API.
        plan_listing_data (Dict): Data from the plan listing API.

    Returns:
        List[Dict]: List of validation results for each field.
    """
    logging.info("Starting quote validation")
    start_time = time.time()
    
    # Log API data for debugging
    logging.debug(f"Proposal data: {json.dumps(proposal_data, indent=2)}")
    logging.debug(f"Plan listing data: {json.dumps(plan_listing_data, indent=2)}")
    
    results = []
    
    # Validate input data structure
    if not isinstance(proposal_data, dict) or not isinstance(plan_listing_data, dict):
        logging.error("Invalid API data structure")
        return [{
            "field": "quote",
            "expected": "valid API data",
            "actual": "invalid",
            "status": "Fail",
            "reason": "Proposal or plan listing data is not a valid dictionary."
        }]
    
    # Find matching quote in plan listing data
    quote = next((q for q in plan_listing_data.get("quotes", []) if q["id"] == proposal_data.get("quotation_id")), None)
    if not quote:
        result = {
            "field": "quote",
            "expected": "valid quote",
            "actual": "none",
            "status": "Fail",
            "reason": f"No quote found in plan listing data for quotation_id '{proposal_data.get('quotation_id', 'unknown')}'. "
                      f"Expected a valid quote to proceed with validation."
        }
        logging.error(f"Validation failed: {result}")
        return [result]

    # Define context for NCB validation
    ncb_context = {
        "is_new_business": proposal_data.get("vehicle", {}).get("business_type") == "new_business",
        "is_ownership_transferred": proposal_data.get("vehicle", {}).get("is_ownership_transferred", False),
        "previous_policy_expiry_date": proposal_data.get("vehicle", {}).get("previous_policy_expiry_date"),
        "previous_policy_type": proposal_data.get("vehicle", {}).get("previous_policy_type"),
        "product_code": test_data.get("product_code", "")
    }
    ncb_applicable = (
        not ncb_context["is_new_business"] and
        not ncb_context["is_ownership_transferred"] and
        ncb_context["previous_policy_type"] != "third_party" and
        "COMPREHENSIVE" in ncb_context["product_code"] and
        (ncb_context["previous_policy_expiry_date"] is None or "recent" in str(ncb_context["previous_policy_expiry_date"]))
    )

    fields_to_validate = []
    
    # Validate addons
    addons_value = test_data.get("addons", "")
    expected_addons = []
    
    if isinstance(addons_value, float) and np.isnan(addons_value):
        expected_addons = []
    elif isinstance(addons_value, str) and addons_value:
        if addons_value.strip().startswith("[") and addons_value.strip().endswith("]"):
            try:
                addons_json = json.loads(addons_value)
                if isinstance(addons_json, list):
                    expected_addons = [
                        addon.get("insurance_cover_code")
                        for addon in addons_json
                        if isinstance(addon, dict) and "insurance_cover_code" in addon
                    ]
                    logging.debug(f"Parsed JSON addons: {expected_addons}")
                else:
                    logging.error(f"Invalid addons JSON format for testcase: {addons_value}")
                    results.append({
                        "field": "addons",
                        "expected": addons_value,
                        "actual": "",
                        "status": "Fail",
                        "reason": f"Invalid addons JSON format: expected a list of objects, got {type(addons_json).__name__}."
                    })
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse addons JSON: {str(e)}")
                results.append({
                    "field": "addons",
                    "expected": addons_value,
                    "actual": "",
                    "status": "Fail",
                    "reason": f"Failed to parse addons JSON: {str(e)}."
                })
        else:
            expected_addons = [addon.strip() for addon in addons_value.split(",") if addon.strip()]
            logging.debug(f"Parsed comma-separated addons: {expected_addons}")
    
    actual_addons = [addon["cover_code"] for addon in proposal_data.get("addons", []) if addon.get("selected")]
    mandatory_addons = [addon["cover_code"] for addon in proposal_data.get("addons", []) if not addon.get("is_removable", True)]
    
    if not expected_addons and actual_addons and all(addon in mandatory_addons for addon in actual_addons):
        results.append({
            "field": "addons",
            "expected": str(addons_value),
            "actual": str(actual_addons),
            "status": "Pass",
            "reason": f"No addons expected in test data ('{addons_value}'). Actual addons {actual_addons} are all mandatory "
                      f"({mandatory_addons}), which is valid as per rules allowing mandatory addons when none are specified."
        })
    elif results and results[-1].get("field") == "addons" and results[-1]["status"] == "Fail":
        pass
    else:
        fields_to_validate.append({
            "field": "addons",
            "expected": expected_addons,
            "actual": actual_addons,
            "context": {"mandatory_addons": mandatory_addons}
        })

    # Validate discounts
    discounts_value = test_data.get("discounts", "")
    allowed_discounts = ["ANTI_THEFT_DISCOUNT", "VOLUNTARY_DEDUCTIBLE_DISCOUNT", "TPPD_DISCOUNT"]
    expected_discounts = []
    
    if isinstance(discounts_value, float) and np.isnan(discounts_value):
        expected_discounts = []
    elif isinstance(discounts_value, str) and discounts_value:
        if discounts_value.strip().startswith("[") and discounts_value.strip().endswith("]"):
            try:
                discounts_json = json.loads(discounts_value)
                if isinstance(discounts_json, list):
                    if all(isinstance(item, str) and item.strip().startswith("{") for item in discounts_json):
                        try:
                            parsed_items = [json.loads(item) for item in discounts_json]
                            expected_discounts = [
                                item.get("discount_code")
                                for item in parsed_items
                                if isinstance(item, dict) and "discount_code" in item
                            ]
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse nested discounts JSON: {str(e)}")
                            results.append({
                                "field": "discounts",
                                "expected": discounts_value,
                                "actual": "",
                                "status": "Fail",
                                "reason": f"Failed to parse nested discounts JSON: {str(e)}."
                            })
                    else:
                        expected_discounts = [
                            item.get("discount_code")
                            for item in discounts_json
                            if isinstance(item, dict) and "discount_code" in item
                        ]
                    logging.debug(f"Parsed JSON discounts: {expected_discounts}")
                else:
                    logging.error(f"Invalid discounts JSON format for testcase: {discounts_value}")
                    results.append({
                        "field": "discounts",
                        "expected": discounts_value,
                        "actual": "",
                        "status": "Fail",
                        "reason": f"Invalid discounts JSON format: expected a list, got {type(discounts_json).__name__}."
                    })
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse discounts JSON: {str(e)}")
                results.append({
                    "field": "discounts",
                    "expected": discounts_value,
                    "actual": "",
                    "status": "Fail",
                    "reason": f"Failed to parse discounts JSON: {str(e)}."
                })
        else:
            expected_discounts = [discount.strip() for discount in discounts_value.split(",") if discount.strip()]
            logging.debug(f"Parsed comma-separated discounts: {expected_discounts}")
    
    actual_discounts = [d.get("code") for d in proposal_data.get("discounts", [])]
    
    # Validate discounts against allowed list
    invalid_discounts = [d for d in expected_discounts if d not in allowed_discounts]
    if invalid_discounts:
        results.append({
            "field": "discounts",
            "expected": str(discounts_value),
            "actual": str(actual_discounts),
            "status": "Fail",
            "reason": f"Invalid discounts in test data: {invalid_discounts}. Only {allowed_discounts} are allowed."
        })
    elif results and results[-1].get("field") == "discounts" and results[-1]["status"] == "Fail":
        pass
    else:
        missing_discounts = [d for d in expected_discounts if d not in actual_discounts]
        if not expected_discounts and (not actual_discounts or actual_discounts == ["NCB_DISCOUNT"]):
            results.append({
                "field": "discounts",
                "expected": str(discounts_value),
                "actual": str(actual_discounts),
                "status": "Pass",
                "reason": f"No discounts expected ('{discounts_value}'). Actual discounts {actual_discounts} "
                          f"{'are empty, which is valid.' if not actual_discounts else 'contain only NCB_DISCOUNT, which is allowed per rules.'}"
            })
        elif ncb_applicable and "NCB_DISCOUNT" not in actual_discounts:
            results.append({
                "field": "discounts",
                "expected": "NCB_DISCOUNT",
                "actual": str(actual_discounts),
                "status": "Fail",
                "reason": f"NCB_DISCOUNT expected for '{ncb_context['product_code']}' as NCB is applicable "
                          f"(new_business={ncb_context['is_new_business']}, "
                          f"ownership_transferred={ncb_context['is_ownership_transferred']}, "
                          f"policy_type={ncb_context['previous_policy_type']}). "
                          f"Actual discounts {actual_discounts} do not include NCB_DISCOUNT."
            })
        elif missing_discounts:
            results.append({
                "field": "discounts",
                "expected": str(expected_discounts),
                "actual": str(actual_discounts),
                "status": "Fail",
                "reason": f"Expected discounts {expected_discounts} not fully present in actual discounts {actual_discounts}. "
                          f"Missing discounts: {missing_discounts}."
            })
        else:
            results.append({
                "field": "discounts",
                "expected": str(expected_discounts),
                "actual": str(actual_discounts),
                "status": "Pass",
                "reason": f"Expected discounts {expected_discounts} are present in actual discounts {actual_discounts}. "
                          f"Extra discounts in actual are allowed."
            })

    # Validate IDV with range checking
    idv_value = test_data.get("idv", "")
    if isinstance(idv_value, float) and np.isnan(idv_value):
        idv_value = ""
    else:
        idv_value = str(idv_value)
    actual_idv = proposal_data.get("vehicle", {}).get("idv")
    idv_context = {
        "min_idv": quote.get("min_sum_insured") if quote else None,
        "max_idv": quote.get("max_sum_insured") if quote else None
    }
    null_values = [None, "", "None", "null", "nan", "NaN"]
    if idv_value in null_values or str(idv_value).lower() in null_values:
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
                        "reason": f"Expected IDV is empty ('{idv_value}'). Actual IDV {actual_idv} falls within the valid range "
                                  f"[{min_idv}, {max_idv}] from plan listing data, which is acceptable per validation rules."
                    })
                else:
                    results.append({
                        "field": "idv",
                        "expected": str(idv_value),
                        "actual": str(actual_idv),
                        "status": "Fail",
                        "reason": f"Expected IDV is empty ('{idv_value}'). Actual IDV {actual_idv} is outside the valid range "
                                  f"[{min_idv}, {max_idv}] from plan listing data, which is not acceptable."
                    })
            except (ValueError, TypeError):
                fields_to_validate.append({
                    "field": "idv",
                    "expected": idv_value,
                    "actual": actual_idv,
                    "context": idv_context
                })
        else:
            fields_to_validate.append({
                "field": "idv",
                "expected": idv_value,
                "actual": actual_idv,
                "context": idv_context
            })
    else:
        fields_to_validate.append({
            "field": "idv",
            "expected": idv_value,
            "actual": actual_idv,
            "context": idv_context
        })

    # Validate other fields
    fields_to_validate.append({
        "field": "previous_ncb",
        "expected": test_data.get("previous_ncb", ""),
        "actual": proposal_data.get("vehicle", {}).get("previous_policy_ncb"),
        "context": {"ncb_applicable": ncb_applicable}
    })

    previous_fields = [
        "previous_expiry_date", "previous_insurer", "previous_tp_expiry_date",
        "previous_tp_insurer", "claim_taken"
    ]
    for field in previous_fields:
        # Normalize field name for CSV lookup
        field_variants = [
            field,
            field.replace("_", " ").title().replace(" ", ""),
            field.replace("_", " ").upper(),
            field.replace("_", ""),
            field.lower(),
            field.upper()
        ]
        expected_value = None
        for variant in field_variants:
            if variant in test_data:
                expected_value = test_data[variant]
                logging.debug(f"Found {field} in test_data as '{variant}': {expected_value}")
                break
        if expected_value is None:
            expected_value = ""
            logging.warning(f"No matching key for {field} in test_data, using empty string")

        if field == "previous_insurer":
            api_field = "previous_policy_carrier_name"
        elif field == "previous_tp_insurer":
            api_field = "previous_tp_policy_carrier_name"
        elif field == "previous_tp_expiry_date":
            api_field = "previous_tp_policy_expiry_date"
        elif field == "claim_taken":
            api_field = "claim_taken"
        else:
            api_field = f"previous_policy_{field.replace('previous_', '')}"
        context = {}
        if field == "previous_tp_expiry_date":
            context = {
                "created_at": proposal_data.get("vehicle", {}).get("created_at"),
                "offset_days": test_data.get("offset_previous_tp_expiry_date", "")
            }
            logging.debug(f"Validating {field}: CSV expected={expected_value}, "
                         f"offset_days={context['offset_days']}, "
                         f"created_at={context['created_at']}")
            if not context.get("offset_days"):
                logging.warning(f"No offset_days provided for '{field}' in testcase {test_data.get('Testcase_id', 'unknown')}")
        elif field == "previous_expiry_date":
            context = {
                "created_at": proposal_data.get("vehicle", {}).get("created_at"),
                "offset_days": test_data.get("offset_previous_expiry_date", "")
            }
            logging.debug(f"Validating {field}: CSV expected={expected_value}, "
                         f"offset_days={context['offset_days']}, "
                         f"created_at={context['created_at']}")
            if not context.get("offset_days"):
                logging.warning(f"No offset_days provided for '{field}' in testcase {test_data.get('Testcase_id', 'unknown')}")
        fields_to_validate.append({
            "field": field,
            "expected": expected_value,
            "actual": proposal_data.get("vehicle", {}).get(api_field),
            "context": context
        })

    fields_to_validate.append({
        "field": "proposal_number",
        "expected": "non-empty",
        "actual": proposal_data.get("proposal_number"),
        "context": {}
    })

    fields_to_validate.append({
        "field": "is_break_in",
        "expected": test_data.get("is_inspection_required", "No") == "Yes",
        "actual": proposal_data.get("vehicle", {}).get("is_break_in"),
        "context": {}
    })

    # Parallelize field validations for efficiency
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_field = {executor.submit(validate_field, f["expected"], f["actual"], f["field"], f["context"]): f for f in fields_to_validate}
        for future in as_completed(future_to_field):
            result = future.result()
            if result:
                results.append(result)

    # Sort results by status: Fail, Pass, Pending
    results.sort(key=lambda x: {"Fail": 0, "Pass": 1, "Pending": 2}.get(x["status"], 2))

    logging.info(f"Quote validation completed in {time.time() - start_time:.2f} seconds")
    return results

@app.route('/')
def index():
    """Serve the index.html page."""
    logging.info("Serving index page")
    return send_file('index.html')

@app.route('/script.js')
def serve_script():
    """Serve the script.js file."""
    logging.info("Serving script.js")
    return send_file('script.js')

@app.route('/upload_csv', methods=['POST', 'OPTIONS'])
def upload_csv():
    """Handle CSV file uploads for input and output test data.

    I implemented this endpoint to securely store uploaded CSV files in temporary locations.
    """
    global INPUT_CSV_PATH, OUTPUT_CSV_PATH
    if request.method == 'OPTIONS':
        logging.info("Handling OPTIONS request for /upload_csv")
        return jsonify({}), 200

    logging.info("Received /upload_csv POST request")
    
    try:
        input_file = request.files.get("test_data_input")
        output_file = request.files.get("testcase_output")
        
        if not input_file and not output_file:
            logging.error("At least one CSV file is required")
            return jsonify({"error": "At least one CSV file is required"}), 400
        
        if input_file:
            input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            input_file.save(input_temp.name)
            pd.read_csv(input_temp.name, dtype=str, keep_default_na=False)  # Validate CSV
            INPUT_CSV_PATH = input_temp.name
            logging.info(f"Input CSV saved to {INPUT_CSV_PATH}")
        
        if output_file:
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            output_file.save(output_temp.name)
            pd.read_csv(output_temp.name, dtype=str, keep_default_na=False)  # Validate CSV
            OUTPUT_CSV_PATH = output_temp.name
            logging.info(f"Output CSV saved to {OUTPUT_CSV_PATH}")
        
        return jsonify({"message": "CSV(s) uploaded successfully"})
    except Exception as e:
        logging.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Upload error: {str(e)}"}), 400

@app.route('/validate', methods=['POST'])
def validate():
    """Validate test cases by comparing CSV data against API responses.

    I designed this endpoint to process uploaded CSVs, fetch API data, and return validation results.
    """
    global INPUT_CSV_PATH, OUTPUT_CSV_PATH
    logging.info("Received /validate request")
    start_time = time.time()
    
    if not INPUT_CSV_PATH or not OUTPUT_CSV_PATH:
        logging.error("Both CSVs are required for validation")
        return jsonify({"error": "Both CSVs are required for processing"}), 400
    
    try:
        logging.info(f"Reading input CSV: {INPUT_CSV_PATH}")
        input_df = pd.read_csv(INPUT_CSV_PATH, dtype=str, keep_default_na=False)
        logging.info(f"Reading output CSV: {OUTPUT_CSV_PATH}")
        output_df = pd.read_csv(OUTPUT_CSV_PATH, dtype=str, keep_default_na=False)
        
        results = []
        skipped = []
        
        # Merge input and output CSVs
        merged_df = input_df.merge(
            output_df,
            left_on="Testcase_id",
            right_on="TestcaseId",
            how="left"
        )
        logging.debug(f"Merged dataframe: {len(merged_df)} rows")
        
        # Log unmatched test cases
        unmatched = input_df[~input_df["Testcase_id"].isin(output_df["TestcaseId"])]
        if not unmatched.empty:
            logging.warning(f"Unmatched Testcase_ids: {unmatched['Testcase_id'].tolist()}")
        
        total_testcases = len(merged_df)
        processed_testcases = 0
        
        # Collect quote IDs for batch processing
        quote_ids = []
        testcase_data = []
        for idx, row in merged_df.iterrows():
            testcase_id = row.get("Testcase_id")
            if pd.isna(testcase_id):
                logging.error(f"Missing Testcase_id in row {idx}")
                skipped.append({"testcase_id": None, "reason": "Missing Testcase_id"})
                continue
            
            thank_url = row.get("ThankURL", "")
            quote_id, error = extract_quote_id(thank_url)
            if not quote_id:
                logging.debug(f"Skipping testcase_id {testcase_id} with ThankURL: {thank_url}, reason: {error}")
                skipped.append({"testcase_id": testcase_id, "reason": error or "Invalid ThankURL"})
                continue
            
            test_data = row.to_dict()
            quote_ids.append(quote_id)
            testcase_data.append((testcase_id, test_data))
            processed_testcases += 1
        
        # Fetch API data in batch
        api_results = fetch_api_data_batch(quote_ids)
        
        for testcase_id, test_data in testcase_data:
            quote_id = quote_ids[testcase_data.index((testcase_id, test_data))]
            if quote_id not in api_results:
                logging.error(f"No API data for quote_id {quote_id}, testcase_id {testcase_id}")
                skipped.append({"testcase_id": testcase_id, "reason": "Failed to fetch API data"})
                continue
            
            proposal_data, plan_listing_data = api_results[quote_id]
            validation_results = validate_quote(test_data, proposal_data, plan_listing_data)
            
            if validation_results and isinstance(validation_results, list):
                results.append({
                    "testcase_id": testcase_id,
                    "quote_id": quote_id,
                    "validation_results": validation_results
                })
            else:
                logging.error(f"Invalid validation results for testcase_id {testcase_id}")
                skipped.append({"testcase_id": testcase_id, "reason": "Invalid validation results"})
        
        # Clean up temporary files after successful validation
        try:
            if INPUT_CSV_PATH:
                os.unlink(INPUT_CSV_PATH)
                INPUT_CSV_PATH = None
            if OUTPUT_CSV_PATH:
                os.unlink(OUTPUT_CSV_PATH)
                OUTPUT_CSV_PATH = None
            logging.info("Temporary CSV files cleaned up")
        except Exception as e:
            logging.error(f"Failed to clean up temporary files: {str(e)}", exc_info=True)
        
        response = {
            "results": results,
            "skipped": skipped,
            "total_testcases": total_testcases,
            "processed_testcases": processed_testcases
        }
        logging.info(f"Validation completed in {time.time() - start_time:.2f} seconds with {len(results)} results, {len(skipped)} skipped")
        return jsonify(response)
    except (KeyError, TypeError, ValueError) as e:
        logging.error(f"Validation error: {str(e)}", exc_info=True)
        logging.debug(f"Current state: results={len(results)}, skipped={len(skipped)}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        logging.debug(f"Current state: results={len(results)}, skipped={len(skipped)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    """Run the Flask application."""
    port = int(os.getenv("PORT", 3000))
    app.run(debug=False, host="0.0.0.0", port=port)