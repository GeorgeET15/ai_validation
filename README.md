<img src="logo_with_bg.png" alt="Logo" width="50"/>

# Motor Quote Validator

A web-based application to validate motor insurance quotes by comparing expected values from a CSV file against actual values fetched from APIs. Built with Flask (backend), HTML/JavaScript (frontend), and integrated with an OpenAI model for complex field comparisons.

## Features

- **Quote Validation**: Validates fields like `idv`, `addons`, `discounts`, `previous_ncb`, `previous_expiry_date`, and `previous_tp_expiry_date` using test data from a CSV and API responses.
- **Minimalistic UI**: Clean, user-friendly interface for uploading CSV files uploading CSVs and viewing validation results.
- **Error Handling**: Displays errors via a custom dialog for invalid inputs, missing CSVs CSVs, or server issues issues.

## Prerequisites

- Python 3.9+
- An OpenAI API key (set in `.env`)
- Access to the Riskcovry APIs (URLs in `app.py`)
- Required Python packages (listed in `requirements.txt`)

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/GeorgeET15/ai_validation.git
   cd ai_validation
   ```

2. **Install Python Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure `requirements.txt` includes:

   ```
   flask
   flask-cors
   pandas
   requests
   langchain-openai
   python-dotenv
   python-dateutil
   cachetools
   ```

3. **Set Environment Variables**
   Create a `.env` file in the root directory:

   ```bash
   OPENAI_API_KEY=your-openai-api-key
   PORT=3000
   ```

4. **Prepare Test Data**
   Place input and output CSV files (e.g., `test_data_input.csv`, `testcase_output.csv`) in the root directory. Input CSV should include columns like `Testcase_id`, `idv`, `addons`, `discounts`, `previous_ncb`, `previous_tp_expiry_date`, `offset_previous_tp_expiry_date`, etc. Example:

   ```csv
   Testcase_id,ThankURL,idv,addons,discounts,previous_ncb,previous_tp_expiry_date,offset_previous_tp_expiry_date
   TC001,https://example.com/thank-you/quote_id=EDmfxVx5szLr-a4HpAdB,,ADDON1,NCB_DISCOUNT,20,05/02/2026,2
   ```

5. **Add Logo**
   Place a `logo.png` file in the root directory for the UI header.

## Running the Application

1. **Start the Flask Server**

   ```bash
   python app.py
   ```

   The server runs on `http://127.0.0.1:3000` (or the port specified in `.env`).

2. **Access the UI**
   Open a browser and navigate to `http://127.0.0.1:3000`. Alternatively, serve `index.html` via a local web server (e.g., `npx serve`).

3. **Validate Quotes**
   - Upload input and output CSV files via the UI.
   - Click "Validate" to process test cases and compare data.
   - View results in a grid, with color-coded status (green for Pass, red for Fail, yellow for Pending).

## File Structure

```
ai_validation/
├── app.py              # Flask backend with validation logic
├── index.html          # Minimalistic frontend UI
├── script.js           # Frontend JavaScript for form handling and result rendering
├── logo.png            # Logo for UI header
├── .env                # Environment variables (not included)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage Notes

- **CSV Format**: Input CSV must have a `Testcase_id` and `ThankURL` for quote ID extraction. Date fields like `previous_tp_expiry_date` support formats like `DD/MM/YYYY` or `DD Mon YYYY`.
- **Date Validation**: For `previous_tp_expiry_date` and `previous_expiry_date`:
  - Compares the CSV expected date (e.g., `05/02/2026`) and computed date (e.g., `created_at + offset_previous_tp_expiry_date`) with the actual date.
  - Passes if either matches, showing the matching date in the `expected` field (CSV date if it matches, else computed date).
  - Example: If CSV date `05/02/2026` matches actual `05 Feb 2026`, `expected` shows `05/02/2026`.
- **Error Dialog**: Closes via the "Close" button or Escape key.
- **API Endpoints**: Uses Riskcovry APIs (`motor/v2/plans/selected_plan_information`, `motor/fetch_quote_list`, `policies/get_policy`). Ensure network access and valid quote IDs.
- **IDV Validation**: Empty expected IDV passes if actual IDV is within the API’s range (`min_sum_insured` to `max_sum_insured`).
- **Discounts/Addons**: Supports JSON or comma-separated values in CSV (e.g., `[{"discount_code":"NCB_DISCOUNT"}]`, `ADDON1,ADDON2`).

## Troubleshooting

- **Server Errors**: Check Flask console logs (`DEBUG` level). Common issues:
  - Missing CSV files or invalid column names.
  - API failures (check network or quote ID validity).
- **Date Issues**:
  - Invalid date formats in CSV (e.g., `2026-13-01`): Ensure `DD/MM/YYYY` or `DD Mon YYYY`.
  - Invalid `offset_previous_tp_expiry_date` (e.g., `abc`): Must be an integer. Empty offsets fall back to CSV date comparison.
- **CORS Issues**: Flask allows all origins (`CORS *`). Verify frontend and backend are on the same domain or adjust CORS settings.
- **OpenAI API**: Ensure the API key in `.env` is valid and has quota.
- **No Results**: Confirm quote ID exists in the API and matches `plan_listing_data`. Check `ThankURL` format in CSV.
