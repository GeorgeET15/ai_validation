# Motor Quote Validator

A web-based application to validate motor insurance quotes by comparing expected values from a CSV file against actual values fetched from APIs. Built with Flask (backend), HTML/JavaScript with Tailwind CSS (frontend), and integrated with an OpenAI model for field comparison.

## Features

- **Quote Validation**: Validates fields like IDV, addons, discounts, and NCB using test data from a CSV and API responses.
- **Minimalistic UI**: Clean, user-friendly interface for entering Quote IDs and viewing validation results.
- **Error Handling**: Displays errors via a custom dialog for invalid inputs or server issues.

## Prerequisites

- Python 3.8+
- Node.js (optional, for local development with a web server)
- An OpenAI API key (set in `.env`)
- Access to the Riskcovry APIs (provided URLs in `app.py`)

## Setup

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd motor-quote-validator
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
   ```

3. **Set Environment Variables**
   Create a `.env` file in the root directory:

   ```bash
   OPENAI_API_KEY=your-openai-api-key
   PORT=3000
   ```

4. **Prepare Test Data**
   Place a `test_data.csv` file in the root directory with columns like `testcase_id`, `idv`, `addons`, `discounts`, `previous_ncb`, etc. Example:

   ```csv
   testcase_id,idv,addons,discounts,previous_ncb
   TC001,,ADDON1,NCB_DISCOUNT,20
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

3. **Validate a Quote**
   - Enter a Quote ID (e.g., `EDmfxVx5szLr-a4HpAdB`) in the input field.
   - Click "Validate" to fetch and compare data.
   - View results in a grid, with color-coded status (green for Pass, red for Fail, yellow for Pending).

## File Structure

```
motor-quote-validator/
├── app.py              # Flask backend with validation logic
├── index.html          # Minimalistic frontend UI
├── script.js           # Frontend JavaScript for form handling and result rendering
├── test_data.csv       # Test data for validation (not included)
├── logo.png            # Logo for UI header (not included)
├── .env                # Environment variables (not included)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage Notes

- **Quote ID Format**: Must be alphanumeric with dashes (e.g., `EDmfxVx5szLr-a4HpAdB`). Invalid formats trigger an error dialog.
- **Error Dialog**: Closes via the "Close" button or Escape key.
- **API Endpoints**: The app uses Riskcovry APIs (`motor/v2/plans/selected_plan_information` and `motor/fetch_quote_list`). Ensure network access and valid Quote IDs.
- **IDV Validation**: Empty expected IDV passes if the actual IDV is within the API-provided range (`min_sum_insured` to `max_sum_insured`).

## Troubleshooting

- **Server Errors**: Check the Flask console for logs (set to `DEBUG` level). Common issues include missing `test_data.csv` or invalid API responses.
- **CORS Issues**: Ensure the Flask server allows the frontend origin (`CORS` is set to `*` by default).
- **OpenAI API**: Verify the API key in `.env` is valid and has sufficient quota.
- **No Results**: Ensure the Quote ID exists in the Riskcovry API and matches a quote in `plan_listing_data`.
