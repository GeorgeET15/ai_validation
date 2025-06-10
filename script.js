const csvFileInput = document.getElementById("csvFile");
const uploadCsvButton = document.getElementById("uploadCsv");
const uploadText = document.getElementById("uploadText");
const uploadLoader = document.getElementById("uploadLoader");
const csvStatus = document.getElementById("csvStatus");
const errorDialog = document.getElementById("errorDialog");
const errorMessage = document.getElementById("errorMessage");
const closeDialog = document.getElementById("closeDialog");
const validateButton = document.getElementById("validateButton");

function showErrorDialog(message) {
  errorMessage.textContent = message;
  errorDialog.classList.remove("hidden");
}

function updateCsvStatus(message, isError = false) {
  csvStatus.textContent = message;
  csvStatus.className = `mt-2 text-sm ${
    isError ? "text-[#e5251f]" : "text-green-600"
  }`;
}

csvFileInput.addEventListener("change", () => {
  if (csvFileInput.files.length > 0) {
    updateCsvStatus("CSV file selected: " + csvFileInput.files[0].name);
  }
});

uploadCsvButton.addEventListener("click", async () => {
  const formData = new FormData();

  if (csvFileInput.files.length > 0) {
    formData.append("csvFile", csvFileInput.files[0]);
  } else {
    showErrorDialog("Please select a test data CSV file.");
    return;
  }

  try {
    uploadCsvButton.disabled = true;
    uploadText.classList.add("hidden");
    uploadLoader.classList.remove("hidden");
    updateCsvStatus("Uploading test data...", false);
    const response = await fetch("http://127.0.0.1:3000/upload_csv", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("Server response:", text);
      showErrorDialog(`Failed to upload CSV: ${text || response.statusText}`);
      updateCsvStatus("Upload failed.", true);
      return;
    }

    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Non-JSON response:", text);
      showErrorDialog("Server returned an invalid response.");
      updateCsvStatus("Upload failed.", true);
      return;
    }

    const result = await response.json();
    if (result.error) {
      showErrorDialog(result.error);
      updateCsvStatus("Upload failed.", true);
      return;
    }

    updateCsvStatus("Test data uploaded successfully.", false);
  } catch (error) {
    console.error("Upload error:", error);
    showErrorDialog(`Upload error: ${error.message}`);
    updateCsvStatus("Upload failed.", true);
  } finally {
    uploadCsvButton.disabled = false;
    uploadText.classList.remove("hidden");
    uploadLoader.classList.add("hidden");
  }
});

closeDialog.addEventListener("click", () => {
  errorDialog.classList.add("hidden");
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !errorDialog.classList.contains("hidden")) {
    errorDialog.classList.add("hidden");
  }
});

document.getElementById("quoteForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  console.log("Form submitted");

  const quoteId = document.getElementById("quoteId").value.trim();
  console.log("Quote ID:", quoteId);

  if (!quoteId) {
    showErrorDialog("Please enter a Quote ID");
    return;
  }

  const resultsSection = document.getElementById("results");
  const validationResults = document.getElementById("validationResults");
  const loading = document.getElementById("loading");

  validationResults.innerHTML = "";
  resultsSection.classList.add("hidden");
  loading.classList.remove("hidden");
  validateButton.disabled = true;

  try {
    console.log("Sending fetch to http://127.0.0.1:3000/validate");
    const response = await fetch("http://127.0.0.1:3000/validate", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({ quote_id: quoteId }),
    });

    console.log("Response status:", response.status);

    if (!response.ok) {
      const text = await response.text();
      console.error("Server error:", text);
      showErrorDialog(`Server error: ${text || "Unknown server error"}`);
      return;
    }

    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await response.text();
      console.error("Non-JSON response:", text);
      showErrorDialog("Server returned an invalid response.");
      return;
    }

    const data = await response.json();
    console.log("Response data:", data);

    resultsSection.classList.remove("hidden");

    if (!data.validation_results || data.validation_results.length === 0) {
      validationResults.innerHTML = `
        <div class="col-span-full text-center text-gray-600 text-sm py-4">
          No validation results found.
        </div>`;
      return;
    }

    data.validation_results.forEach((result) => {
      const statusColor =
        {
          Pass: "bg-[#28a745]",
          Fail: "bg-[#e5251f]",
          Pending: "bg-[#ffc107]",
        }[result.status] || "bg-[#28a745]";

      const card = document.createElement("div");
      card.className = `p-2 bg-white border-l-4 rounded-md shadow-sm hover:shadow-md transition-shadow text-[#3123a8]`;

      card.innerHTML = `
        <div class="flex items-center gap-2 mb-1">
          <span class="w-2 h-2 rounded-full ${statusColor}"></span>
          <h3 class="text-xs font-semibold">${result.field}</h3>
        </div>
        <p class="text-xs mb-0.5"><span class="font-medium">Expected:</span> ${
          result.expected
        }</p>
        <p class="text-xs mb-0.5"><span class="font-medium">Actual:</span> ${
          result.actual
        }</p>
        <p class="text-xs mb-0.5"><span class="font-medium">Status:</span> ${
          result.status
        }</p>
        <p class="text-xs mb-0.5"><span class="font-medium">Reason:</span> ${
          result.reason
        }</p>
        ${
          result.is_mandatory
            ? `<p class="text-xs"><span class="font-medium">Mandatory:</span> Yes</p>`
            : ""
        }
      `;

      validationResults.appendChild(card);
    });
  } catch (error) {
    console.error("Fetch error:", error);
    showErrorDialog(`Network error: ${error.message}`);
  } finally {
    loading.classList.add("hidden");
    validateButton.disabled = false;
  }
});
