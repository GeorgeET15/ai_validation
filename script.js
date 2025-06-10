const elements = {
  csvFile: document.getElementById("csvFile"),
  uploadCsv: document.getElementById("uploadCsv"),
  uploadText: document.getElementById("uploadText"),
  uploadLoader: document.getElementById("uploadLoader"),
  csvStatus: document.getElementById("csvStatus"),
  errorDialog: document.getElementById("errorDialog"),
  errorMessage: document.getElementById("errorMessage"),
  closeDialog: document.getElementById("closeDialog"),
  validateButton: document.getElementById("validateButton"),
  quoteForm: document.getElementById("quoteForm"),
  quoteId: document.getElementById("quoteId"),
  results: document.getElementById("results"),
  validationResults: document.getElementById("validationResults"),
  loading: document.getElementById("loading"),
};

const showError = (message) => {
  elements.errorMessage.textContent = message;
  elements.errorDialog.classList.remove("hidden");
};

const updateStatus = (message, isError = false) => {
  elements.csvStatus.textContent = message;
  elements.csvStatus.className = `mt-2 text-sm ${
    isError ? "text-red-600" : "text-green-600"
  }`;
};

elements.csvFile.addEventListener("change", () => {
  if (elements.csvFile.files.length) {
    updateStatus(`Selected: ${elements.csvFile.files[0].name}`);
  }
});

elements.uploadCsv.addEventListener("click", async () => {
  if (!elements.csvFile.files.length) {
    showError("Please select a CSV file.");
    return;
  }

  const formData = new FormData();
  formData.append("csvFile", elements.csvFile.files[0]);

  try {
    elements.uploadCsv.disabled = true;
    elements.uploadText.classList.add("hidden");
    elements.uploadLoader.classList.remove("hidden");
    updateStatus("Uploading...", false);

    const response = await fetch("http://127.0.0.1:3000/upload_csv", {
      method: "POST",
      body: formData,
    });

    if (!response.ok)
      throw new Error((await response.text()) || response.statusText);
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error("Invalid server response.");
    }

    const { error } = await response.json();
    if (error) throw new Error(error);

    updateStatus("Upload successful.", false);
    elements.uploadCsv.disabled = true;
  } catch (error) {
    showError(`Upload error: ${error.message}`);
    updateStatus("Upload failed.", true);
  } finally {
    elements.uploadText.classList.remove("hidden");
    elements.uploadLoader.classList.add("hidden");
  }
});

elements.closeDialog.addEventListener("click", () => {
  elements.errorDialog.classList.add("hidden");
});

document.addEventListener("keydown", ({ key }) => {
  if (key === "Escape" && !elements.errorDialog.classList.contains("hidden")) {
    elements.errorDialog.classList.add("hidden");
  }
});

elements.quoteForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const quoteId = elements.quoteId.value.trim();

  if (!quoteId) {
    showError("Please enter a Quote ID");
    return;
  }

  elements.validationResults.innerHTML = "";
  elements.results.classList.add("hidden");
  elements.loading.classList.remove("hidden");
  elements.validateButton.disabled = true;

  try {
    const response = await fetch("http://127.0.0.1:3000/validate", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ quote_id: quoteId }),
    });

    if (!response.ok)
      throw new Error((await response.text()) || "Server error");
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error("Invalid server response.");
    }

    const { validation_results } = await response.json();
    elements.results.classList.remove("hidden");

    if (!validation_results?.length) {
      elements.validationResults.innerHTML = `
        <div class="col-span-full text-center text-gray-500 text-sm py-4">
          No validation results found.
        </div>`;
      return;
    }

    validation_results.forEach(
      ({ status, field, expected, actual, reason, is_mandatory }) => {
        const statusStyles =
          {
            Pass: "bg-green-500 text-green-50 border-green-400",
            Fail: "bg-red-500 text-red-50 border-red-400",
            Pending: "bg-yellow-500 text-yellow-50 border-yellow-400",
          }[status] || "bg-green-500 text-green-50 border-green-400";

        const card = document.createElement("div");
        card.className = `relative bg-white border rounded-xl shadow-sm p-4 mb-1 w-full max-w-md`;

        card.innerHTML = `
        <div class="absolute top-0 left-0 right-0 ${statusStyles} text-xs font-medium text-center py-1 rounded-t-xl">
          ${status}
        </div>
        <div class="pt-6 pb-2">
          <div class="mb-3">
            <h3 class="text-base font-semibold text-gray-800">${field}</h3>
          </div>
          <div class="space-y-1 text-sm text-gray-600">
            <p><span class="font-medium text-gray-700">Expected:</span> ${expected}</p>
            <p><span class="font-medium text-gray-700">Actual:</span> ${actual}</p>
            <p><span class="font-medium text-gray-700">Reason:</span> ${reason}</p>
            ${
              is_mandatory
                ? `<p><span class="font-medium text-gray-700">Mandatory:</span> Yes</p>`
                : ""
            }
          </div>
        </div>
      `;
        elements.validationResults.appendChild(card);
      }
    );
  } catch (error) {
    showError(`Error: ${error.message}`);
  } finally {
    elements.loading.classList.add("hidden");
    elements.validateButton.disabled = false;
  }
});
