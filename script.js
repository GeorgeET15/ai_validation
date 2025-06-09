document.getElementById("quoteForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  console.log("Form submitted");

  const quoteId = document.getElementById("quoteId").value.trim();
  console.log("Quote ID:", quoteId);

  const errorDialog = document.getElementById("errorDialog");
  const errorMessage = document.getElementById("errorMessage");
  const closeDialog = document.getElementById("closeDialog");

  // Function to show error dialog
  function showErrorDialog(message) {
    errorMessage.textContent = message;
    errorDialog.classList.remove("hidden");
  }

  // Close dialog on button click
  closeDialog.addEventListener("click", () => {
    errorDialog.classList.add("hidden");
  });

  // Close dialog on Escape key press
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !errorDialog.classList.contains("hidden")) {
      errorDialog.classList.add("hidden");
    }
  });

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
      const errorData = await response.json();
      console.error("Server error:", errorData);
      showErrorDialog(errorData.error || "Unknown server error");
      loading.classList.add("hidden");
      return;
    }

    const data = await response.json();
    console.log("Response data:", data);

    resultsSection.classList.remove("hidden");
    loading.classList.add("hidden");

    if (!data.validation_results || data.validation_results.length === 0) {
      validationResults.innerHTML = `
        <div class="col-span-full text-center text-gray-600">
          No validation results found.
        </div>`;
      return;
    }

    data.validation_results.forEach((result) => {
      const statusColor =
        {
          Pass: "bg-green-100 text-green-800 border-green-400",
          Fail: "bg-red-100 text-red-800 border-red-400",
          Pending: "bg-yellow-100 text-yellow-800 border-yellow-400",
        }[result.status] || "bg-gray-100 text-gray-800 border-gray-300";

      const card = document.createElement("div");
      card.className = `border-l-4 p-4 rounded-xl shadow-sm ${statusColor} bg-white`;

      card.innerHTML = `
        <h3 class="text-lg font-medium text-gray-800 mb-1">${result.field}</h3>
        <p class="text-sm text-gray-600"><strong>Expected:</strong> ${
          result.expected
        }</p>
        <p class="text-sm text-gray-600"><strong>Actual:</strong> ${
          result.actual
        }</p>
        <p class="text-sm text-gray-600"><strong>Status:</strong> ${
          result.status
        }</p>
        <p class="text-sm text-gray-600"><strong>Reason:</strong> ${
          result.reason
        }</p>
        ${
          result.is_mandatory
            ? `<p class="text-sm text-gray-600"><strong>Mandatory:</strong> Yes</p>`
            : ""
        }
      `;

      validationResults.appendChild(card);
    });
  } catch (error) {
    console.error("Fetch error:", error);
    showErrorDialog(`Network error: ${error.message}`);
    loading.classList.add("hidden");
  }
});
