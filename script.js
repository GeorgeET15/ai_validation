const elements = {
  inputCsvFile: document.getElementById("inputCsvFile"),
  outputCsvFile: document.getElementById("outputCsvFile"),
  inputCsvStatus: document.getElementById("inputCsvStatus"),
  outputCsvStatus: document.getElementById("outputCsvStatus"),
  validateAllButton: document.getElementById("validateAllButton"),
  validateText: document.getElementById("validateText"),
  quoteIdScroll: document.getElementById("quoteIdScroll"),
  validationResults: document.getElementById("validationResults"),
  skippedTestCases: document.getElementById("skippedTestCases"),
  loading: document.getElementById("loading"),
  errorDialog: document.getElementById("errorDialog"),
  errorMessage: document.getElementById("errorMessage"),
  closeDialog: document.getElementById("closeDialog"),
  validationStatus: document.getElementById("validationStatus"),
};

// Verify all elements exist
for (const [key, el] of Object.entries(elements)) {
  if (!el) console.error(`Element ${key} not found in DOM`);
}

const showError = (message) => {
  console.error("Error displayed:", message);
  if (elements.errorMessage) elements.errorMessage.textContent = message;
  if (elements.errorDialog) {
    elements.errorDialog.classList.remove("hidden");
    elements.errorDialog.classList.add("modal-enter-active");
    elements.errorDialog.classList.remove("modal-enter");
    setTimeout(
      () => elements.errorDialog.classList.remove("modal-enter-active"),
      200
    );
  }
};

const updateStatus = (element, message, isError = false) => {
  if (element) {
    element.innerHTML = isError ? `❌ ${message}` : `✅ ${message}`;
    element.className = `mt-2 text-sm flex items-center gap-1 ${
      isError ? "text-red-600" : "text-green-600"
    }`;
  }
};

if (elements.inputCsvFile) {
  elements.inputCsvFile.addEventListener("change", () => {
    if (elements.inputCsvFile.files.length) {
      updateStatus(
        elements.inputCsvStatus,
        `Selected: ${elements.inputCsvFile.files[0].name}`
      );
    }
  });
}

if (elements.outputCsvFile) {
  elements.outputCsvFile.addEventListener("change", () => {
    if (elements.outputCsvFile.files.length) {
      updateStatus(
        elements.outputCsvStatus,
        `Selected: ${elements.outputCsvFile.files[0].name}`
      );
    }
  });
}

const uploadCsv = async (fileInput, statusElement, fileType) => {
  if (!fileInput.files.length) {
    showError(`Please select a ${fileType} CSV file.`);
    return;
  }

  // New: Validate file extension
  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showError(`Please select a valid .csv file for ${fileType}.`);
    updateStatus(statusElement, "Invalid file type.", true);
    return;
  }

  const formData = new FormData();
  formData.append(
    fileType === "input" ? "test_data_input" : "testcase_output",
    file
  );

  try {
    updateStatus(statusElement, "Uploading...", false);
    const response = await fetch("http://127.0.0.1:3000/upload_csv", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || response.statusText);
    }
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error("Invalid server response.");
    }

    const { error, message } = await response.json();
    if (error) throw new Error(error);

    updateStatus(statusElement, message, false);
    updateValidateButton();
  } catch (error) {
    showError(`Upload error: ${error.message}`);
    updateStatus(statusElement, "Upload failed.", true);
    updateValidateButton();
  }
};

if (elements.inputCsvFile) {
  elements.inputCsvFile.addEventListener("change", () =>
    uploadCsv(elements.inputCsvFile, elements.inputCsvStatus, "input")
  );
}
if (elements.outputCsvFile) {
  elements.outputCsvFile.addEventListener("change", () =>
    uploadCsv(elements.outputCsvFile, elements.outputCsvStatus, "output")
  );
}

const updateValidateButton = () => {
  if (
    elements.inputCsvStatus?.textContent.includes("successfully") &&
    elements.outputCsvStatus?.textContent.includes("successfully")
  ) {
    if (elements.validateAllButton) {
      elements.validateAllButton.disabled = false;
      elements.validateAllButton.classList.add("hover:animate-pulse");
    }
  } else {
    if (elements.validateAllButton) {
      elements.validateAllButton.disabled = true;
      elements.validateAllButton.classList.remove("hover:animate-pulse");
    }
  }
};

if (elements.closeDialog) {
  elements.closeDialog.addEventListener("click", () => {
    if (elements.errorDialog) {
      elements.errorDialog.classList.add("modal-exit");
      elements.errorDialog.classList.remove("modal-exit-active");
      setTimeout(() => {
        elements.errorDialog.classList.add("hidden");
        elements.errorDialog.classList.remove("modal-exit");
      }, 200);
    }
  });
}

if (document) {
  document.addEventListener("keydown", ({ key }) => {
    if (
      key === "Escape" &&
      elements.errorDialog &&
      !elements.errorDialog.classList.contains("hidden")
    ) {
      elements.errorDialog.classList.add("modal-exit");
      setTimeout(() => {
        elements.errorDialog.classList.add("hidden");
        elements.errorDialog.classList.remove("modal-exit");
      }, 200);
    }
  });
}

let allValidationResults = [];

const escapeHtml = (str) => {
  if (str == null) return "";
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
};

if (elements.validateAllButton) {
  elements.validateAllButton.addEventListener("click", async () => {
    console.log("Validate clicked");
    if (elements.validationResults) elements.validationResults.innerHTML = "";
    if (elements.quoteIdScroll) elements.quoteIdScroll.innerHTML = "";
    if (elements.skippedTestCases) elements.skippedTestCases.innerHTML = "";
    if (elements.validationStatus) {
      elements.validationStatus.classList.add("fade-exit");
      setTimeout(() => {
        elements.validationStatus.textContent = "";
        elements.validationStatus.classList.add("hidden");
        elements.validationStatus.classList.remove("fade-exit");
      }, 300);
    }
    if (elements.loading) {
      elements.loading.classList.add("fade-enter");
      elements.loading.classList.remove("hidden");
      setTimeout(() => elements.loading.classList.add("fade-enter-active"), 10);
    }
    if (elements.validateAllButton) elements.validateAllButton.disabled = true;

    try {
      console.log("Sending /validate request");
      const response = await fetch("http://127.0.0.1:3000/validate", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });

      console.log("Received /validate response, status:", response.status);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Server error");
      }
      if (!response.headers.get("content-type")?.includes("application/json")) {
        throw new Error("Invalid server response.");
      }

      const data = await response.json();
      console.log("Parsed /validate response:", data);

      // New: Check for empty or malformed response
      if (!data || typeof data !== "object") {
        throw new Error("Empty or malformed server response");
      }

      if (data.error) {
        throw new Error(data.error);
      }

      const {
        results = [],
        skipped = [],
        total_testcases = 0,
        processed_testcases = 0,
      } = data;
      console.log(
        `Results: ${results.length}, Skipped: ${skipped.length}, Total: ${total_testcases}, Processed: ${processed_testcases}`
      );

      // New: Group skipped test cases by reason for logging
      const skippedByReason = skipped.reduce((acc, item) => {
        const reason = item.reason || "Unknown";
        acc[reason] = acc[reason] || [];
        acc[reason].push(item.testcase_id);
        return acc;
      }, {});
      console.log("Skipped test cases by reason:", skippedByReason);

      allValidationResults = results;

      // Update validation status with skip info
      try {
        if (elements.validationStatus) {
          const skipCount = total_testcases - processed_testcases;
          elements.validationStatus.textContent = `${processed_testcases} out of ${total_testcases} test cases validated${
            skipCount > 0 ? ` (${skipCount} skipped)` : ""
          }`;
          elements.validationStatus.classList.remove("hidden");
          elements.validationStatus.classList.add("fade-enter");
          setTimeout(
            () => elements.validationStatus.classList.add("fade-enter-active"),
            10
          );
        }
      } catch (err) {
        console.error("Error updating validation status:", err);
        showError("Failed to update validation status");
      }

      // Populate testcase ID scroll bar
      let testcaseIds = [];
      try {
        testcaseIds = [
          ...new Set(
            results
              .map((result) => result.testcase_id)
              .filter((id) => id && typeof id === "string")
          ),
        ];
        console.log("Populating testcase IDs:", testcaseIds);
        if (elements.quoteIdScroll) {
          elements.quoteIdScroll.innerHTML =
            testcaseIds.length > 0
              ? testcaseIds
                  .map(
                    (testcaseId, index) => `
                        <button class="quote-id-button snap-center bg-blue-900 text-white px-4 py-2 rounded-lg hover:bg-red-500 transition-all transform hover:scale-105 ${
                          index === 0 ? "active" : ""
                        }" data-testcase-id="${testcaseId}">
                            ${escapeHtml(testcaseId)}
                        </button>
                    `
                  )
                  .join("")
              : "<p class='text-gray-500 text-sm'>No testcase IDs available</p>";
          elements.quoteIdScroll.scrollTo({ behavior: "smooth" });
        }
      } catch (err) {
        console.error("Error populating testcase IDs:", err);
        showError("Failed to render testcase IDs");
      }

      // Display skipped test cases with enhanced UI
      try {
        console.log("Populating skipped test cases:", skipped);
        if (elements.skippedTestCases && skipped.length > 0) {
          const unmatchedSkips = skipped.filter((item) =>
            item.reason.includes("No matching")
          );
          const otherSkips = skipped.filter(
            (item) => !item.reason.includes("No matching")
          );
          const hasSkips = skipped.length > 0;
          elements.skippedTestCases.innerHTML = `
                        <div class="bg-white border rounded-2xl shadow-md p-5">
                            <h3 class="text-lg font-semibold text-blue-900 mb-3 flex items-center">
                                Skipped Test Cases (${skipped.length})
                                <button id="toggleSkipped" class="ml-2 text-sm text-blue-600 hover:text-blue-800 focus:outline-none">
                                    ${hasSkips ? "Hide" : "Show"}
                                </button>
                            </h3>
                            <div id="skippedContent" class="${
                              hasSkips ? "" : "hidden"
                            }">
                                ${
                                  unmatchedSkips.length > 0
                                    ? `
                                        <h4 class="text-sm font-medium text-blue-900 mb-2">Unmatched Test Cases</h4>
                                        <ul class="list-disc pl-5 text-yellow-600 space-y-2 mb-4">
                                            ${unmatchedSkips
                                              .map(
                                                (item) =>
                                                  `<li class="text-sm">${escapeHtml(
                                                    item.testcase_id ||
                                                      "Unknown"
                                                  )}: ${escapeHtml(
                                                    item.reason || "No reason"
                                                  )}</li>`
                                              )
                                              .join("")}
                                        </ul>
                                    `
                                    : ""
                                }
                                ${
                                  otherSkips.length > 0
                                    ? `
                                        <h4 class="text-sm font-medium text-blue-900 mb-2">Other Skipped Test Cases</h4>
                                        <ul class="list-disc pl-5 text-gray-600 space-y-2">
                                            ${otherSkips
                                              .map(
                                                (item) =>
                                                  `<li class="text-sm">${escapeHtml(
                                                    item.testcase_id ||
                                                      "Unknown"
                                                  )}: ${escapeHtml(
                                                    item.reason || "No reason"
                                                  )}</li>`
                                              )
                                              .join("")}
                                        </ul>
                                    `
                                    : ""
                                }
                            </div>
                        </div>
                    `;
          // Add toggle functionality
          const toggleButton = document.getElementById("toggleSkipped");
          const skippedContent = document.getElementById("skippedContent");
          if (toggleButton && skippedContent) {
            toggleButton.addEventListener("click", () => {
              skippedContent.classList.toggle("hidden");
              toggleButton.textContent = skippedContent.classList.contains(
                "hidden"
              )
                ? "Show"
                : "Hide";
            });
          }
        } else if (elements.skippedTestCases) {
          elements.skippedTestCases.innerHTML = `
                        <div class="bg-white border rounded-2xl shadow-md p-5">
                            <h3 class="text-lg font-semibold text-blue-900 mb-3">Skipped Test Cases (0)</h3>
                            <p class="text-gray-500 text-sm">No test cases were skipped.</p>
                        </div>
                    `;
        }
      } catch (err) {
        console.error("Error populating skipped test cases:", err);
        showError("Failed to render skipped test cases");
      }

      // Add click listeners to testcase ID buttons
      try {
        if (elements.quoteIdScroll) {
          elements.quoteIdScroll
            .querySelectorAll(".quote-id-button")
            .forEach((button) => {
              button.addEventListener("click", () => {
                const testcaseId = button.dataset.testcaseId;
                console.log("Displaying results for testcase ID:", testcaseId);
                document
                  .querySelectorAll(".quote-id-button")
                  .forEach((btn) => btn.classList.remove("active"));
                button.classList.add("active");
                displayResultsForTestcaseId(testcaseId);
                button.scrollIntoView({ behavior: "smooth", inline: "center" });
              });
            });
        }
      } catch (err) {
        console.error("Error adding testcase ID button listeners:", err);
        showError("Failed to add testcase ID listeners");
      }

      // Display results for the first testcase_id by default
      try {
        if (testcaseIds.length > 0) {
          console.log(
            "Displaying default results for testcase ID:",
            testcaseIds[0]
          );
          displayResultsForTestcaseId(testcaseIds[0]);
        } else {
          console.log("No valid results to display");
          if (elements.validationResults) {
            elements.validationResults.innerHTML = `
                            <div class="col-span-full text-center text-gray-500 text-sm py-4">
                                No valid results found.
                            </div>`;
          }
        }
      } catch (err) {
        console.error("Error displaying default results:", err);
        showError("Failed to render default results");
      }
    } catch (error) {
      console.error("Validation error:", error.message, error.stack);
      showError(`Error: ${error.message}`);
      if (elements.validationStatus) {
        elements.validationStatus.textContent =
          "Validation failed due to an error";
        elements.validationStatus.classList.remove("hidden");
        elements.validationStatus.classList.add("fade-enter");
        setTimeout(
          () => elements.validationStatus.classList.add("fade-enter-active"),
          10
        );
      }
      if (elements.validationResults) {
        elements.validationResults.innerHTML = `
                    <div class="col-span-full text-center text-gray-500 text-sm py-4">
                        Failed to load results due to an error.
                    </div>`;
      }
    } finally {
      console.log("Hiding loading screen");
      if (elements.loading) {
        elements.loading.classList.add("fade-exit");
        setTimeout(() => {
          elements.loading.classList.add("hidden");
          elements.loading.classList.remove("fade-exit");
        }, 300);
      }
      if (elements.validateAllButton)
        elements.validateAllButton.disabled = false;
    }
  });
}

const displayResultsForTestcaseId = (testcaseId) => {
  try {
    const filteredResults = allValidationResults.filter(
      (result) => result.testcase_id === testcaseId
    );

    filteredResults.sort((a, b) => {
      const hasFailA =
        Array.isArray(a.validation_results) &&
        a.validation_results.some((vr) => vr.status === "Fail");
      const hasFailB =
        Array.isArray(b.validation_results) &&
        b.validation_results.some((vr) => vr.status === "Fail");
      return hasFailB - hasFailA;
    });

    console.log(
      `Rendering results for testcase ID ${testcaseId}:`,
      filteredResults
    );
    if (elements.validationResults) {
      elements.validationResults.innerHTML =
        filteredResults.length > 0
          ? filteredResults
              .map((result) => {
                const validationResults = Array.isArray(
                  result.validation_results
                )
                  ? result.validation_results
                  : [];
                return `
                    <div class="bg-white border rounded-2xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300">
                        <h3 class="text-base font-semibold text-blue-900 mb-2">Testcase ID: ${escapeHtml(
                          result.testcase_id || "Unknown"
                        )}</h3>
                        <h4 class="text-sm font-medium text-blue-900 mb-3">Quote ID: ${escapeHtml(
                          result.quote_id || "Unknown"
                        )}</h4>
                        <div class="space-y-3">
                            ${validationResults
                              .map(
                                ({
                                  status,
                                  field,
                                  expected,
                                  actual,
                                  reason,
                                  is_mandatory,
                                  range_validated,
                                }) => `
                                <div class="relative bg-gray-50 border rounded-lg p-4">
                                    <div class="absolute top-0 left-0 right-0 text-xs font-medium text-center py-1 rounded-t-lg ${
                                      status === "Pass"
                                        ? "bg-green-100 text-green-800 border-green-200"
                                        : status === "Fail"
                                        ? "bg-red-100 text-red-800 border-red-200"
                                        : "bg-yellow-100 text-yellow-800 border-yellow-200"
                                    }">
                                        ${escapeHtml(status || "Unknown")}
                                    </div>
                                    <div class="pt-6 text-sm text-gray-600 space-y-1.5">
                                        <p><span class="font-medium text-gray-700">Field:</span> ${escapeHtml(
                                          field || "N/A"
                                        )}</p>
                                        <p><span class="font-medium text-gray-700">Expected:</span> ${escapeHtml(
                                          String(expected ?? "N/A")
                                        )}</p>
                                        <p><span class="font-medium text-gray-700">Actual:</span> ${escapeHtml(
                                          String(actual ?? "N/A")
                                        )}</p>
                                        <p><span class="font-medium text-gray-700">Reason:</span> ${escapeHtml(
                                          reason || "No reason provided"
                                        )}</p>
                                        ${
                                          is_mandatory
                                            ? `<p><span class="font-medium text-gray-700">Mandatory:</span> Yes</p>`
                                            : ""
                                        }
                                        ${
                                          range_validated
                                            ? `<p><span class="font-medium text-gray-700">Range Validated:</span> Yes</p>`
                                            : ""
                                        }
                                    </div>
                                </div>
                            `
                              )
                              .join("")}
                        </div>
                    </div>
                `;
              })
              .join("")
          : `
                <div class="col-span-full text-center text-gray-500 text-sm py-4">
                    No results found for this testcase ID.
                </div>`;
    }
  } catch (err) {
    console.error(
      `Error rendering results for testcase ID ${testcaseId}:`,
      err
    );
    showError("Failed to render validation results");
    if (elements.validationResults) {
      elements.validationResults.innerHTML = `
                <div class="col-span-full text-center text-gray-500 text-sm py-4">
                    Failed to render results for this testcase ID.
                </div>`;
    }
  }
};
