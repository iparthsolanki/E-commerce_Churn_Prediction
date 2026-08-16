"use strict";

/* ==========================================================
   CONFIG
========================================================== */
const API_URL = "http://127.0.0.1:8000/api/v1/predict";
const HEALTH_URL = "http://127.0.0.1:8000/health";

/* ==========================================================
   DOM REFERENCES
========================================================== */
const form = document.getElementById("churnForm");
const predictBtn = document.getElementById("predictBtn");
const predictBtnLabel = document.getElementById("predictBtnLabel");
const formAlert = document.getElementById("formAlert");

const resultEmpty = document.getElementById("resultEmpty");
const resultFilled = document.getElementById("resultFilled");

const resultBadge = document.getElementById("resultBadge");
const resultPrediction = document.getElementById("resultPrediction");
const resultRisk = document.getElementById("resultRisk");

const ringProgress = document.getElementById("ringProgress");
const ringValue = document.getElementById("ringValue");

const probMeterFill = document.getElementById("probMeterFill");
const probMeterMarker = document.getElementById("probMeterMarker");

const metricPrediction = document.getElementById("metricPrediction");
const metricThreshold = document.getElementById("metricThreshold");
const metricApiStatus = document.getElementById("metricApiStatus");

const recommendationText = document.getElementById("recommendationText");

const apiStatusDot = document.getElementById("apiStatusDot");
const apiStatusLabel = document.getElementById("apiStatusLabel");
const modelInfoStatus = document.getElementById("modelInfoStatus");

const RING_CIRCUMFERENCE = 2 * Math.PI * 60; // r = 60

/* ==========================================================
   HEALTH CHECK (updates navbar AI MODEL status pill)
========================================================== */
async function checkApiHealth() {
  try {
    const res = await fetch(HEALTH_URL, { method: "GET" });
    if (!res.ok) throw new Error("Health check failed");

    apiStatusDot.classList.remove("status-dot--offline");
    apiStatusDot.classList.add("status-dot--online");
    apiStatusLabel.textContent = "AI MODEL — online";
    modelInfoStatus.textContent = "Online";
  } catch (err) {
    apiStatusDot.classList.remove("status-dot--online");
    apiStatusDot.classList.add("status-dot--offline");
    apiStatusLabel.textContent = "AI MODEL — offline";
    modelInfoStatus.textContent = "Offline";
  }
}

/* ==========================================================
   FORM DATA COLLECTION
========================================================== */
function collectFormData() {
  const formData = new FormData(form);

  const numericFields = [
    "age",
    "avg_time_spent",
    "avg_transaction_value",
    "points_in_wallet",
    "days_since_last_login",
    "avg_frequency_login_days",
    "customer_tenure_days",
    "visit_hour",
  ];

  const payload = {};

  for (const [key, rawValue] of formData.entries()) {
    if (numericFields.includes(key)) {
      payload[key] = Number(rawValue);
    } else {
      payload[key] = rawValue;
    }
  }

  return payload;
}

/* ==========================================================
   ALERT HELPERS
========================================================== */
function showAlert(message) {
  formAlert.textContent = message;
  formAlert.classList.remove("alert--hidden");
}

function hideAlert() {
  formAlert.textContent = "";
  formAlert.classList.add("alert--hidden");
}

/* ==========================================================
   LOADING STATE
========================================================== */
function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  predictBtn.classList.toggle("predict-btn--loading", isLoading);
  predictBtnLabel.textContent = isLoading ? "Analyzing Customer..." : "Predict Churn Risk";
}

/* ==========================================================
   RESET RESULT PANEL (e.g. when inputs change after a prediction)
========================================================== */
function resetPrediction() {
  resultFilled.classList.add("result-filled--hidden");
  resultEmpty.style.display = "flex";
}

/* ==========================================================
   UPDATE RING + PROBABILITY METER
========================================================== */
function updateProbability(probability, isChurn) {
  const percent = probability * 100;

  const offset = RING_CIRCUMFERENCE - (percent / 100) * RING_CIRCUMFERENCE;
  ringProgress.style.strokeDashoffset = offset;
  ringProgress.style.stroke = isChurn ? "var(--accent)" : "var(--success)";
  ringValue.textContent = `${percent.toFixed(2)}%`;

  probMeterFill.style.width = `${percent}%`;
  probMeterMarker.style.left = `${percent}%`;
}

/* ==========================================================
   UPDATE RISK STATUS (badge, headline, sub-label)
========================================================== */
function updateRiskStatus(isChurn, riskStatus) {
  resultBadge.textContent = isChurn ? "High Risk" : "Low Risk";
  resultBadge.className = `result-status__badge ${isChurn ? "result-status__badge--danger" : "result-status__badge--success"}`;

  resultPrediction.textContent = riskStatus.toUpperCase();
  resultRisk.textContent = isChurn ? "CHURN" : "NO CHURN";
}

/* ==========================================================
   DISPLAY PREDICTION
========================================================== */
function displayPrediction(data) {
  hideAlert();

  const isChurn = data.predicted_churn === 1;

  updateRiskStatus(isChurn, data.risk_status);
  updateProbability(data.churn_probability, isChurn);

  metricPrediction.textContent = isChurn ? "CHURN" : "NO CHURN";
  metricThreshold.textContent = data.threshold.toFixed(2);
  metricApiStatus.textContent = "Prediction Successful";

  recommendationText.textContent = isChurn
    ? "Customer shows a high probability of churn. Consider targeted retention actions such as personalized offers, loyalty incentives, and proactive engagement."
    : "Customer currently shows a lower churn probability. Continue regular engagement and monitor future behavioral changes.";

  resultEmpty.style.display = "none";
  resultFilled.classList.remove("result-filled--hidden");
}

/* ==========================================================
   DISPLAY ERROR
========================================================== */
function displayError(message) {
  showAlert(message);
}

/* ==========================================================
   PREDICT CHURN (API CALL)
========================================================== */
async function predictChurn(payload) {
  setLoading(true);
  hideAlert();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.status === 422 || response.status === 400) {
      displayError("Please check the customer information.");
      return;
    }

    if (!response.ok) {
      displayError("Please check the customer information.");
      return;
    }

    const data = await response.json();
    displayPrediction(data);
  } catch (err) {
    displayError("Unable to connect to prediction API. Please make sure the FastAPI server is running.");
  } finally {
    setLoading(false);
  }
}

/* ==========================================================
   FORM SUBMIT HANDLER
========================================================== */
form.addEventListener("submit", (event) => {
  event.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = collectFormData();
  predictChurn(payload);
});

/* ==========================================================
   RESET PREDICTION WHEN USER EDITS INPUTS AFTER A RESULT
========================================================== */
form.addEventListener("input", () => {
  if (!resultFilled.classList.contains("result-filled--hidden")) {
    resetPrediction();
  }
});

/* ==========================================================
   INIT
========================================================== */
checkApiHealth();
