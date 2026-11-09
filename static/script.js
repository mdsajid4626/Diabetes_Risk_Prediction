/**
 * EndoScan AI • Clinical Diabetes Risk Assessment Engine
 * Frontend Controller & Diagnostic Visualization
 */

(function () {
    "use strict";

    // -------------------------------------------------------------------------
    // CONSTANTS & CONFIGURATION
    // -------------------------------------------------------------------------
    const GAUGE_CIRCUMFERENCE = 502.65; // 2 * PI * 80

    const FIELDS = [
        "Glucose",
        "Insulin",
        "BloodPressure",
        "BMI",
        "SkinThickness",
        "Age",
        "Pregnancies",
        "DiabetesPedigreeFunction"
    ];

    const PRESETS = {
        healthy: {
            Glucose: 88,
            Insulin: 72,
            BloodPressure: 68,
            BMI: 22.4,
            SkinThickness: 18,
            Age: 26,
            Pregnancies: 0,
            DiabetesPedigreeFunction: 0.22,
            label: "Healthy Baseline Profile Loaded"
        },
        borderline: {
            Glucose: 122,
            Insulin: 135,
            BloodPressure: 78,
            BMI: 28.6,
            SkinThickness: 28,
            Age: 42,
            Pregnancies: 2,
            DiabetesPedigreeFunction: 0.48,
            label: "Borderline / Pre-Diabetic Profile Loaded"
        },
        highrisk: {
            Glucose: 176,
            Insulin: 245,
            BloodPressure: 88,
            BMI: 35.8,
            SkinThickness: 38,
            Age: 52,
            Pregnancies: 5,
            DiabetesPedigreeFunction: 0.88,
            label: "High Risk Diabetic Profile Loaded"
        }
    };

    // -------------------------------------------------------------------------
    // DOM REFERENCES
    // -------------------------------------------------------------------------
    const htmlEl = document.documentElement;
    const form = document.getElementById("predictionForm");
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const resetFormBtn = document.getElementById("resetFormBtn");
    const predictBtn = document.getElementById("predictButton");
    const openInfoModalBtn = document.getElementById("openInfoModalBtn");
    const closeInfoModalBtn = document.getElementById("closeInfoModalBtn");
    const closeModalActionBtn = document.getElementById("closeModalActionBtn");
    const infoModal = document.getElementById("infoModal");
    const toastContainer = document.getElementById("toastContainer");
    const printReportBtn = document.getElementById("printReportBtn");
    const retryBtn = document.getElementById("retryBtn");

    // State Containers
    const idleState = document.getElementById("idleState");
    const loadingState = document.getElementById("loadingState");
    const activeResultState = document.getElementById("activeResultState");
    const errorState = document.getElementById("errorState");
    const errorMessageText = document.getElementById("errorMessageText");

    // Result Visualization Elements
    const gaugeFill = document.getElementById("gaugeFill");
    const probabilityDisplay = document.getElementById("probabilityDisplay");
    const confidenceTier = document.getElementById("confidenceTier");
    const riskBadge = document.getElementById("riskBadge");
    const riskStatusText = document.getElementById("riskStatusText");
    const biomarkerBreakdownGrid = document.getElementById("biomarkerBreakdownGrid");
    const recList = document.getElementById("recList");
    const recTitle = document.getElementById("recTitle");
    const assessmentTimestamp = document.getElementById("assessmentTimestamp");

    // -------------------------------------------------------------------------
    // THEME CONTROLLER
    // -------------------------------------------------------------------------
    function initTheme() {
        const storedTheme = localStorage.getItem("endoscan-theme");
        if (storedTheme) {
            htmlEl.setAttribute("data-theme", storedTheme);
        } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
            htmlEl.setAttribute("data-theme", "light");
        } else {
            htmlEl.setAttribute("data-theme", "dark");
        }
    }

    function toggleTheme() {
        const currentTheme = htmlEl.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        htmlEl.setAttribute("data-theme", newTheme);
        localStorage.setItem("endoscan-theme", newTheme);
        showToast(`Switched to ${newTheme.toUpperCase()} theme`, "info");
    }

    // -------------------------------------------------------------------------
    // BIOMARKER STATUS EVALUATION
    // -------------------------------------------------------------------------
    function evaluateBiomarkerStatus(field, rawValue) {
        const val = parseFloat(rawValue);
        switch (field) {
            case "Glucose":
                if (val < 100) return { label: "Normal", tier: "normal", status: "ok", text: "Optimal Fasting" };
                if (val <= 125) return { label: "Pre-DM", tier: "warning", status: "warn", text: "Impaired Glucose" };
                return { label: "Elevated", tier: "danger", status: "alert", text: "Diabetic Threshold" };

            case "BloodPressure":
                if (val < 80) return { label: "Optimal", tier: "normal", status: "ok", text: "Normal Diastolic" };
                if (val <= 89) return { label: "Pre-HTN", tier: "warning", status: "warn", text: "Pre-Hypertension" };
                return { label: "Stage 1+", tier: "danger", status: "alert", text: "Elevated Pressure" };

            case "BMI":
                if (val < 18.5) return { label: "Underweight", tier: "warning", status: "warn", text: "Low BMI" };
                if (val <= 24.9) return { label: "Normal", tier: "normal", status: "ok", text: "Healthy Body Mass" };
                if (val <= 29.9) return { label: "Overweight", tier: "warning", status: "warn", text: "Overweight (25-29.9)" };
                return { label: "Obese", tier: "danger", status: "alert", text: "Obese (≥30.0)" };

            case "Insulin":
                if (val < 16) return { label: "Low", tier: "warning", status: "warn", text: "Hypoinsulinemia" };
                if (val <= 166) return { label: "Normal", tier: "normal", status: "ok", text: "Normal Post-Load" };
                return { label: "Elevated", tier: "danger", status: "alert", text: "Hyperinsulinemia" };

            case "SkinThickness":
                if (val <= 30) return { label: "Normal", tier: "normal", status: "ok", text: "Standard Triceps" };
                return { label: "Elevated", tier: "warning", status: "warn", text: "High Subcutaneous Fat" };

            case "Age":
                if (val < 35) return { label: "Younger", tier: "normal", status: "ok", text: "Age < 35" };
                if (val <= 50) return { label: "Moderate", tier: "warning", status: "warn", text: "Age 35-50" };
                return { label: "Mature", tier: "danger", status: "alert", text: "Age > 50" };

            case "Pregnancies":
                if (val <= 2) return { label: "0-2", tier: "normal", status: "ok", text: "Low Gestational Risk" };
                if (val <= 5) return { label: "3-5", tier: "warning", status: "warn", text: "Moderate Parity" };
                return { label: "6+", tier: "warning", status: "warn", text: "High Multi-Parity" };

            case "DiabetesPedigreeFunction":
                if (val < 0.40) return { label: "Low DPF", tier: "normal", status: "ok", text: "Low Genetic Signal" };
                if (val <= 0.65) return { label: "Average", tier: "normal", status: "ok", text: "Population Median" };
                return { label: "Strong DPF", tier: "danger", status: "alert", text: "High Hereditary Risk" };

            default:
                return { label: "Standard", tier: "normal", status: "ok", text: "Recorded" };
        }
    }

    function updateBadge(field, value) {
        const badgeEl = document.getElementById(`badge-${field}`);
        if (!badgeEl) return;

        const info = evaluateBiomarkerStatus(field, value);
        badgeEl.textContent = info.label;
        badgeEl.className = `range-badge ${info.tier}`;
    }

    // -------------------------------------------------------------------------
    // INPUT CONTROLS SYNCHRONIZATION
    // -------------------------------------------------------------------------
    function initFieldSync() {
        FIELDS.forEach((field) => {
            const numInput = document.getElementById(field);
            const rangeInput = document.getElementById(`${field}-range`);

            if (!numInput || !rangeInput) return;

            // Slider input change -> Number input
            rangeInput.addEventListener("input", () => {
                numInput.value = rangeInput.value;
                updateBadge(field, rangeInput.value);
            });

            // Number input change -> Slider
            numInput.addEventListener("input", () => {
                const val = parseFloat(numInput.value);
                if (!isNaN(val)) {
                    rangeInput.value = val;
                    updateBadge(field, val);
                }
            });

            // Trigger initial badge state
            updateBadge(field, numInput.value);
        });
    }

    // -------------------------------------------------------------------------
    // PRESETS HANDLER
    // -------------------------------------------------------------------------
    function loadPreset(presetKey) {
        const preset = PRESETS[presetKey];
        if (!preset) return;

        FIELDS.forEach((field) => {
            const val = preset[field];
            const numInput = document.getElementById(field);
            const rangeInput = document.getElementById(`${field}-range`);

            if (numInput && rangeInput && val !== undefined) {
                numInput.value = val;
                rangeInput.value = val;
                updateBadge(field, val);
            }
        });

        showToast(preset.label, "success");
    }

    // -------------------------------------------------------------------------
    // FORM RESET
    // -------------------------------------------------------------------------
    function resetAllFields() {
        loadPreset("healthy");
        showIdleState();
        showToast("Parameters reset to healthy baseline defaults", "info");
    }

    // -------------------------------------------------------------------------
    // VIEW STATE SWITCHING
    // -------------------------------------------------------------------------
    function showIdleState() {
        idleState.style.display = "block";
        loadingState.style.display = "none";
        activeResultState.style.display = "none";
        errorState.style.display = "none";
        predictBtn.disabled = false;
        assessmentTimestamp.textContent = "Realtime Engine";
    }

    function showLoadingState() {
        idleState.style.display = "none";
        loadingState.style.display = "block";
        activeResultState.style.display = "none";
        errorState.style.display = "none";
        predictBtn.disabled = true;
    }

    function showActiveResultState() {
        idleState.style.display = "none";
        loadingState.style.display = "none";
        activeResultState.style.display = "block";
        errorState.style.display = "none";
        predictBtn.disabled = false;
    }

    function showErrorState(message) {
        idleState.style.display = "none";
        loadingState.style.display = "none";
        activeResultState.style.display = "none";
        errorState.style.display = "block";
        errorMessageText.textContent = message || "An unexpected error occurred during prediction.";
        predictBtn.disabled = false;
    }

    // -------------------------------------------------------------------------
    // ANIMATED GAUGE & COUNTER
    // -------------------------------------------------------------------------
    function animateGauge(targetPercentage) {
        // Clamp between 0 and 100
        const percentage = Math.max(0, Math.min(100, targetPercentage));
        const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * (percentage / 100));

        // Gauge stroke styling based on risk severity
        gaugeFill.classList.remove("low", "moderate", "high");
        if (percentage < 33.3) {
            gaugeFill.classList.add("low");
        } else if (percentage < 66.6) {
            gaugeFill.classList.add("moderate");
        } else {
            gaugeFill.classList.add("high");
        }

        // Apply animated stroke-dashoffset
        gaugeFill.style.strokeDashoffset = offset;

        // Animated number counter
        let startVal = 0;
        const duration = 1200; // ms
        const startTime = performance.now();

        function countStep(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentVal = (startVal + (percentage - startVal) * eased).toFixed(1);

            probabilityDisplay.textContent = `${currentVal}%`;

            if (progress < 1) {
                requestAnimationFrame(countStep);
            } else {
                probabilityDisplay.textContent = `${percentage.toFixed(1)}%`;
            }
        }

        requestAnimationFrame(countStep);
    }

    // -------------------------------------------------------------------------
    // RENDER BIOMARKER BREAKDOWN MATRIX
    // -------------------------------------------------------------------------
    function renderBiomarkerMatrix(data) {
        biomarkerBreakdownGrid.innerHTML = "";

        const items = [
            { key: "Glucose", name: "Plasma Glucose", unit: "mg/dL" },
            { key: "BMI", name: "Body Mass Index", unit: "kg/m²" },
            { key: "BloodPressure", name: "Blood Pressure", unit: "mm Hg" },
            { key: "Insulin", name: "Serum Insulin", unit: "µU/mL" },
            { key: "DiabetesPedigreeFunction", name: "Pedigree DPF", unit: "Score" },
            { key: "Age", name: "Patient Age", unit: "Yrs" }
        ];

        items.forEach((item) => {
            const rawVal = data[item.key];
            const info = evaluateBiomarkerStatus(item.key, rawVal);

            const div = document.createElement("div");
            div.className = "breakdown-item";
            div.innerHTML = `
                <div class="item-left">
                    <span class="item-name">${item.name}</span>
                    <span class="item-val">${rawVal} <small style="font-size:0.7em;color:var(--text-tertiary);">${item.unit}</small></span>
                </div>
                <span class="item-tag ${info.status}">${info.label}</span>
            `;
            biomarkerBreakdownGrid.appendChild(div);
        });
    }

    // -------------------------------------------------------------------------
    // GENERATE TAILORED CLINICAL RECOMMENDATIONS
    // -------------------------------------------------------------------------
    function renderRecommendations(probability, data) {
        recList.innerHTML = "";
        const recommendations = [];

        const glucose = parseFloat(data.Glucose);
        const bmi = parseFloat(data.BMI);
        const bp = parseFloat(data.BloodPressure);
        const dpf = parseFloat(data.DiabetesPedigreeFunction);

        if (probability >= 50) {
            recTitle.textContent = "High Risk Clinical Management Plan";
            recommendations.push("Schedule a formal diagnostic laboratory evaluation (Fasting Plasma Glucose & Glycated Hemoglobin HbA1c test) with your healthcare provider.");
            if (glucose >= 126) {
                recommendations.push("Your fasting plasma glucose is within the elevated hyperglycemic range. Immediate clinical correlation and metabolic panel follow-up is indicated.");
            }
            if (bmi >= 30) {
                recommendations.push("Structured medical nutritional therapy (MNT) and a clinical weight-management regimen can improve insulin sensitivity by 30–50%.");
            }
            if (bp >= 85) {
                recommendations.push("Concomitant blood pressure elevation warrants routine 24-hour ambulatory monitoring to mitigate composite cardiovascular risk.");
            }
            recommendations.push("Self-monitoring of blood glucose (SMBG) and reduction of refined simple carbohydrates are strongly advised.");
        } else if (probability >= 30) {
            recTitle.textContent = "Moderate / Pre-Diabetic Lifestyle Interventions";
            recommendations.push("Recommended for 6-month or annual metabolic screening to track glycemic progression.");
            if (bmi >= 25) {
                recommendations.push("A modest 5–7% sustained reduction in body mass can substantially decrease the probability of progression to overt Type 2 Diabetes.");
            }
            if (dpf > 0.6) {
                recommendations.push("Given elevated familial predisposition (high DPF score), preventive lifestyle behaviors provide protective resilience against hereditary onset.");
            }
            recommendations.push("Incorporate at least 150 minutes of moderate-intensity aerobic physical activity weekly, complemented by resistance training.");
        } else {
            recTitle.textContent = "Optimal Health Maintenance Guidelines";
            recommendations.push("Biomarkers align with a low metabolic risk profile. Continue routine annual preventive wellness checks.");
            recommendations.push("Maintain a balanced dietary pattern rich in whole fiber, lean proteins, legumes, and omega-3 fatty acids.");
            recommendations.push("Regular physical activity and adequate restorative sleep (7–8 hours/night) preserve metabolic and cardiovascular homeostasis.");
        }

        recommendations.forEach((text) => {
            const li = document.createElement("li");
            li.className = "rec-item";
            li.innerHTML = `<span class="rec-item-bullet"></span><span>${text}</span>`;
            recList.appendChild(li);
        });
    }

    // -------------------------------------------------------------------------
    // FORM SUBMISSION & PREDICTION DISPATCH
    // -------------------------------------------------------------------------
    async function handlePredictionSubmit(event) {
        event.preventDefault();

        // Gather and validate inputs
        const payload = {};
        for (const field of FIELDS) {
            const input = document.getElementById(field);
            if (!input || input.value.trim() === "") {
                showToast(`Please specify a valid value for ${field}`, "error");
                input.focus();
                return;
            }
            payload[field] = parseFloat(input.value);
        }

        showLoadingState();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server responded with status ${response.status}`);
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || "Model prediction returned unsuccessful status");
            }

            // Successfully received prediction
            const probability = parseFloat(result.probability);
            const isHighRisk = result.prediction === 1 || probability >= 50;

            showActiveResultState();

            // Animate Radial Gauge
            animateGauge(probability);

            // Configure Risk Badge & Status
            riskBadge.classList.remove("low", "moderate", "high");
            if (probability < 33.3) {
                riskBadge.classList.add("low");
                riskStatusText.textContent = "LOW RISK PROFILE";
                confidenceTier.textContent = "Within Healthy Normative Range";
            } else if (probability < 60.0) {
                riskBadge.classList.add("moderate");
                riskStatusText.textContent = "MODERATE RISK";
                confidenceTier.textContent = "Borderline / Pre-Diabetic Markers";
            } else {
                riskBadge.classList.add("high");
                riskStatusText.textContent = "HIGH RISK PROFILE";
                confidenceTier.textContent = "Elevated Diabetic Indicator Consensus";
            }

            // Render Biomarker Breakdown
            renderBiomarkerMatrix(payload);

            // Render Recommendations
            renderRecommendations(probability, payload);

            // Update timestamp
            const now = new Date();
            assessmentTimestamp.textContent = `Completed ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

            showToast("Risk assessment successfully generated", "success");

        } catch (err) {
            console.error("Prediction Error:", err);
            showErrorState(err.message || "Failed to communicate with prediction service. Please ensure the server is active.");
            showToast("Inference error occurred", "error");
        }
    }

    // -------------------------------------------------------------------------
    // TOAST NOTIFICATIONS
    // -------------------------------------------------------------------------
    function showToast(message, type = "info") {
        if (!toastContainer) return;

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;

        let icon = "ℹ️";
        if (type === "success") icon = "✅";
        if (type === "error") icon = "❌";

        toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(10px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 3200);
    }

    // -------------------------------------------------------------------------
    // MODAL DIALOG CONTROLS
    // -------------------------------------------------------------------------
    function openModal() {
        if (!infoModal) return;
        infoModal.classList.add("open");
        infoModal.setAttribute("aria-hidden", "false");
    }

    function closeModal() {
        if (!infoModal) return;
        infoModal.classList.remove("open");
        infoModal.setAttribute("aria-hidden", "true");
    }

    // -------------------------------------------------------------------------
    // ATTACH EVENT LISTENERS
    // -------------------------------------------------------------------------
    function bindEvents() {
        // Theme Toggle
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener("click", toggleTheme);
        }

        // Presets Buttons
        document.querySelectorAll(".btn-preset").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                const targetBtn = e.currentTarget;
                const presetKey = targetBtn.getAttribute("data-preset");
                loadPreset(presetKey);
            });
        });

        // Reset Form
        if (resetFormBtn) {
            resetFormBtn.addEventListener("click", resetAllFields);
        }

        // Form Submit
        if (form) {
            form.addEventListener("submit", handlePredictionSubmit);
        }

        // Retry Button
        if (retryBtn) {
            retryBtn.addEventListener("click", () => {
                showIdleState();
            });
        }

        // Print Report
        if (printReportBtn) {
            printReportBtn.addEventListener("click", () => {
                window.print();
            });
        }

        // Info Modal
        if (openInfoModalBtn) openInfoModalBtn.addEventListener("click", openModal);
        if (closeInfoModalBtn) closeInfoModalBtn.addEventListener("click", closeModal);
        if (closeModalActionBtn) closeModalActionBtn.addEventListener("click", closeModal);

        if (infoModal) {
            infoModal.addEventListener("click", (e) => {
                if (e.target === infoModal) closeModal();
            });
        }

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && infoModal && infoModal.classList.contains("open")) {
                closeModal();
            }
        });
    }

    // -------------------------------------------------------------------------
    // INITIALIZATION
    // -------------------------------------------------------------------------
    function init() {
        initTheme();
        initFieldSync();
        bindEvents();
        // Load default healthy profile values into view
        loadPreset("healthy");
    }

    // Run when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();