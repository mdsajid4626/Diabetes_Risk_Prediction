# EndoScan AI • Clinical Diabetes Risk Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20KNN-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Clinical%20Screening-emerald.svg)]()

**EndoScan AI** is a state-of-the-art HealthTech metabolic risk intelligence dashboard. Powered by a calibrated K-Nearest Neighbors (KNN) classification model trained on the standardized PIMA Indians Diabetes benchmark, it provides real-time, evidence-informed risk stratification, biomarker analysis, and tailored lifestyle guidance.

---

## Key Features

- **Modern Clinical Glassmorphic UI**: High-tech clinical dashboard aesthetic with ambient lighting, glassmorphism cards, and Google Fonts (`Plus Jakarta Sans` & `Inter`).
- **Dark & Light Mode**: Seamless theme switching with local storage persistence and operating system preference detection.
- **Interactive Dual-Control Inputs**: Synchronized range sliders and precise numeric stepper inputs for all 8 patient biomarkers.
- **Live Clinical Reference Badges**: Real-time status indicators (Optimal / Borderline / Elevated) updating dynamically as measurements change.
- **One-Click Patient Presets**: Instant demonstration with pre-configured profiles:
  - *Healthy Baseline*
  - *Borderline / Pre-Diabetic*
  - *High Risk Diabetic*
- **Animated SVG Circular Risk Gauge**: Radial SVG visualization with dynamic color transitions (Emerald, Amber, Rose) and smooth percentage counters.
- **Biomarker Risk Analysis Matrix**: Granular breakdown flagging which specific parameters contribute to the elevated risk score.
- **Tailored Clinical Action Plan**: Evidence-based preliminary recommendations customized to the patient's individual risk tier and elevated biomarkers.
- **Printable Clinical Summary**: Optimized `@media print` layout allowing one-click printing or PDF export of the patient's assessment.
- **Model Architecture & Reference Modal**: In-depth transparency regarding the training dataset, Z-score standard scaling, and feature definitions.

---

## Clinical Biomarkers Evaluated

| Biomarker | Clinical Description | Unit | Healthy Normative Range |
| :--- | :--- | :--- | :--- |
| **Glucose** | 2-hour plasma glucose from oral glucose tolerance test | mg/dL | 70 – 99 mg/dL |
| **Insulin** | 2-hour post-load serum insulin concentration | µU/mL | 16 – 166 µU/mL |
| **Blood Pressure** | Diastolic blood pressure | mm Hg | < 80 mm Hg |
| **BMI** | Body Mass Index (weight in kg / height in m²) | kg/m² | 18.5 – 24.9 kg/m² |
| **Skin Thickness** | Triceps skinfold measurement | mm | 10 – 30 mm |
| **Age** | Chronological patient age | Years | Adult benchmark |
| **Pregnancies** | Number of gestational events | Count | Obstetric history |
| **Diabetes Pedigree** | Synthesis of family diabetic lineage | Score | Population avg: ~0.47 |

---

## Installation & Local Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Machine Learning Pipeline

1. **Dataset**: PIMA Indians Diabetes Database (768 patient samples).
2. **Preprocessing**: Missing physiological zero-values imputed and normalized using `StandardScaler` (Z-score scaling).
3. **Model**: Calibrated K-Nearest Neighbors Classifier (`k=9`, uniform distance weighting).
4. **Inference**: Sub-millisecond local execution returning binary classification alongside continuous class probability distribution.

---

## Medical Disclaimer

> **Important**: EndoScan AI is an investigational risk stratification and triage tool intended for educational, research, and preliminary screening purposes. It is not intended to provide definitive medical diagnosis or replace standardized clinical laboratory evaluations (such as laboratory HbA1c or Oral Glucose Tolerance Testing).
# Diabetes_Risk_Prediction
