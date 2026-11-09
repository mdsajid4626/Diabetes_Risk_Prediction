import pandas as pd
import joblib

model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("=" * 60)
print("DIABETES RISK PREDICTION")
print("=" * 60)

print("\nEnter Patient Details")
print("-" * 40)

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

input_data = pd.DataFrame([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]], columns=[
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"\nDiabetes Probability: {probability * 100:.2f}%")

if prediction[0] == 1:
    print("Risk Status: HIGH RISK")
else:
    print("Risk Status: LOW RISK")

print("\n" + "=" * 60)