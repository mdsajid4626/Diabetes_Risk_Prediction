from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        input_data = pd.DataFrame([[
            float(data["Pregnancies"]),
            float(data["Glucose"]),
            float(data["BloodPressure"]),
            float(data["SkinThickness"]),
            float(data["Insulin"]),
            float(data["BMI"]),
            float(data["DiabetesPedigreeFunction"]),
            float(data["Age"])
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

        # Scale input using saved scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Get diabetes probability
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            risk_status = "HIGH RISK"
        else:
            risk_status = "LOW RISK"

        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "probability": round(float(probability * 100), 2),
            "risk_status": risk_status
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)