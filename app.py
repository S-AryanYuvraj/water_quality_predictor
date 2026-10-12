from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap
import requests

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model/wqi_model.pkl")
features = joblib.load("model/features.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)


# ============================================================
# WQI CLASSIFICATION
# ============================================================

def classify_wqi(wqi):

    if wqi <= 25:
        return "Excellent"
    elif wqi <= 50:
        return "Good"
    elif wqi <= 75:
        return "Poor"
    elif wqi <= 100:
        return "Very Poor"
    else:
        return "Not Suitable for Drinking"


# ============================================================
# GENAI EXPLANATION
# ============================================================
def generate_explanation(wqi, category, data, shap_values):

    prompt = f"""
Predicted WQI: {wqi}
Category: {category}

Water parameters:
Temperature: {data["Temp"].iloc[0]}
Dissolved Oxygen: {data["D.O. (mg/l)"].iloc[0]}
pH: {data["PH"].iloc[0]}
Conductivity: {data["CONDUCTIVITY (µmhos/cm)"].iloc[0]}
BOD: {data["B.O.D. (mg/l)"].iloc[0]}
Nitrate + Nitrite: {data["NITRATENAN N+ NITRITENANN (mg/l)"].iloc[0]}
Total Coliform: {data["TOTAL COLIFORM (MPN/100ml)Mean"].iloc[0]}

In 3-4 simple sentences, explain why this water quality category was obtained and mention the most concerning parameters.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]

        return f"Ollama error: {response.status_code}"

    except requests.exceptions.Timeout:
        return "AI explanation timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Make sure Ollama is running."

    except Exception as e:
        return f"AI explanation error: {str(e)}"


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    category = None
    explanation = None

    if request.method == "POST":

        data = {
            "Temp": float(request.form["Temp"]),
            "D.O. (mg/l)": float(request.form["DO"]),
            "PH": float(request.form["PH"]),
            "CONDUCTIVITY (µmhos/cm)": float(
                request.form["Conductivity"]
            ),
            "B.O.D. (mg/l)": float(request.form["BOD"]),
            "NITRATENAN N+ NITRITENANN (mg/l)": float(
                request.form["Nitrate"]
            ),
            "TOTAL COLIFORM (MPN/100ml)Mean": float(
                request.form["Coliform"]
            )
        }

        # Create dataframe
        input_data = pd.DataFrame([data])[features]

        # ----------------------------------------------------
        # WQI PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        category = classify_wqi(prediction)

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_values = explainer.shap_values(input_data)

        # ----------------------------------------------------
        # GENAI EXPLANATION
        # ----------------------------------------------------

        explanation = generate_explanation(
            prediction,
            category,
            input_data,
            shap_values
        )

    return render_template(
        "index.html",
        category=category,
        prediction=prediction,
        explanation=explanation
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)