import os

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import shap
import requests

import wawqi

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model/wqi_model.pkl")
features = joblib.load("model/features.pkl")

# Guard against drift between the training notebook and this app:
# the pickled feature list must match the model's own input width.
assert model.n_features_in_ == len(features), (
    f"model expects {model.n_features_in_} features but "
    f"features.pkl lists {len(features)} - re-run notebooks/WqiPred.ipynb"
)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Form field name -> dataset column name (the 7 Brown WAWQI parameters).
FIELD_MAP = {
    "DO": "D.O. (mg/l)",
    "PH": "PH",
    "Conductivity": "CONDUCTIVITY (\u00b5mhos/cm)",
    "BOD": "B.O.D. (mg/l)",
    "Nitrate": "NITRATENAN N+ NITRITENANN (mg/l)",
    "FecalColiform": "FECAL COLIFORM (MPN/100ml)",
    "Coliform": "TOTAL COLIFORM (MPN/100ml)Mean",
}

LABELS = {
    "D.O. (mg/l)": "Dissolved Oxygen (mg/l)",
    "PH": "pH",
    "CONDUCTIVITY (\u00b5mhos/cm)": "Conductivity (\u00b5mhos/cm)",
    "B.O.D. (mg/l)": "BOD (mg/l)",
    "NITRATENAN N+ NITRITENANN (mg/l)": "Nitrate + Nitrite (mg/l as N)",
    "FECAL COLIFORM (MPN/100ml)": "Fecal Coliform (MPN/100ml)",
    "TOTAL COLIFORM (MPN/100ml)Mean": "Total Coliform (MPN/100ml)",
}

# Acceptable input ranges. The WAWQI sub-index handles out-of-standard values
# (clipped to [0, 100]), so the form only rejects NON-PHYSICAL inputs: pH is
# bounded to the physical 0-14 scale, everything else is a concentration or
# count and must simply be non-negative. No invented upper caps.
RANGES = {
    "PH": (0.0, 14.0),
    "D.O. (mg/l)": (0.0, None),
    "CONDUCTIVITY (\u00b5mhos/cm)": (0.0, None),
    "B.O.D. (mg/l)": (0.0, None),
    "NITRATENAN N+ NITRITENANN (mg/l)": (0.0, None),
    "FECAL COLIFORM (MPN/100ml)": (0.0, None),
    "TOTAL COLIFORM (MPN/100ml)Mean": (0.0, None),
}

# ============================================================
# PARAMETER-LEVEL STANDARD CHECK (the WAWQI's own standards)
# ============================================================
# Each parameter is checked against the SAME permissible value the WAWQI
# sub-index uses (CPCB Designated Best Use Class A where it exists, otherwise
# a cited authoritative source - see wawqi.SOURCE). This is a per-parameter
# view of the index inputs; it is SEPARATE from the aggregate WQI and is NOT a
# potability certification (the aggregate index and the class bands are).


def assess_parameters(values_by_column):
    """Per-parameter check against the WAWQI standard for each input.

    ``values_by_column`` maps dataset column -> numeric value. Each item is a
    dict with label/value/unit/status/note/provenance, where status is
    'within' or 'outside' the permissible value used by the index. No overall
    verdict is produced here - the WQI class is separate.
    """

    assessments = []

    for col in wawqi.PARAMS:
        val = values_by_column.get(col)
        s = wawqi.STANDARDS[col]
        unit = wawqi.UNITS[col]
        direction = wawqi.DIRECTION[col]

        if direction == "min":
            std_txt = f"permissible {s:g} {unit} (minimum)"
            within = val is not None and val >= s
        elif direction == "two_sided":
            std_txt = f"permissible band 6.5-8.5 (ideal 7.0)"
            within = val is not None and 6.5 <= val <= 8.5
        else:
            std_txt = f"permissible {s:g} {unit} (maximum)"
            within = val is not None and val <= s

        q = wawqi.sub_index(col, val) if val is not None else None

        item = {
            "label": LABELS[col],
            "value": val,
            "unit": unit,
            "status": "within" if within else "outside",
            "note": (
                f"{std_txt}; sub-index q={q:.1f} of 100."
                if q is not None else f"{std_txt}."
            ),
            "provenance": wawqi.SOURCE[col],
        }
        assessments.append(item)

    return assessments


# Ollama / GenAI settings (defaults keep the original llama3.2 setup)
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://localhost:11434/api/generate"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
# The first request after Ollama unloads a model pays a load penalty,
# so this stays generous; a slow reply degrades to the fallback text.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(form):
    """Return (values, errors) from the submitted form."""

    values = {}
    errors = []

    for field, column in FIELD_MAP.items():
        raw = (form.get(field) or "").strip()

        if raw == "":
            errors.append(f"{field}: value is required.")
            continue

        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{field}: '{raw}' is not a number.")
            continue

        if not np.isfinite(val):
            errors.append(f"{field}: value must be finite.")
            continue

        lo, hi = RANGES[column]

        if lo is not None and val < lo:
            errors.append(
                f"{LABELS[column]}: must be at least {lo}."
            )
            continue

        if hi is not None and val > hi:
            errors.append(
                f"{LABELS[column]}: must be at most {hi}."
            )
            continue

        values[field] = val

    return values, errors


# ============================================================
# WQI CLASSIFICATION
# ============================================================

def classify_wqi(wqi):
    """Classify a Brown WAWQI value into a quality class.

    Delegates to wawqi.classify_wqi so the app and the training pipeline share
    one definition. LOWER is BETTER (0-25 Excellent, 25-50 Good, 50-75 Poor,
    75-100 Very Poor); q_i = 0 at the ideal value and 100 at the permissible
    limit, so a lower aggregate index means better water.
    """

    return wawqi.classify_wqi(wqi)


# ============================================================
# GENAI EXPLANATION
# ============================================================
def generate_explanation(wqi, category, data, contribs):

    drivers = "; ".join(
        f"{c['label']}={c['value']} contributed {c['contribution']:+.2f} to "
        f"the predicted WQI vs the model average "
        f"({'raised/worsened' if c['contribution'] > 0 else 'lowered/improved'} it)"
        for c in contribs[:3]
    )

    parameters = "\n".join(
        f"{LABELS[col]}: {data[col].iloc[0]}" for col in data.columns
    )

    prompt = f"""
Predicted Water Quality Index (WQI) on a 0-100 scale: {wqi} ({category}).

Direction (MUST be respected): this is a Brown-style Weighted Arithmetic WQI -
the weighted mean of per-parameter sub-indices, each scoring 0 at the ideal
value and 100 at the permissible limit. A LOWER WQI means BETTER quality and a
HIGHER WQI means WORSE quality. Classes: 0-25 Excellent, 25-50 Good, 50-75
Poor, 75-100 Very Poor. This sample scores {wqi}, i.e. "{category}". Because
higher is worse, describe it consistently with that class: do NOT call a Poor
or Very Poor index "good", "relatively good", "acceptable" or "safe".

Water parameters:
{parameters}

Model contributions (SHAP): these are the amounts each parameter moved the
MODEL PREDICTION up or down relative to the model average. They are NOT causal
effects and NOT direct measurements of water quality or safety:
{drivers}

Framing: this index, its class bands and the per-parameter thresholds are the
PROJECT'S OWN adopted screening values. "Within" a threshold means within the
project's selected value only - NOT regulatory compliance and NOT a
drinking-water safety verdict.

In 3-4 simple sentences, explain how the parameters above combine to produce
this WQI and its "{category}" class, naming the parameters that contributed most
to the prediction. Base the explanation ONLY on the value, class, parameters and
contributions above, and keep the higher-is-worse direction consistent. Do not
state or imply that this water is safe or unsafe to drink.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()["response"]

        return f"Ollama error: {response.status_code}"

    except requests.exceptions.Timeout:
        return "AI explanation timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return ("Could not connect to Ollama at "
                f"{OLLAMA_URL}. Make sure Ollama is running.")

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
    contribs = []
    base = None
    assessments = []
    errors = []

    if request.method == "POST":

        values, errors = validate_inputs(request.form)

        if not errors:

            data = {FIELD_MAP[f]: values[f] for f in FIELD_MAP}

            # Create dataframe in the exact trained column order
            input_data = pd.DataFrame([data])[features]

            # ------------------------------------------------
            # WQI PREDICTION
            # ------------------------------------------------

            prediction = float(model.predict(input_data)[0])

            # The index is defined on 0-100; guard against any
            # numeric drift outside it before displaying/classifying.
            prediction = min(max(prediction, 0.0), 100.0)
            prediction = round(prediction, 2)

            # WAWQI quality class (LOWER is BETTER).
            category = classify_wqi(prediction)

            # ------------------------------------------------
            # PARAMETER-LEVEL STANDARD CHECK (WAWQI standards)
            # ------------------------------------------------

            assessments = assess_parameters(data)

            # ------------------------------------------------
            # SHAP - per-prediction feature contributions
            # ------------------------------------------------

            sv = np.asarray(
                explainer.shap_values(input_data)
            ).reshape(1, -1)[0]

            base = float(np.asarray(explainer.expected_value).ravel()[0])

            contribs = sorted(
                (
                    {
                        "label": LABELS.get(col, col),
                        "value": float(input_data[col].iloc[0]),
                        "contribution": float(c),
                    }
                    for col, c in zip(features, sv)
                ),
                key=lambda d: abs(d["contribution"]),
                reverse=True
            )

            # ------------------------------------------------
            # GENAI EXPLANATION
            # ------------------------------------------------

            explanation = generate_explanation(
                prediction,
                category,
                input_data,
                contribs
            )

    return render_template(
        "index.html",
        prediction=prediction,
        category=category,
        explanation=explanation,
        contribs=contribs,
        base=base,
        assessments=assessments,
        errors=errors
    )


# ============================================================
# RUN
# ============================================================

# Development server only - use a real WSGI server for any deployment.
if __name__ == "__main__":
    app.run(debug=True)
