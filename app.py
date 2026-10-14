import os

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import shap
import requests

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

# Form field name -> dataset column name
FIELD_MAP = {
    "DO": "D.O. (mg/l)",
    "PH": "PH",
    "Conductivity": "CONDUCTIVITY (\u00b5mhos/cm)",
    "BOD": "B.O.D. (mg/l)",
    "Nitrate": "NITRATENAN N+ NITRITENANN (mg/l)",
    "Coliform": "TOTAL COLIFORM (MPN/100ml)Mean",
}

LABELS = {
    "D.O. (mg/l)": "Dissolved Oxygen (mg/l)",
    "PH": "pH",
    "CONDUCTIVITY (\u00b5mhos/cm)": "Conductivity (\u00b5mhos/cm)",
    "B.O.D. (mg/l)": "BOD (mg/l)",
    "NITRATENAN N+ NITRITENANN (mg/l)": "Nitrate + Nitrite (mg/l)",
    "TOTAL COLIFORM (MPN/100ml)Mean": "Total Coliform (MPN/100ml)",
}

# Acceptable input ranges.
# pH is bounded on both sides (the scale itself is 0-14, and the training
# data discarded out-of-range pH). Every other parameter is a concentration
# or count, so only non-negativity is enforced. Deliberately NO invented
# upper caps: the raw data legitimately contains values far above
# regulatory limits, and the rating bands already map those to 0.
RANGES = {
    "PH": (0.0, 14.0),
    "D.O. (mg/l)": (0.0, None),
    "CONDUCTIVITY (\u00b5mhos/cm)": (0.0, None),
    "B.O.D. (mg/l)": (0.0, None),
    "NITRATENAN N+ NITRITENANN (mg/l)": (0.0, None),
    "TOTAL COLIFORM (MPN/100ml)Mean": (0.0, None),
}

# ============================================================
# PARAMETER-LEVEL GUIDELINE ASSESSMENT (IS 10500:2012)
# ============================================================
# Per-parameter check against the applicable drinking-water standard.
# This is SEPARATE from the aggregate WQI and is NOT a WQI classification.
# Limits were verified from IS 10500:2012 (Drinking Water - Specification,
# Second Revision, Reaffirmed 2018; Amendments 1-4 checked) as extracted
# from the provided bis-wq-standards.pdf. Only parameters with a directly
# comparable limit return a verdict; the others are reported honestly as
# "Not assessed" with the reason. DO/Conductivity/BOD are intentionally
# omitted: IS 10500 prescribes no drinking-water limit for them, and no
# unit conversion is invented here.
STANDARDS = [
    {
        "field": "PH",
        "label": "pH",
        "unit": "pH",
        "kind": "range",
        "low": 6.5,
        "high": 8.5,
        "provenance": (
            "IS 10500:2012, Table 1, Sl. No. iii "
            "(acceptable 6.5-8.5; permissible: no relaxation)"
        ),
    },
    {
        "field": "Nitrate",
        "label": "Nitrate + Nitrite",
        "unit": "mg/l (as N)",
        "kind": "not_assessed",
        "reason": (
            "IS 10500 expresses nitrate as NO3 (mg/l), whereas this input "
            "is nitrate+nitrite reported as N. The unit bases differ, so no "
            "comparison is made (conversion not verified)."
        ),
        "provenance": "IS 10500:2012, Table 2, Sl. No. xvi",
    },
    {
        "field": "Coliform",
        "label": "Total Coliform",
        "unit": "MPN/100ml",
        "kind": "not_assessed",
        "reason": (
            "IS 10500 requires total coliform to be 'not detectable in any "
            "100 ml sample' (presence/absence), whereas this input is a "
            "numeric MPN/100 ml count. The method bases differ."
        ),
        "provenance": "IS 10500:2012, Table 6",
    },
]


def assess_parameters(values):
    """Return a per-parameter guideline assessment against IS 10500:2012.

    Each item is a dict with label/value/unit/status/note/provenance where
    status is one of: 'within', 'outside', 'not_assessed'. No aggregate or
    overall verdict is produced here - the WQI classification is separate.
    """

    assessments = []

    for spec in STANDARDS:

        val = values.get(spec["field"])

        item = {
            "label": spec["label"],
            "value": val,
            "unit": spec["unit"],
            "provenance": spec["provenance"],
        }

        if spec["kind"] == "range" and val is not None:
            within = spec["low"] <= val <= spec["high"]
            item["status"] = "within" if within else "outside"
            item["note"] = (
                f"Acceptable range {spec['low']}-{spec['high']} {spec['unit']}."
            )
        else:
            item["status"] = "not_assessed"
            item["note"] = spec.get("reason", "")

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
    """Map a Water Quality Index value to its rating category.

    Conventional five-tier WQI rating scheme used with weighted
    arithmetic indices:

        0-25    Excellent
        25-50   Good
        50-75   Poor
        75-100  Very Poor
        >100    Not suitable

    Note on the last class: this project's index is an un-normalised
    weighted sum whose unit weights total 0.998, so its attainable
    range is [0, 99.8]. A value above 100 therefore cannot be produced
    by this index, and the "Not Suitable for Drinking" branch is kept
    only to preserve the published scheme - it is unreachable in
    practice.

    The category is an aggregate screening label for overall water
    quality. It does NOT certify that water is safe to drink; potability
    requires comparing individual parameters against the applicable
    regulatory limits.
    """

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
def generate_explanation(wqi, data, contribs):

    drivers = "; ".join(
        f"{c['label']}={c['value']} pushing WQI {c['contribution']:+.2f} "
        f"from the model average"
        for c in contribs[:3]
    )

    parameters = "\n".join(
        f"{LABELS[col]}: {data[col].iloc[0]}" for col in data.columns
    )

    prompt = f"""
Predicted Water Quality Index (WQI) on a 0-100 scale: {wqi}

Note: this WQI is a project-defined aggregate index computed with a
discrete rating-band methodology (in this scheme a lower value indicates
better quality). It is a screening indicator, NOT a potability
certification and NOT a per-parameter regulatory verdict.

Water parameters:
{parameters}

Model feature contributions (SHAP) for the three strongest drivers:
{drivers}

In 3-4 simple sentences, explain how the parameters above combine to
produce this WQI value and mention the parameters that most influence it.
Base the explanation only on the parameters and contributions given above.
Do not state or imply that this water is safe or unsafe to drink; this
index is a screening indicator, not a potability certification.
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

            # ------------------------------------------------
            # PARAMETER-LEVEL GUIDELINE ASSESSMENT (IS 10500:2012)
            # ------------------------------------------------

            assessments = assess_parameters(values)

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
                input_data,
                contribs
            )

    return render_template(
        "index.html",
        prediction=prediction,
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
