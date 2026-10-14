# AI-Based Water Quality Prediction and Assessment System

Predicts a **Water Quality Index (WQI)** from six physico-chemical /
microbial parameters using a gradient-boosting regressor, reports the
**numeric WQI** (lower = better) instead of a single category label, adds a
**separate parameter-level guideline check** against **IS 10500:2012** (the
BIS drinking-water standard), explains the per-parameter drivers with
**SHAP**, and generates a plain-language narrative with a local
**Ollama / Llama 3.2** model.

---

## Project layout

```
WaterQualityPred/
├── app.py                     Flask application (prediction + SHAP + Ollama)
├── requirements.txt
├── data/water_dataX.csv       raw input dataset
├── notebooks/WqiPred.ipynb    cleaning -> WQI -> train/eval/tune -> SHAP -> save
├── notebooks/eval_experiments.ipynb  read-only evaluation: baseline, CV, error analysis, stacking
├── model/wqi_model.pkl        trained GradientBoostingRegressor
├── model/features.pkl         feature column order (must match the model)
├── templates/index.html
├── static/css/style.css
└── backups/                   pre-change snapshots (git-ignored)
```

---

## Running the application

From the **repository root**:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>.

`app.py` loads `model/wqi_model.pkl` and `model/features.pkl` with relative
paths, so it **must** be started from the repository root.

### Optional: AI narrative (Ollama)

The prediction, category and SHAP table work without Ollama. Only the
"AI Water Quality Explanation" panel needs it:

```powershell
ollama serve          # if not already running
ollama pull llama3.2
```

Configuration via environment variables (defaults shown):

| Variable | Default |
|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | `llama3.2` |
| `OLLAMA_TIMEOUT` | `120` (seconds; generous because the first call may load the model) |

If Ollama is unreachable the app degrades gracefully and shows a
connection message instead of failing.

---

## Retraining

The notebook reads `../data/water_dataX.csv` and writes `../model/*.pkl`,
so it must be executed **from the `notebooks/` directory**.

```powershell
jupyter notebook notebooks/WqiPred.ipynb
```

or headlessly:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\WqiPred.ipynb
```

Executing it regenerates `model/wqi_model.pkl` and `model/features.pkl`.
`app.py` asserts at start-up that the pickled feature list matches the
model's expected input width, so notebook/app drift fails fast.

---

## Methodology

### The WQI equation

\[ \text{WQI} = \sum_{i=1}^{6} n_i \, w_i \]

* \(n_i \in \{0, 40, 60, 80, 100\}\) — **discretely binned** normalized
  rating of parameter *i*
* \(w_i\) — unit weight of parameter *i*

| Parameter | Bands → rating | \(w_i\) |
|---|---|---|
| pH | <6.5→0, 6.5–6.7→40, 6.7–6.8→60, 6.8–7.0→80, 7.0–8.5→100, 8.5–8.6→80, 8.6–8.8→60, 8.8–9.0→40, >9.0→0 | 0.165 |
| Dissolved Oxygen (mg/l) | <3→0, 3–4.1→40, 4.1–5.1→60, 5.1–6→80, ≥6→100 | 0.281 |
| BOD (mg/l) | 0–3→100, 3–6→80, 6–80→60, 80–125→40, >125→0 | 0.234 |
| Conductivity (µmhos/cm) | 0–75→100, 75–150→80, 150–225→60, 225–300→40, >300→0 | 0.009 |
| Nitrate + Nitrite (mg/l) | 0–20→100, 20–50→80, 50–100→60, 100–200→40, >200→0 | 0.028 |
| Total Coliform (MPN/100ml) | 0–5→100, 5–50→80, 50–500→60, 500–1000→40, >1000→0 | 0.281 |

**Important properties**

* \(\sum w_i = 0.998\). The sum is **not normalised**, so the attainable
  range is **[0, 99.8]**, not [0, 100].
* Temperature and faecal coliform are **not** part of this index and are
  not model inputs.
* Because ratings are quantised into five levels, many different parameter
  combinations collapse onto the same WQI value (238 distinct WQI values
  across 1593 rows).
* The WQI label is an **aggregate screening indicator**. It does not
  certify drinking-water suitability; potability requires checking each
  parameter against the applicable regulatory limits.

### Rating categories

| WQI | Category |
|---|---|
| 0–25 | Excellent |
| 25–50 | Good |
| 50–75 | Poor |
| 75–100 | Very Poor |
| >100 | Not Suitable for Drinking |

This is the conventional five-tier scheme used with weighted-arithmetic
indices. Under this project's un-normalised index the last class is
**unreachable** (maximum possible value 99.8); it is retained only to
preserve the published scheme.

### Interface output — numeric WQI only

The web app displays the **numeric WQI only** and no longer prints an
overall Excellent/Good/Poor/Very Poor label, so that a single category is
never presented as if it certified suitability. The five-tier scheme above
is kept as methodology background (and the internal `classify_wqi` helper)
but is not shown in the UI.

## Parameter-level guideline assessment (IS 10500:2012)

The app shows a **separate** per-parameter table that checks each input
against the applicable **BIS drinking-water standard**, so a user can see
where an individual parameter sits relative to a regulatory limit. This
check is **independent of the aggregate WQI and is not a WQI
classification**.

* **Standard & provenance** — IS 10500 : 2012, *Drinking Water —
  Specification* (Second Revision), © BIS 2012, **Reaffirmed 2018**, read
  from the provided `bis-wq-standards.pdf`. **Amendments No. 1 (Jun 2015),
  No. 2 (Sep 2018), No. 3 (Feb 2021) and No. 4 (Nov 2021)** were reviewed;
  they change mineral oil / boron / uranium limits, the table notes, and the
  bacteriological *sampling* method (→ IS 15185), and **do not alter** any
  value used below.
* **pH** (Table 1, Sl. No. iii) — acceptable **6.5–8.5**, permissible "no
  relaxation" (identical, so the check is effectively binary) → the panel
  returns *Within* / *Outside* guideline directly.
* **Nitrate + Nitrite** — **Not assessed**: IS 10500 states nitrate as NO₃
  (mg/l, Table 2, Sl. No. xvi) whereas this model's input is nitrate+nitrite
  reported **as N**; the unit bases differ and no conversion is invented.
* **Total Coliform** — **Not assessed**: IS 10500 (Table 6) requires total
  coliform to be *"not detectable in any 100 ml sample"* (presence/absence),
  whereas this input is a numeric **MPN/100 ml** count; the method bases
  differ.
* **Dissolved Oxygen, Conductivity, BOD** — **omitted**: IS 10500 prescribes
  no drinking-water limit for them (conductivity is covered only indirectly
  through Total dissolved solids, 500 / 2000 mg/l, Table 2, Sl. No. vi; DO
  and BOD are absent). No substitute or converted limit is assumed.

> **This is not a safety certification.** Checking six parameters — of which
> only pH currently has a directly comparable IS 10500 limit — does **not**
> establish that water is safe to drink. The full standard covers many other
> chemical, bacteriological, virological and radioactive parameters together
> with specific sampling and analysis methods; the app's output is a limited
> screening aid, not compliance testing.

### Relationship to Paper 6

Paper 6 (*Scientific Reports* 2025, 15:31139) predicts a **Weighted
Arithmetic WQI (Brown et al. 1972)** with a stacked ensemble and SHAP-based
explainable AI. This project is **aligned with, but not a reproduction of**
that work:

| Aspect | Paper 6 | This project |
|---|---|---|
| WQI rating \(q_i\) | continuous, vs ideal and permissible limits | discrete five bands, no ideal values |
| Aggregation | \(\sum q_iw_i/\sum w_i\) | \(\sum n_iw_i\) (un-normalised) |
| Parameters | 7, incl. faecal coliform | 6, excl. faecal coliform |
| Missing data | median imputation + IQR | row deletion on the 6 index parameters |
| Split | stratified 80:20, n = 390 | random 80:20, n = 1593 |
| Models | 6 base learners + LinearRegression meta-learner | 4 regressors, Gradient Boosting selected |
| Explainability | SHAP global + local (bar, dependence, force, distribution) | SHAP summary + per-prediction contributions in the UI |
| Interface | none reported | Flask app + local LLM narrative |

Paper 6's numeric unit weights, permissible limits, WQI class thresholds and
per-model hyperparameters appear only in its **Supplementary material**
(Tables 1S, 2S, 3S–11S), which was not available. The weights and band edges
above are therefore **this project's own index definition** and are not
attributed to the paper.

---

## Current results

Dataset: `data/water_dataX.csv`, 1991 raw rows → **1593 modelled rows**
(89 invalid-pH rows removed, exact duplicate rows removed, then 1 duplicate
parameter vector removed; 0 missing values remain in the six parameters).

Train/test = 1274 / 319 (`random_state=42`). Leakage audit: **0** identical
parameter vectors shared between splits.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | 0.4793 | 0.7851 | 0.9968 |
| Random Forest | 0.5399 | 1.3154 | 0.9910 |
| Decision Tree | 0.7177 | 2.1942 | 0.9750 |
| Linear Regression | 8.0387 | 9.3294 | 0.5477 |

Tuned Gradient Boosting (`GridSearchCV`, 5-fold, scoring R²):
`learning_rate=0.1`, `max_depth=2`, `min_samples_leaf=1`,
`min_samples_split=2`, `n_estimators=150`; best CV R² = 0.9933;
test **MAE = 0.5115**, **RMSE = 0.8516**, **R² = 0.9962**.

Mean WQI 73.89, median 76.34, std 14.11, observed range 19.3–99.62.
Feature importance / SHAP rank: Total Coliform > Dissolved Oxygen > pH >
BOD > Conductivity > Nitrate. This ordering agrees with Paper 6, which
identifies DO, BOD, conductivity and pH as the dominant WQI drivers.

**These numbers are not comparable to Paper 6's** (R = 0.9952, RMSE = 1.0704):
different dataset, different target definition, different sample size.

---

## Evaluation protocol & extended experiments

`notebooks/eval_experiments.ipynb` re-derives the cleaning + WQI pipeline
**read-only** (it never writes to `model/`) and runs a stronger evaluation on
one frozen split (`train_test_split`, `test_size=0.2`, `random_state=42`;
1274 / 319) shared by every model.

**Data-validity gate.** Before any experiment the notebook asserts the
independently re-derived facts: raw 1991 rows, raw pH max **67115** (a sensor /
entry artifact — 89 rows lie outside `[0,14]` and are removed, leaving a cleaned
pH range of **`[2.6, 9.01]`**); the row chain
`1991 → 1902 → 1901 → 1594 → 1593`; 0 missing values after cleaning; the final
feature order equals `model/features.pkl`; **238** distinct WQI values; and
**determinism** — 0 of the 1593 unique input vectors map to more than one WQI,
proving the label is a pure function of the six inputs. Any drift fails loudly.

**Baseline + cross-validation (5-fold, mean ± std) + held-out test:**

| Model | Test MAE | Test RMSE | Test R² | CV R² (mean ± std) |
|---|---|---|---|---|
| Mean baseline (`DummyRegressor`) | 11.5459 | 13.9122 | −0.0058 | −0.0093 ± 0.0070 |
| Linear Regression | 8.0387 | 9.3294 | 0.5477 | 0.5523 ± 0.0289 |
| Decision Tree | 0.7177 | 2.1942 | 0.9750 | 0.9686 ± 0.0064 |
| Random Forest | 0.5399 | 1.3154 | 0.9910 | 0.9764 ± 0.0108 |
| Gradient Boosting (untuned) | 0.4793 | 0.7851 | 0.9968 | 0.9907 ± 0.0052 |
| **Gradient Boosting (tuned)** | **0.5115** | **0.8516** | **0.9962** | **0.9927 ± 0.0036** |
| Stacking (DT+RF+tunedGB / Ridge) | 0.4989 | 0.8486 | 0.9963 | 0.9928 ± 0.0034 |

Hyperparameter tuning uses **only the training set** (`GridSearchCV`, cv=5,
scoring R²); the best configuration's CV R² is reported as **mean ± std**
(0.99325 ± 0.00211), not a single number. The re-measured tuned-GB test metrics
(0.5115 / 0.8516 / 0.9962) independently **confirm the previously reported**
0.51155 / 0.85160 / 0.99623.

**Error analysis.** Residual mean ≈ 0.04, std ≈ 0.85, max |residual| ≈ 4.26.
Error is heteroscedastic across the target range — MAE falls from ≈ 1.16 in the
lowest-WQI quintile to ≈ 0.22 in the highest — and is larger for samples whose
inputs sit near a rating-band boundary (mean |error| ≈ 0.59 near vs ≈ 0.43 far,
2 % window), consistent with the five-level quantisation of the index.

**Stacking (pre-registered rule).** A `StackingRegressor` with out-of-fold base
predictions was compared to the tuned GB on the same test set. It is **retained
only if** test RMSE improves ≥ 5 % relative **and** the CV-R² gain exceeds the
CV std. Measured gain was **0.35 % RMSE** and CV-R² gain **≈ 1e-5** (< CV std
0.0036), so **stacking is not adopted** and the deployed model is unchanged —
the honest expected outcome for a deterministic target.

Run it headlessly with:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\eval_experiments.ipynb
```

---

## Limitations

1. The label is computed deterministically from the six inputs (proved in the
   evaluation notebook: 0 input vectors map to more than one WQI), so the
   regressor is a **surrogate for a known formula**; high R² measures formula
   recovery, **not** independent prediction of real-world water quality. A model
   cannot be claimed to "outperform" the exact WQI formula — the formula *is* the
   ground-truth label.
2. Quantised ratings mean many distinct samples share one WQI value.
3. The index is un-normalised (\(\sum w_i = 0.998\)).
4. Faecal coliform and temperature are excluded from the index.
5. Rows are deleted rather than imputed; no IQR outlier treatment and no
   feature normalization (tree models do not need it).
6. A stacked ensemble was **evaluated but not adopted** (it failed the
   pre-registered improvement bar); no external or temporal validation is used.
7. Weights and band edges are project-defined, not derived from a cited
   regulatory table.
8. The parameter-level guideline check compares only the six model inputs
   against IS 10500:2012; only pH has a directly comparable limit, so nitrate
   and coliform are reported "Not assessed" (unit / method-basis mismatch) and
   DO, conductivity and BOD are outside IS 10500's drinking-water scope. It is
   a screening aid, not compliance testing, and confers no potability
   conclusion.
