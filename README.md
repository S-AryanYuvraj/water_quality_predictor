# AI-Based Water Quality Prediction and Assessment System

Predicts a **Water Quality Index (WQI)** from seven physico-chemical /
microbial parameters using a gradient-boosting regressor. The target is a
**Brown-style Weighted Arithmetic WQI (WAWQI)** (a **lower** value means
**better** water), reported as a numeric index plus a quality class, with a
**per-parameter standard check**, **SHAP** explanations, and a plain-language
narrative from a local **Ollama / Llama 3.2** model.

The index, standards, cleaning and classification all live in one module,
**`wawqi.py`**, which is imported by the notebooks *and* `app.py`, so the
training target and the served model can never diverge.

> **Not a reproduction of Paper 6.** This is a *corrected, scientifically
> defensible* Brown-style WAWQI. It follows the paper's equation framework but
> fixes several defects in its published code (see *Deviations*). No claim of
> byte-exact reproduction is made.

---

## Project layout

```
WaterQualityPred/
├── app.py                     Flask application (prediction + SHAP + Ollama)
├── wawqi.py                   Brown WAWQI: equations, standards, cleaning, tests
├── data/water_dataX.csv       raw input dataset
├── notebooks/WqiPred.ipynb    clean -> WAWQI target -> train/tune -> SHAP -> save
├── notebooks/eval_experiments.ipynb  read-only evaluation: baseline, CV, error, stacking
├── model/wqi_model.pkl        trained GradientBoostingRegressor (7 inputs)
├── model/features.pkl         feature column order (must match the model)
├── templates/index.html
├── static/css/style.css
└── backups/                   pre-change snapshots (git-ignored)
```

---

## Running

From the **repository root**:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. `app.py` and `wawqi.py` are both at the root, so
the app imports the index module directly.

Validate the index maths independently (no model required):

```powershell
python wawqi.py        # runs the self-test battery; exits non-zero on failure
```

### Optional: AI narrative (Ollama)

Prediction, class and SHAP work without Ollama; only the explanation panel uses
it (`ollama serve` + `ollama pull llama3.2`). Env vars: `OLLAMA_URL`
(default `http://localhost:11434/api/generate`), `OLLAMA_MODEL` (`llama3.2`),
`OLLAMA_TIMEOUT` (`120`). If unreachable, the app degrades gracefully.

## Retraining

The notebook reads `../data/water_dataX.csv` and writes `../model/*.pkl`, so it
must run **from `notebooks/`** (it adds `..` to `sys.path` to import `wawqi`):

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\WqiPred.ipynb
```

`app.py` asserts at start-up that `features.pkl` matches the model's input
width, so notebook/app drift fails fast.

---

## Methodology

### The WQI equation (Brown et al., 1972 — Paper 6's framework, corrected)

For each parameter *i* a sub-index \(q_i\) is scored so \(q_i=0\) at the ideal
value and \(q_i=100\) at the permissible limit \(S_i\):

* **higher-is-worse** (ideal 0) — EC, BOD, Nitrate, Fecal & Total coliform:
  \[ q_i = 100\cdot\frac{V_i}{S_i} \]
* **Dissolved Oxygen** (higher-is-better; standard is a *minimum*):
  \[ q_{DO} = 100\cdot\frac{V_{ideal}-V}{V_{ideal}-S_{min}},\quad V_{ideal}=14.6 \]
* **pH** (two-sided about 7):
  \[ q_{pH} = 100\cdot\frac{|V_i-7|}{S_i-7} \]

Unit weights are inverse-permissible-value, and the index is the weight-
normalised mean of the sub-indices (the constant \(k=1/\sum S_i\) cancels):

\[ w_i=\frac{k}{S_i},\qquad \text{WQI}=\frac{\sum_i q_i w_i}{\sum_i w_i} \]

All sub-indices are **clipped to [0, 100]**, so WQI \(\in[0,100]\).

### Standards (see `wawqi.SOURCE` for per-parameter provenance)

| Parameter | Ideal | Permissible S | Unit weight | Source |
|---|---|---|---|---|
| Dissolved Oxygen | 14.6 | ≥ 6.0 mg/L (min) | 0.18239 | **CPCB Class A** |
| pH | 7.0 | 6.5–8.5 | 0.12874 | general surface-water band |
| Conductivity | 0 | ≤ 1000 µS/cm | 0.00109 | adopted guideline (**not** a Class A figure) |
| BOD | 0 | ≤ 2.0 mg/L | 0.54716 | **CPCB Class A** |
| Nitrate + Nitrite | 0 | ≤ 11.3 mg/L **as N** | 0.09684 | WHO 50 mg/L as NO₃ × 0.2259 |
| Fecal coliform | 0 | ≤ 50 MPN/100 mL | 0.02189 | consistent with Class A total-coliform |
| Total coliform | 0 | ≤ 50 MPN/100 mL | 0.02189 | **CPCB Class A** |

The designated-best-use choice is **CPCB Class A** (the most protective). Where
Class A gives no value, an authoritative unit-compatible source is used and
flagged; standards from different use-classes are **not** silently mixed.

### Direction of the index (important)

**LOWER WQI = BETTER** water: **0–25 Excellent, 25–50 Good, 50–75 Poor,
75–100 Very Poor**. Verified on the data: WQI correlates **−0.33** with
dissolved oxygen (more DO → lower/better) and **+0.22** with BOD (more BOD →
higher/worse). This is re-asserted by the evaluation notebook's validity gate.

### Interface output — numeric WQI + class

The app shows the numeric WQI **and** its quality class (e.g. `48.26 — Good`),
the seven parameter standard checks, and the SHAP contributions. The class
comes from `wawqi.classify_wqi`, shared with the pipeline.

### Data cleaning (`wawqi.clean_and_index`)

A single documented path: coerce to numeric → remove **demonstrably
impossible** records only (pH outside [0,14]; any negative concentration/count)
→ drop exact duplicate rows → **complete-case** handling of missing values on
the seven parameters → drop duplicate parameter-vectors (leakage control) →
compute the index. Physically-possible but **polluted** samples are **kept**
(exceedances show up as clipped \(q_i=100\)); the model therefore learns across
the full quality range.

### Deviations from Paper 6's `t2.docx` implementation (deliberate)

1. **DO** uses a *minimum* standard in the inverted form; the paper used
   S_DO = 10 as an upper cap with ideal 14.6, forcing \(q_{DO}\ge100\) for every
   in-range sample.
2. **pH** uses \(|V-ideal|\); the paper omitted the absolute value, giving
   negative sub-indices for pH < 7.
3. Sub-indices are **clipped to [0, 100]**; the paper applied no bounds.
4. **Nitrate** standard is on the **as-N** basis matching the data column; the
   paper applied 45 mg/L (an as-NO₃ figure) to an as-N column.
5. **Cleaning keeps polluted samples**; Paper 6's 390-row pipeline additionally
   filtered every row to within permissible ranges *before* indexing, discarding
   the very samples a WQI model must learn.

---

## Per-parameter standard check

The app's table compares each input against the **same permissible value the
WAWQI uses** (CPCB Class A where it exists, otherwise the cited source), and
shows the sub-index \(q_i\) (0 = ideal, 100 = permissible limit). This is a
view of the index inputs, **separate from** the aggregate WQI, and confers no
potability conclusion. (The earlier IS 10500:2012 drinking-water comparison was
removed because the index now rests on surface-water Class A criteria.)

---

## Current results

Dataset `data/water_dataX.csv`: **1991 raw rows → 1571 modelled rows**
(89 impossible records removed, 1 all-column duplicate, 329 rows dropped by
complete-case missing handling — largely faecal coliform, 1 duplicate parameter
vector). Train/test = **1256 / 319** (`random_state=42`); leakage audit **0**.

WAWQI distribution: **min 22.39, mean 67.96, median 72.88, max 92.28,
std 15.74**; 1566 distinct values (the index is continuous). Class counts:
Very Poor 742, Poor 544, Good 284, Excellent 1.

**Held-out test metrics** (single frozen split, `notebooks/eval_experiments.ipynb`):

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Mean baseline (Dummy) | 14.0451 | 16.0046 | −0.0014 |
| Linear Regression | 12.0798 | 14.5282 | 0.1748 |
| Decision Tree | 1.2108 | 1.7133 | 0.9885 |
| Random Forest | 0.8530 | 1.2508 | 0.9939 |
| Gradient Boosting (untuned) | 0.5727 | 0.7582 | 0.9978 |
| **Gradient Boosting (tuned)** | **0.4690** | **0.6504** | **0.9983** |
| Stacking (DT+RF+GBtuned / Ridge) | 0.4876 | 0.6688 | 0.9983 |

Tuned GB (`GridSearchCV`, cv=5, R²): `learning_rate=0.1, max_depth=3,
min_samples_leaf=2, min_samples_split=5, n_estimators=150`; best CV R² =
**0.99751 ± 0.00052**; test MAE 0.4690 / RMSE 0.6504 / R² 0.99835. Residual
mean 0.026, std 0.650, max |resid| 4.03.

GB feature importance: **BOD 0.956 ≫ pH 0.025 > DO 0.009 > Nitrate 0.008 >
Fecal 0.001 ≈ Total 0.001 > EC 0.0001** — consistent with the inverse-standard
unit weights (BOD carries the largest weight).

**Stacking was evaluated but NOT adopted** (pre-registered rule): test-RMSE gain
−2.83 % (no improvement) and CV-R² gain 0.0001 < CV std. The deployed model is
the tuned Gradient Boosting.

Run the evaluation headlessly:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\eval_experiments.ipynb
```

---

## Limitations

1. **The label is a deterministic function of the seven inputs** (proved by the
   validity gate: 0 input vectors map to more than one WQI). The regressor is a
   **surrogate for a known formula**; high R² measures formula recovery, **not**
   independent prediction of real-world water quality, and cannot be claimed to
   "outperform" the WQI equation.
2. **Standards provenance is partial.** DO, BOD and Total coliform use CPCB
   Class A; pH, conductivity, nitrate and fecal coliform rely on cited
   supplements or documented derivations (see `wawqi.SOURCE`). Conductivity has
   **no** Class A limit — the adopted 1000 µS/cm is a conservative guideline,
   not a Class A figure.
3. **Nitrate basis.** The data column is nitrate+nitrite *as N*; the standard
   is converted from the WHO as-NO₃ guideline (×0.2259). The conversion factor
   uses definitional molar masses but the WHO limit is a *drinking-water*
   guideline applied to a surface-water index.
4. **Coliform sub-indices saturate** on the linear \(q=100V/S\) scale (clipped
   at 100); a log transform would discriminate better but is a deviation not
   adopted here.
5. Complete-case deletion (not imputation) is used; the faecal-coliform
   requirement drops ~330 rows. No outlier capping — clipping handles extremes.
6. The index is a **screening aggregate**, not a safety certification; only
   seven parameters are covered.
7. Results are **not numerically comparable to Paper 6** (different cleaning,
   n, model set, and corrected index).

## Environment

`scikit-learn 1.9.0`, `shap 0.52.0`, `pandas 3.0.5`, `numpy 2.5.2`.
