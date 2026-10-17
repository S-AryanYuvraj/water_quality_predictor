"""Brown-style Weighted Arithmetic Water Quality Index (WAWQI).

A scientifically defensible, corrected implementation of the Weighted
Arithmetic WQI framework described in ``article6.pdf`` (Paper 6) and its
supplementary files ``t1.docx`` (Table 1S classification, Table 2S
permissible limits) and ``t2.docx`` (reference Python code).

This module intentionally CORRECTS documented defects in the paper's own
code (see DEVIATIONS below). It is a Brown-STYLE WAWQI, NOT a claim of
byte-exact reproduction of Paper 6's numbers.

Framework (Brown et al., 1972; identical equations 1-4 to the paper)
--------------------------------------------------------------------
    q_i = sub-index of parameter i, bounded to [0, 100]
    w_i = k / S_i                     (unit weight, inverse permissible value)
    WQI = sum_i(q_i * w_i) / sum_i(w_i)
    k   = 1 / sum_i(S_i)              (cancels under the sum(w_i) normalisation)

Direction: LOWER WQI = BETTER water (q_i = 0 at the ideal value, 100 at the
permissible limit). 0-25 Excellent, 25-50 Good, 50-75 Poor, 75-100 Very Poor.

Sub-index forms by parameter direction (the two-sided / minimum handling the
paper's code got wrong):
    higher-is-worse, ideal 0 (EC, BOD, Nitrate, Fecal, Total):
        q = 100 * (V - 0) / (S - 0) = 100 * V / S
    higher-is-better, DO (standard is a MINIMUM, ideal = saturation 14.6):
        q = 100 * (V_ideal - V) / (V_ideal - S_min)
    two-sided, pH (ideal 7, permissible band 6.5-8.5):
        q = 100 * abs(V - 7) / (S_upper - 7)

DEVIATIONS from Paper 6's t2.docx implementation (documented, deliberate):
  1. DO uses a MINIMUM standard in the inverted form; the paper used S_DO=10
     as an upper cap with ideal 14.6, forcing q_DO >= 100 for every in-range
     sample (a sign/range defect).
  2. pH uses |V - ideal|; the paper omitted the absolute value, producing
     NEGATIVE q for pH < 7.
  3. Sub-indices are CLIPPED to [0, 100]; the paper applied no bounds.
  4. Nitrate standard is expressed on the SAME chemical basis as the data
     column (nitrate+nitrite AS N). The paper applied 45 mg/L (an as-NO3
     figure) to an as-N column, a ~4x unit-basis error.

Standards provenance (CPCB Class A where it exists; unit-compatible
authoritative sources otherwise - see SOURCE dict and README). Parameters
not covered by CPCB Class A are flagged individually.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ============================================================
# CANONICAL COLUMN NAMES (dataset order == model feature order)
# ============================================================
# The electrical-conductivity column name contains U+00B5 (MICRO SIGN); it is
# written as an escape so this file is encoding-independent and matches the
# latin-1-decoded CSV exactly.
DO = "D.O. (mg/l)"
PH = "PH"
EC = "CONDUCTIVITY (\u00b5mhos/cm)"
BOD = "B.O.D. (mg/l)"
NO3 = "NITRATENAN N+ NITRITENANN (mg/l)"
FC = "FECAL COLIFORM (MPN/100ml)"
TC = "TOTAL COLIFORM (MPN/100ml)Mean"

# The single authoritative parameter list. Defined once; reused by cleaning,
# index calculation, model training and the web app so they cannot diverge.
PARAMS = [DO, PH, EC, BOD, NO3, FC, TC]

# ============================================================
# STANDARDS (permissible value S_i) AND IDEALS (V_ideal)
# ============================================================
# S_i drives BOTH the sub-index denominator and the inverse-standard weight,
# exactly as in Brown's Eqs (2)-(3). Values are the CPCB "Designated Best Use"
# Class A limits where Class A provides them, and cited authoritative sources
# where it does not (see SOURCE). DO is a MINIMUM; the others are maxima; pH is
# the upper end of its permissible band (the two-sided form uses |V - ideal|).
STANDARDS = {
    DO: 6.0,       # mg/L   CPCB Class A (minimum DO)
    PH: 8.5,       # -      permissible band 6.5-8.5 (upper bound)
    EC: 1000.0,    # uS/cm  general freshwater / potable-source guideline (NOT a Class A figure - see SOURCE)
    BOD: 2.0,      # mg/L   CPCB Class A
    NO3: 11.3,     # mg/L AS N  (WHO 50 mg/L as NO3- x 14.007/62.004 = 11.29)
    FC: 50.0,      # MPN/100mL  consistent with CPCB Class A total-coliform limit (fecal subset of total)
    TC: 50.0,      # MPN/100mL  CPCB Class A
}

IDEALS = {
    DO: 14.6,      # mg/L  DO near saturation at ~20 C (paper's own ideal)
    PH: 7.0,       # -     neutral
    EC: 0.0,
    BOD: 0.0,
    NO3: 0.0,
    FC: 0.0,
    TC: 0.0,
}

DIRECTION = {
    DO: "min",          # higher is better -> inverted sub-index
    PH: "two_sided",    # deviation from 7 in either direction
    EC: "max",
    BOD: "max",
    NO3: "max",
    FC: "max",
    TC: "max",
}

UNITS = {
    DO: "mg/L",
    PH: "pH",
    EC: "\u00b5S/cm (= \u00b5mho/cm)",
    BOD: "mg/L",
    NO3: "mg/L as N",
    FC: "MPN/100mL",
    TC: "MPN/100mL",
}

# Per-parameter standard provenance (verbatim source vs adopted/converted is
# distinguished; nothing is silently invented).
SOURCE = {
    DO: "CPCB Designated Best Use, Class A (DO >= 6.0 mg/L) [fetched official table]",
    PH: "General inland-surface-water permissible band 6.5-8.5 (CPCB criterion; not in the fetched abbreviated Class A table)",
    EC: "ADOPTED PROJECT threshold (1000 uS/cm) - NOT an official CPCB Class A limit; Class A specifies no EC value and Class E (2250, irrigation) was deliberately not used",
    BOD: "CPCB Designated Best Use, Class A (BOD <= 2.0 mg/L)",
    NO3: "WHO *drinking-water* guideline 50 mg/L as NO3- converted to the as-N basis (x 0.2259) = 11.3 mg/L as N to match the data column; a drinking-water limit adopted here for this surface-water index",
    FC: "ADOPTED PROJECT threshold (50 MPN/100mL) - NOT an official CPCB Class A fecal limit (Class A lists none); equals the Class A total-coliform limit since fecal coliforms are a subset of total",
    TC: "CPCB Designated Best Use, Class A (Total coliform <= 50 MPN/100mL)",
}

# Nitrate conversion factor (definitional molar masses, not an assumption):
#   NO3-N fraction = M(N) / M(NO3) = 14.007 / (14.007 + 3*15.999) = 14.007/62.005
NITRATE_AS_N_FACTOR = 14.007 / (14.007 + 3 * 15.999)  # ~0.2259

# ============================================================
# SUB-INDEX
# ============================================================


def sub_index(param: str, value: float) -> float:
    """Return the clipped [0, 100] sub-index q_i for one parameter.

    q = 0 at the ideal value and 100 at the permissible limit. Returns NaN for
    a NaN input so missing data are never silently treated as ideal.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")

    value = float(value)
    s = STANDARDS[param]
    ideal = IDEALS[param]
    direction = DIRECTION[param]

    if direction == "max":
        q = 100.0 * (value - ideal) / (s - ideal)
    elif direction == "min":
        q = 100.0 * (ideal - value) / (ideal - s)
    elif direction == "two_sided":
        q = 100.0 * abs(value - ideal) / (s - ideal)
    else:  # pragma: no cover - guarded by DIRECTION definition
        raise ValueError(f"unknown direction {direction!r} for {param!r}")

    # Clip to the definitional range (DEVIATION 3).
    return float(min(max(q, 0.0), 100.0))


def raw_weights() -> dict:
    """Unit weights w_i = k / S_i (k=1; it cancels in the normalised WQI)."""
    return {p: 1.0 / STANDARDS[p] for p in PARAMS}


def normalized_weights() -> dict:
    """Weights normalised to sum to 1, for reporting."""
    w = raw_weights()
    total = sum(w.values())
    return {p: w[p] / total for p in PARAMS}


def compute_wawqi(values) -> float:
    """Weighted-arithmetic WQI for one sample.

    ``values`` may be a mapping (column name -> value) or a pandas row/Series
    containing the PARAMS columns.
    """
    w = raw_weights()
    num = 0.0
    den = 0.0
    for p in PARAMS:
        q = sub_index(p, values[p])
        if math.isnan(q):
            return float("nan")
        num += q * w[p]
        den += w[p]
    return float(num / den)


# ============================================================
# CLASSIFICATION (Table 1S, corrected to contiguous bands)
# ============================================================


def classify_wqi(wqi: float) -> str:
    """Map a WAWQI value to a quality class. LOWER is BETTER.

    Bands are made contiguous (the printed Table 1S had a 51-55 gap and a
    label mismatch between t1.docx and t2.docx; resolved here as documented).
    """
    if wqi is None or (isinstance(wqi, float) and math.isnan(wqi)):
        return "Unknown"
    if wqi <= 25.0:
        return "Excellent"
    if wqi <= 50.0:
        return "Good"
    if wqi <= 75.0:
        return "Poor"
    return "Very Poor"


# ============================================================
# DOCUMENTED DATA-CLEANING PIPELINE
# ============================================================


def clean_and_index(df: pd.DataFrame):
    """Clean raw water-quality data and add the WAWQI target column.

    Procedure (single documented path, shared by every consumer):
      1. Coerce the 7 parameters to numeric (invalid strings -> NaN).
      2. Remove DEMONSTRABLY impossible records only:
           - pH outside the physical [0, 14] scale (sensor/entry artifacts),
           - any negative concentration / count.
         Physically-possible but polluted samples are KEPT (removing them, as
         Paper 6's permissible-range filter did, deletes the very samples a WQI
         model must learn from). Exceedances are expressed via the clipped
         sub-indices instead.
      3. Drop exact duplicate rows (all columns).
      4. Complete-case handling of missing values on the 7 parameters
         (dropna) - a single, stated policy; no imputation.
      5. Drop duplicate parameter-vectors on the 7 parameters (leakage control).
      6. Compute WAWQI per row.

    Returns (wqi_df, info) where info holds the row counts at each step.
    """
    info: dict = {}
    d = df.copy()

    for p in PARAMS:
        d[p] = pd.to_numeric(d[p], errors="coerce")
    info["raw"] = int(len(d))

    # Step 2: impossible values only.
    impossible = (d[PH] < 0) | (d[PH] > 14)
    for p in (DO, EC, BOD, NO3, FC, TC):
        impossible = impossible | (d[p] < 0)
    n_invalid = int(impossible.sum())
    d = d[~impossible]
    info["removed_impossible"] = n_invalid
    info["after_invalid"] = int(len(d))

    # Step 3: exact duplicate rows.
    before = len(d)
    d = d.drop_duplicates()
    info["removed_allcol_dupes"] = int(before - len(d))

    # Step 4: complete-case on the 7 parameters.
    before = len(d)
    d = d.dropna(subset=PARAMS)
    info["removed_missing"] = int(before - len(d))
    info["after_dropna"] = int(len(d))

    # Step 5: duplicate parameter-vector (leakage control).
    before = len(d)
    d = d.drop_duplicates(subset=PARAMS)
    info["removed_vector_dupes"] = int(before - len(d))

    # Step 6: index.
    d = d.copy()
    d["WQI"] = d.apply(compute_wawqi, axis=1)
    info["final_n"] = int(len(d))

    return d, info


# ============================================================
# SELF-TESTS
# ============================================================


def _row(overrides=None) -> dict:
    """A fully-ideal sample with selected parameters overridden.

    ``overrides`` is a dict keyed by the PARAMS column-name strings (NOT by
    the short module constants, whose names differ from their values).
    """
    base = {p: IDEALS[p] for p in PARAMS}
    if overrides:
        base.update(overrides)
    return base


def run_tests(verbose: bool = True) -> int:
    """Run the validation battery; returns the number of failures (0 == pass)."""
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)
            if verbose:
                print(f"  FAIL: {name}")
        elif verbose:
            print(f"  ok:   {name}")

    # No division-by-zero: every S differs from its ideal.
    for p in PARAMS:
        check(f"S!=ideal for {p}", STANDARDS[p] != IDEALS[p])

    # Weights normalise to 1.
    wn = normalized_weights()
    check("weights sum to 1", abs(sum(wn.values()) - 1.0) < 1e-12)

    # --- ideal water -> WQI 0 (best) ---
    check("ideal water WQI == 0", abs(compute_wawqi(_row()) - 0.0) < 1e-9)

    # --- poor water -> high WQI, still <= 100 ---
    poor = _row({DO: 0.5, PH: 4.0, EC: 3000, BOD: 20, NO3: 80,
                 FC: 500000, TC: 2_000_000})
    pw = compute_wawqi(poor)
    check("poor water WQI high (>75)", pw > 75)
    check("poor water WQI <= 100", pw <= 100 + 1e-9)

    # --- single parameter at its standard equals its normalized weight * 100 ---
    for p in PARAMS:
        if DIRECTION[p] == "min":
            val = STANDARDS[p]  # DO at its minimum -> q=100
        elif DIRECTION[p] == "two_sided":
            val = STANDARDS[p]  # pH at upper bound -> q=100
        else:
            val = STANDARDS[p]  # others at max -> q=100
        wqi_one = compute_wawqi(_row({p: val}))
        expected = wn[p] * 100.0
        check(f"{p} at standard -> q=100 & WQI=w*100",
              abs(sub_index(p, val) - 100.0) < 1e-9
              and abs(wqi_one - expected) < 1e-9)

    # --- pH symmetry (DEVIATION 2 fix: no negative q, symmetric about 7) ---
    check("pH 6.8 == 7.2 sub-index",
          abs(sub_index(PH, 6.8) - sub_index(PH, 7.2)) < 1e-9)
    check("pH 7 -> q=0", sub_index(PH, 7.0) == 0.0)
    check("pH sub-index never negative",
          sub_index(PH, 5.0) >= 0 and sub_index(PH, 9.5) >= 0)

    # --- DO direction (DEVIATION 1 fix: higher DO -> lower/better q) ---
    check("DO at ideal -> q=0", abs(sub_index(DO, 14.6) - 0.0) < 1e-9)
    check("DO at min standard -> q=100", abs(sub_index(DO, 6.0) - 100.0) < 1e-9)
    check("DO below min clipped to 100", sub_index(DO, 2.0) == 100.0)
    check("DO above ideal clipped to 0", sub_index(DO, 15.5) == 0.0)
    check("DO monotonic decreasing q",
          sub_index(DO, 5.0) > sub_index(DO, 8.0) > sub_index(DO, 12.0))

    # --- parameter exceedance clipped to 100 (DEVIATION 3) ---
    check("BOD huge -> q clipped 100", sub_index(BOD, 1e6) == 100.0)
    check("EC huge -> q clipped 100", sub_index(EC, 1e6) == 100.0)

    # --- finite output for absurd coliform counts (no inf/nan) ---
    absurd = _row({FC: 272_521_616, TC: 511_090_873})
    aw = compute_wawqi(absurd)
    check("absurd coliform -> finite WQI",
          np.isfinite(aw) and 0 <= aw <= 100)

    # --- nitrate as-N basis (DEVIATION 4) ---
    check("nitrate 11.3 as N -> q=100", abs(sub_index(NO3, 11.3) - 100.0) < 1e-9)
    check("nitrate WHO conversion ~0.2259", abs(NITRATE_AS_N_FACTOR - 0.2259) < 5e-4)

    # --- monotonicity of the full index ---
    prev = compute_wawqi(_row({BOD: 1.0}))
    for b in (1.5, 2.0, 4.0, 10.0):
        cur = compute_wawqi(_row({BOD: b}))
        check(f"BOD {b} -> WQI non-decreasing", cur >= prev - 1e-9)
        prev = cur
    prev = compute_wawqi(_row({DO: 14.0}))
    for do in (12.0, 10.0, 8.0, 6.0):
        cur = compute_wawqi(_row({DO: do}))
        check(f"DO {do} -> WQI non-decreasing (worse)", cur >= prev - 1e-9)
        prev = cur

    # --- determinism ---
    check("determinism", compute_wawqi(poor) == compute_wawqi(poor))

    # --- classification direction & contiguity ---
    check("classify 0 Excellent", classify_wqi(0) == "Excellent")
    check("classify 100 Very Poor", classify_wqi(100) == "Very Poor")
    check("classify lower is better",
          classify_wqi(20) == "Excellent" and classify_wqi(60) == "Poor")

    if verbose:
        total_msg = f"\n{len(failures)} failure(s)."
        print("=" * 50)
        print("WAWQI self-test:", "ALL PASSED" if not failures else "FAILED")
        print(total_msg)

    return len(failures)


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_tests() else 0)
