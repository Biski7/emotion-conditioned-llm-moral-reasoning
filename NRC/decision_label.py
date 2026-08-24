"""
Decision Transition Taxonomy Analysis
Computes Moral Recovery, Unchanged Rate, and Harmful Disruption
for Anger+Age and Sadness+Age conditions.
"""

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
# model = 'gptoss'
model = 'gptoss'

if model == 'gptoss':
    ANGER_AGE_CSV   = "anger_age_emotion_results_full_gpt_oss_1.csv"
    SADNESS_AGE_CSV = "sadness_age_emotion_results_full_gpt_oss_1.csv"
elif model == 'llama4':
    ANGER_AGE_CSV   = "anger_age_emotion_results_full_llama4_1.csv"
    SADNESS_AGE_CSV = "sadness_age_emotion_results_full_llama4_1.csv"

# ── COLUMN NAMES ──────────────────────────────────────────────────────────────
# Adjust these if your CSVs use different column names
GOLD_COL     = "Label"               # ground truth  (0 or 1)
BASE_COL     = "Base0-shot"              # baseline prediction
ANGER_AGE_COL   = "AngerAge_Label"    # anger+age prediction
SADNESS_AGE_COL = "SadnessAge_Label"  # sadness+age prediction


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_and_clean(path, condition_col):
    """Load CSV and coerce label columns to int."""
    df = pd.read_csv(path)
    print(f"\n[INFO] Loaded {path}")
    print(f"       Columns: {list(df.columns)}")
    print(f"       Shape  : {df.shape}")
    for col in [GOLD_COL, BASE_COL, condition_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    before = len(df)
    df = df.dropna(subset=[GOLD_COL, BASE_COL, condition_col])
    df[[GOLD_COL, BASE_COL, condition_col]] = df[[GOLD_COL, BASE_COL, condition_col]].astype(int)
    after = len(df)
    if before != after:
        print(f"       Dropped {before - after} rows with non-numeric labels.")
    return df


def compute_transitions(df, condition_col, condition_name):
    """
    Compute all three decision-transition categories.

    Notation: (Gold → Baseline → Conditioned)

    1) Moral Recovery
       1→0→1  : gold=1, base=0, cond=1  (wrong correctly recovered)
       0→1→0  : gold=0, base=1, cond=0  (acceptable correctly recovered)

    2) Unchanged Decision Rate
       0→0→0  : gold=0, base=0, cond=0
       1→1→1  : gold=1, base=1, cond=1
       0→1→1  : gold=0, base=1, cond=1  (error stays wrong)
       1→0→0  : gold=1, base=0, cond=0  (error stays wrong)

    3) Harmful Disruption
       0→0→1  : gold=0, base=0, cond=1  (correct→incorrect)
       1→1→0  : gold=1, base=1, cond=0  (correct→incorrect)
    """
    g = df[GOLD_COL]
    b = df[BASE_COL]
    c = df[condition_col]

    T_eval = len(df)

    # ── Moral Recovery ────────────────────────────────────────────────────────
    # Among baseline errors only
    base_errors = df[b != g]
    T_errors = len(base_errors)

    # 1→0→1  (missed "wrong", recovered by condition)
    rec_1_0_1 = base_errors[
        (base_errors[GOLD_COL] == 1) &
        (base_errors[BASE_COL] == 0) &
        (base_errors[condition_col] == 1)
    ]

    # 0→1→0  (missed "acceptable", recovered by condition)
    rec_0_1_0 = base_errors[
        (base_errors[GOLD_COL] == 0) &
        (base_errors[BASE_COL] == 1) &
        (base_errors[condition_col] == 0)
    ]

    n_rec_1_0_1 = len(rec_1_0_1)
    n_rec_0_1_0 = len(rec_0_1_0)
    n_rec_total = n_rec_1_0_1 + n_rec_0_1_0

    # Recovery rates are computed relative to the relevant baseline-error pool
    pool_1_0 = base_errors[(base_errors[GOLD_COL] == 1) & (base_errors[BASE_COL] == 0)]
    pool_0_1 = base_errors[(base_errors[GOLD_COL] == 0) & (base_errors[BASE_COL] == 1)]

    rate_1_0_1 = (n_rec_1_0_1 / len(pool_1_0) * 100) if len(pool_1_0) > 0 else 0.0
    rate_0_1_0 = (n_rec_0_1_0 / len(pool_0_1) * 100) if len(pool_0_1) > 0 else 0.0
    rate_rec_total = (n_rec_total / T_errors * 100) if T_errors > 0 else 0.0

    # ── Unchanged Decision Rate ───────────────────────────────────────────────
    unchanged = df[b == c]
    T_changed  = (b != c).sum()

    # Sub-patterns
    unch_0_0_0 = df[(g == 0) & (b == 0) & (c == 0)]
    unch_1_1_1 = df[(g == 1) & (b == 1) & (c == 1)]
    unch_0_1_1 = df[(g == 0) & (b == 1) & (c == 1)]   # error unchanged
    unch_1_0_0 = df[(g == 1) & (b == 0) & (c == 0)]   # error unchanged

    UR = (T_eval - T_changed) / T_eval * 100

    # ── Harmful Disruption ────────────────────────────────────────────────────
    # Among baseline-correct scenarios only
    base_correct = df[b == g]
    T_correct = len(base_correct)

    # 0→0→1  (was correct-accept, now wrongly called wrong)
    dis_0_0_1 = base_correct[
        (base_correct[GOLD_COL] == 0) &
        (base_correct[BASE_COL] == 0) &
        (base_correct[condition_col] == 1)
    ]

    # 1→1→0  (was correct-wrong, now wrongly called acceptable)
    dis_1_1_0 = base_correct[
        (base_correct[GOLD_COL] == 1) &
        (base_correct[BASE_COL] == 1) &
        (base_correct[condition_col] == 0)
    ]

    n_dis_0_0_1 = len(dis_0_0_1)
    n_dis_1_1_0 = len(dis_1_1_0)
    n_dis_total = n_dis_0_0_1 + n_dis_1_1_0

    rate_0_0_1  = (n_dis_0_0_1 / T_correct * 100) if T_correct > 0 else 0.0
    rate_1_1_0  = (n_dis_1_1_0 / T_correct * 100) if T_correct > 0 else 0.0
    rate_dis_total = (n_dis_total / T_correct * 100) if T_correct > 0 else 0.0

    # ── Print Report ──────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  MODEL: {model.upper()}   |   CONDITION: {condition_name}")
    print(sep)
    print(f"\n  Total scenarios evaluated (T_eval)  : {T_eval}")
    print(f"  Baseline correct                     : {T_correct}  ({T_correct/T_eval*100:.1f}%)")
    print(f"  Baseline errors                      : {T_errors}  ({T_errors/T_eval*100:.1f}%)")

    print(f"\n{'─'*64}")
    print(f"  1) MORAL RECOVERY")
    print(f"{'─'*64}")
    print(f"  1→0→1  (wrong recovered)     : {n_rec_1_0_1:>4}  /  {len(pool_1_0)}  "
          f"baseline errors  →  {rate_1_0_1:.1f}% recovery rate")
    print(f"  0→1→0  (accept recovered)    : {n_rec_0_1_0:>4}  /  {len(pool_0_1)}  "
          f"baseline errors  →  {rate_0_1_0:.1f}% recovery rate")
    print(f"  Total recovery               : {n_rec_total:>4}  /  {T_errors}  "
          f"baseline errors  →  {rate_rec_total:.1f}%")

    print(f"\n{'─'*64}")
    print(f"  2) UNCHANGED DECISION RATE (UR)")
    print(f"{'─'*64}")
    print(f"  Decisions changed            : {T_changed}")
    print(f"  Decisions unchanged          : {len(unchanged)}")
    print(f"  UR                           : {UR:.1f}%")
    print(f"    ↳ 0→0→0 (correct, stays)  : {len(unch_0_0_0)}")
    print(f"    ↳ 1→1→1 (correct, stays)  : {len(unch_1_1_1)}")
    print(f"    ↳ 0→1→1 (error, unchanged): {len(unch_0_1_1)}")
    print(f"    ↳ 1→0→0 (error, unchanged): {len(unch_1_0_0)}")

    print(f"\n{'─'*64}")
    print(f"  3) HARMFUL DISRUPTION")
    print(f"{'─'*64}")
    print(f"  0→0→1  (accept→wrong)        : {n_dis_0_0_1:>4}  /  {T_correct}  "
          f"baseline correct  →  {rate_0_0_1:.1f}%")
    print(f"  1→1→0  (wrong→accept)        : {n_dis_1_1_0:>4}  /  {T_correct}  "
          f"baseline correct  →  {rate_1_1_0:.1f}%")
    print(f"  Total disruption             : {n_dis_total:>4}  /  {T_correct}  "
          f"baseline correct  →  {rate_dis_total:.1f}%")
    print(f"\n{sep}\n")

    return {
        "model": model,
        "condition": condition_name,
        "T_eval": T_eval,
        "baseline_correct": T_correct,
        "baseline_errors": T_errors,
        "n_rec_1_0_1": n_rec_1_0_1,
        "n_rec_0_1_0": n_rec_0_1_0,
        "n_rec_total": n_rec_total,
        "rate_rec_1_0_1_pct": round(rate_1_0_1, 2),
        "rate_rec_0_1_0_pct": round(rate_0_1_0, 2),
        "rate_rec_total_pct": round(rate_rec_total, 2),
        "T_changed": int(T_changed),
        "T_unchanged": len(unchanged),
        "UR_pct": round(UR, 2),
        "unch_0_0_0": len(unch_0_0_0),
        "unch_1_1_1": len(unch_1_1_1),
        "unch_0_1_1": len(unch_0_1_1),
        "unch_1_0_0": len(unch_1_0_0),
        "n_dis_0_0_1": n_dis_0_0_1,
        "n_dis_1_1_0": n_dis_1_1_0,
        "n_dis_total": n_dis_total,
        "rate_dis_0_0_1_pct": round(rate_0_0_1, 2),
        "rate_dis_1_1_0_pct": round(rate_1_1_0, 2),
        "rate_dis_total_pct": round(rate_dis_total, 2),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    results = []

    # ── Anger + Age ──────────────────────────────────────────────────────────
    df_anger = load_and_clean(ANGER_AGE_CSV, ANGER_AGE_COL)
    results.append(compute_transitions(df_anger, ANGER_AGE_COL, "Anger + Age"))

    # ── Sadness + Age ────────────────────────────────────────────────────────
    df_sadness = load_and_clean(SADNESS_AGE_CSV, SADNESS_AGE_COL)
    results.append(compute_transitions(df_sadness, SADNESS_AGE_COL, "Sadness + Age"))

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SUMMARY TABLE")
    print("=" * 64)
    summary = pd.DataFrame(results)[[
        "condition",
        "T_eval",
        "rate_rec_1_0_1_pct", "rate_rec_0_1_0_pct", "rate_rec_total_pct",
        "UR_pct",
        "rate_dis_0_0_1_pct", "rate_dis_1_1_0_pct", "rate_dis_total_pct",
    ]].rename(columns={
        "condition"            : "Condition",
        "T_eval"               : "N",
        "rate_rec_1_0_1_pct"   : "Rec 1→0→1 %",
        "rate_rec_0_1_0_pct"   : "Rec 0→1→0 %",
        "rate_rec_total_pct"   : "Rec Total %",
        "UR_pct"               : "Unchanged %",
        "rate_dis_0_0_1_pct"   : "Dis 0→0→1 %",
        "rate_dis_1_1_0_pct"   : "Dis 1→1→0 %",
        "rate_dis_total_pct"   : "Dis Total %",
    })
    print(summary.to_string(index=False))
    print()