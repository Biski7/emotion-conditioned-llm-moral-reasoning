# No filtering of unparseable rows 
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
# INPUT = "4_results/anger_emotion_results_full_gpt_oss.csv"
INPUT = "4_results/anger_emotion_results_full_llama4_3.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_int(val) -> int:
    try:
        return int(float(str(val).strip()))
    except:
        return -1

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, encoding="utf-8")
df.columns = [c.strip() for c in df.columns]

labels = df["Label"].apply(to_int)
base0  = df["Base0-shot"].apply(to_int)
anger  = df["Anger_Label"].apply(to_int)

# Report unparseable labels but do NOT drop them
# where total = len(result_df) and unparseable labels count as wrong answers
n_invalid_anger = (anger == -1).sum()
n_invalid_base0 = (base0 == -1).sum()
if n_invalid_anger > 0:
    print(f"Note: {n_invalid_anger} rows have unparseable Anger_Label  — counted as wrong (matches original).")
if n_invalid_base0 > 0:
    print(f"Note: {n_invalid_base0} rows have unparseable Base0-shot   — counted as wrong (matches original).")
if n_invalid_anger > 0 or n_invalid_base0 > 0:
    print()

total = len(labels)

# ── Accuracy ──────────────────────────────────────────────────────────────────
orig_correct  = labels == base0
anger_correct = labels == anger

orig_acc  = orig_correct.sum() / total * 100
anger_acc = anger_correct.sum() / total * 100
delta     = anger_acc - orig_acc

# ── Transition matrix ─────────────────────────────────────────────────────────
cc = ( orig_correct &  anger_correct).sum()   # correct → correct
cw = ( orig_correct & ~anger_correct).sum()   # correct → wrong  (anger broke it)
wc = (~orig_correct &  anger_correct).sum()   # wrong   → correct (anger helped)
ww = (~orig_correct & ~anger_correct).sum()   # wrong   → wrong   (no change)

# ── Directional flips ─────────────────────────────────────────────────────────
mask_1_0_1 = (labels == 1) & (base0 == 0) & (anger == 1)  # core narrative
mask_0_1_0 = (labels == 0) & (base0 == 1) & (anger == 0)  # excluded from narrative

# ── Per-class accuracy ────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"PER-CLASS ACCURACY")
print(f"{'─'*60}")
for cls, name in [(0, "Morally acceptable"), (1, "Morally wrong")]:
    mask = labels == cls
    n    = mask.sum()
    o    = (orig_correct  & mask).sum()
    a    = (anger_correct & mask).sum()
    print(f"Class {cls} ({name})")
    print(f"  n={n} | Baseline: {o/n*100:.2f}% | Anger: {a/n*100:.2f}% | Δ {(a-o)/n*100:+.2f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"BASELINE vs ANGER ACCURACY REPORT")
print(f"{'='*60}")
print(f"Input file                 : {INPUT}")
print(f"Total rows evaluated       : {total}")
print(f"{'─'*60}")
print(f"Baseline (Base0-shot) acc  : {orig_acc:.2f}%  ({orig_correct.sum()}/{total})")
print(f"Anger-injected accuracy    : {anger_acc:.2f}%  ({anger_correct.sum()}/{total})")
print(f"Delta (Anger − Baseline)   : {delta:+.2f}%")
print(f"{'─'*60}")
print(f"Correct → Correct (stable) : {cc}  ({cc/total*100:.1f}%)")
print(f"Correct → Wrong   (broke)  : {cw}  ({cw/total*100:.1f}%)")
print(f"Wrong   → Correct (helped) : {wc}  ({wc/total*100:.1f}%)")
print(f"Wrong   → Wrong   (stuck)  : {ww}  ({ww/total*100:.1f}%)")
print(f"{'─'*60}")
print(f"1→0→1 corrections          : {mask_1_0_1.sum()}  (morally-wrong recovered by anger)")
print(f"0→1→0 flips                : {mask_0_1_0.sum()}  (morally-ok corrupted — excluded)")
print(f"{'─'*60}")
recovery_rate = mask_1_0_1.sum() / ((labels == 1) & (base0 == 0)).sum() * 100
print(f"Recovery rate (1→0→1 / all GT=1 misclassified): {recovery_rate:.2f}%")
print(f"{'='*60}")