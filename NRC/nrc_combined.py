import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon
import re
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

GPTOSS_ANGER_AGE_CSV   = "anger_age_emotion_results_full_gpt_oss_1.csv"
GPTOSS_SADNESS_AGE_CSV = "sadness_age_emotion_results_full_gpt_oss_1.csv"
LLAMA4_ANGER_AGE_CSV   = "anger_age_emotion_results_full_llama4_1.csv"
LLAMA4_SADNESS_AGE_CSV = "sadness_age_emotion_results_full_llama4_1.csv"

NRC_PATH   = "NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
OUTPUT_DIR = "New_NRC_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PLOT STYLE ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize":   16,
    "axes.titlesize":   16,
    "axes.titleweight": "bold",
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
    "figure.dpi":       200,
    "savefig.dpi":      400,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# Four conditions — two models × two emotion conditions
PALETTE = {
    "gptoss_anger_age":   "#D6604D",   # red
    "gptoss_sadness_age": "#F4A582",   # light red/salmon
    "llama4_anger_age":   "#4393C3",   # blue
    "llama4_sadness_age": "#92C5DE",   # light blue
}

FOCUS_EMOTIONS = ["anger", "fear", "sadness"]
FOCUS_ALL      = FOCUS_EMOTIONS + ["positive", "negative"]

MIN_VIS = 0.0003

# ── NRC UTILITIES ─────────────────────────────────────────────────────────────
def load_nrc(path):
    lexicon = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if flag == "1":
                lexicon[word].add(emotion)
    return lexicon

def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[a-z]+\b", text.lower())

def emotion_vector(text, lexicon):
    tokens = tokenize(text)
    if not tokens:
        return {e: 0.0 for e in FOCUS_ALL}
    counts = defaultdict(int)
    for token in tokens:
        for tag in lexicon.get(token, []):
            counts[tag] += 1
    total = len(tokens)
    return {e: counts[e] / total for e in FOCUS_ALL}

# ── LOAD & COMPUTE ────────────────────────────────────────────────────────────
lexicon = load_nrc(NRC_PATH)

def load_deltas(csv_path, base_col, cond_col):
    df        = pd.read_csv(csv_path)
    base_vecs = pd.DataFrame([emotion_vector(t, lexicon) for t in df[base_col]])
    cond_vecs = pd.DataFrame([emotion_vector(t, lexicon) for t in df[cond_col]])
    delta     = cond_vecs - base_vecs
    return base_vecs, cond_vecs, delta

def run_wilcoxon(base_vecs, cond_vecs):
    results = {}
    for e in FOCUS_ALL:
        b = base_vecs[e].values
        c = cond_vecs[e].values
        if np.allclose(b, c):
            results[e] = {"p": 1.0, "sig": "n.s."}
            continue
        _, p = wilcoxon(c, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        results[e] = {"p": p, "sig": sig}
    return results

print("Loading GPT-oss Anger+Age …")
base_gptoss_aa, cond_gptoss_aa, delta_gptoss_aa = load_deltas(
    GPTOSS_ANGER_AGE_CSV, "Justification", "AngerAge_Justification")
stats_gptoss_aa = run_wilcoxon(base_gptoss_aa, cond_gptoss_aa)

print("Loading GPT-oss Sadness+Age …")
base_gptoss_sa, cond_gptoss_sa, delta_gptoss_sa = load_deltas(
    GPTOSS_SADNESS_AGE_CSV, "Justification", "SadnessAge_Justification")
stats_gptoss_sa = run_wilcoxon(base_gptoss_sa, cond_gptoss_sa)

print("Loading Llama 4 Anger+Age …")
base_llama4_aa, cond_llama4_aa, delta_llama4_aa = load_deltas(
    LLAMA4_ANGER_AGE_CSV, "Justification", "AngerAge_Justification")
stats_llama4_aa = run_wilcoxon(base_llama4_aa, cond_llama4_aa)

print("Loading Llama 4 Sadness+Age …")
base_llama4_sa, cond_llama4_sa, delta_llama4_sa = load_deltas(
    LLAMA4_SADNESS_AGE_CSV, "Justification", "SadnessAge_Justification")
stats_llama4_sa = run_wilcoxon(base_llama4_sa, cond_llama4_sa)

# ── BUILD PLOT DATA ───────────────────────────────────────────────────────────
# Four bars per emotion category, grouped with a small gap between model pairs
conditions = [
    ("gptoss_anger_age",   delta_gptoss_aa, stats_gptoss_aa),
    ("gptoss_sadness_age", delta_gptoss_sa, stats_gptoss_sa),
    ("llama4_anger_age",   delta_llama4_aa, stats_llama4_aa),
    ("llama4_sadness_age", delta_llama4_sa, stats_llama4_sa),
]

n_cats  = len(FOCUS_ALL)
n_bars  = len(conditions)
width   = 0.18          # individual bar width
# Offsets: two tight pairs with a small gap between models
pair_gap = 0.06
offsets = np.array([
    -1.5 * width - pair_gap / 2,   # GPT-oss Anger+Age
    -0.5 * width - pair_gap / 2,   # GPT-oss Sadness+Age
     0.5 * width + pair_gap / 2,   # Llama 4 Anger+Age
     1.5 * width + pair_gap / 2,   # Llama 4 Sadness+Age
])
x = np.arange(n_cats)

def to_plot_height(d):
    if abs(d) < MIN_VIS:
        return MIN_VIS if d >= 0 else -MIN_VIS
    return d

# ── DRAW ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

all_bars = []
for i, (key, delta_df, stats) in enumerate(conditions):
    raw    = [delta_df[e].mean() for e in FOCUS_ALL]
    plot_h = [to_plot_height(d) for d in raw]
    bars   = ax.bar(x + offsets[i], plot_h, width,
                    color=PALETTE[key], alpha=0.88,
                    edgecolor="white", linewidth=0.6,
                    zorder=3)
    all_bars.append((bars, raw, stats, key))

# ── ANNOTATE ──────────────────────────────────────────────────────────────────
def annotate_bar(ax, bar, raw_d, sig, p, color):
    plot_h = bar.get_height()
    x_ctr  = bar.get_x() + bar.get_width() / 2
    ypos   = plot_h + 0.00012 if plot_h >= 0 else plot_h - 0.00012
    va     = "bottom" if plot_h >= 0 else "top"

    if p is None or sig == "n.s.":
        ax.text(x_ctr, ypos, "n.s.", ha="center", va=va,
                fontsize=7.5, color="gray", zorder=5)
    else:
        p_str = "p<.001" if p < 0.001 else f"p={p:.2f}"
        label = f"{sig}\n{p_str}"
        fc    = "#FFF3F0" if "anger" in color else "#F0F4FF"
        ax.text(x_ctr, ypos, label, ha="center", va=va,
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc=fc,
                          ec=PALETTE[color], alpha=0.92, lw=0.7),
                zorder=5)

for bars, raw, stats, key in all_bars:
    for bar, raw_d, e in zip(bars, raw, FOCUS_ALL):
        sig = stats.get(e, {}).get("sig", "n.s.")
        p   = stats.get(e, {}).get("p", None)
        annotate_bar(ax, bar, raw_d, sig, p, key)

# ── AXES & LABELS ─────────────────────────────────────────────────────────────
ax.axhline(0, color="black", lw=0.9, ls="--", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([e.capitalize() for e in FOCUS_ALL], fontsize=15)
ax.set_ylabel("Mean Δ Token Density (Condition−Baseline)", fontsize=15)

# Thin vertical separators between emotion groups
for xi in x[:-1]:
    ax.axvline(xi + 0.5, color="lightgray", lw=0.8, ls=":", zorder=1)

y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo - abs(y_lo) * 0.12,
            y_hi + abs(y_hi) * 0.72)

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=PALETTE["gptoss_anger_age"],   alpha=0.88, label="GPT-oss  |  Anger + Age"),
    mpatches.Patch(color=PALETTE["gptoss_sadness_age"], alpha=0.88, label="GPT-oss  |  Sadness + Age"),
    mpatches.Patch(color=PALETTE["llama4_anger_age"],   alpha=0.88, label="Llama 4  |  Anger + Age"),
    mpatches.Patch(color=PALETTE["llama4_sadness_age"], alpha=0.88, label="Llama 4  |  Sadness + Age"),
    mpatches.Patch(color="none", label="n.s. = not significant (p ≥ 0.05)"),
]
ax.legend(handles=legend_handles, framealpha=0.9, edgecolor="lightgray",
          loc="upper right", ncol=1, fontsize=15)

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "combined_delta_bar.png")
plt.savefig(out_path)
plt.close()
print(f"\n✓ Saved → {out_path}")