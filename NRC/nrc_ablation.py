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

# model = 'gptoss'
model = 'llama4'

if model == 'gptoss':
    ANGER_AGE_CSV    = "anger_age_emotion_results_full_gpt_oss_1.csv"
    SADNESS_AGE_CSV  = "sadness_age_emotion_results_full_gpt_oss_1.csv"
elif model == 'llama4':
    ANGER_AGE_CSV    = "anger_age_emotion_results_full_llama4_1.csv"
    SADNESS_AGE_CSV  = "sadness_age_emotion_results_full_llama4_1.csv"

NRC_PATH   = "NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
OUTPUT_DIR = "New_NRC_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PLOT STYLE ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize":   20,        # FIX 3: larger axis labels
    "axes.titlesize":   20,
    "axes.titleweight": "bold",
    "xtick.labelsize":  19,        # FIX 3: larger x-tick labels
    "ytick.labelsize":  19,        # FIX 3: larger y-tick labels
    "legend.fontsize":  18,        # FIX 3: larger legend
    "figure.dpi":       200,
    "savefig.dpi":      400,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

PALETTE = {
    "baseline":   "#2166AC",
    "anger_age":  "#D6604D",
    "sadness_age":"#7B3F9E",
    "pos_up":     "#92C5DE",
    "pos_down":   "#2166AC",
    "neg_up_aa":  "#D6604D",
    "neg_down_aa":"#F4A582",
    "neg_up_sa":  "#7B3F9E",
    "neg_down_sa":"#C9A8E0",
}

FOCUS_EMOTIONS = ["anger", "fear", "sadness"]
FOCUS_ALL      = FOCUS_EMOTIONS + ["positive", "negative"]
MIN_VIS        = 0.0003

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

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
lexicon = load_nrc(NRC_PATH)

df_aa = pd.read_csv(ANGER_AGE_CSV)
df_sa = pd.read_csv(SADNESS_AGE_CSV)
print(f"Anger+Age rows:   {len(df_aa)}")
print(f"Sadness+Age rows: {len(df_sa)}")

# ── COMPUTE EMOTION VECTORS ───────────────────────────────────────────────────
def compute_vecs(df, base_col, cond_col):
    base_vecs = pd.DataFrame([emotion_vector(t, lexicon) for t in df[base_col]])
    cond_vecs = pd.DataFrame([emotion_vector(t, lexicon) for t in df[cond_col]])
    delta     = cond_vecs - base_vecs
    return base_vecs, cond_vecs, delta

base_vecs_aa, aa_vecs, delta_aa = compute_vecs(
    df_aa, "Justification", "AngerAge_Justification")

base_vecs_sa, sa_vecs, delta_sa = compute_vecs(
    df_sa, "Justification", "SadnessAge_Justification")

# ── WILCOXON TESTS ────────────────────────────────────────────────────────────
def run_wilcoxon(base_vecs, cond_vecs, emotions):
    results = {}
    print(f"\n{'Emotion':<12} {'Base Mean':>10} {'Cond Mean':>10} {'Delta':>10} {'p-value':>10} {'Sig':>5}")
    print("─" * 60)
    for e in emotions:
        b = base_vecs[e].values
        c = cond_vecs[e].values
        if np.allclose(b, c):
            results[e] = {"p": 1.0, "sig": "n.s."}
            continue
        _, p = wilcoxon(c, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        results[e] = {"p": p, "sig": sig}
        print(f"{e:<12} {b.mean():>10.4f} {c.mean():>10.4f} {(c-b).mean():>+10.4f} {p:>10.4f} {sig:>5}")
    return results

print("\n── Anger+Age vs Baseline ───────────────────────────────────────")
stats_aa = run_wilcoxon(base_vecs_aa, aa_vecs, FOCUS_ALL)

print("\n── Sadness+Age vs Baseline ─────────────────────────────────────")
stats_sa = run_wilcoxon(base_vecs_sa, sa_vecs, FOCUS_ALL)

# ── P-VALUE LABEL HELPER ──────────────────────────────────────────────────────
# FIX 2: use p< thresholds instead of exact values
def fmt_p(p):
    if p < 0.001:
        return "p<0.001"
    elif p < 0.01:
        return "p<0.01"
    elif p < 0.05:
        return "p<0.05"
    else:
        return f"p={p:.3f}"   # only show exact value for n.s. cases (unused in sig path)

# ── FIGURE 1: GROUPED DELTA BAR ───────────────────────────────────────────────
n_cats = len(FOCUS_ALL)
x      = np.arange(n_cats)
width  = 0.32

deltas_aa = [delta_aa[e].mean() for e in FOCUS_ALL]
deltas_sa = [delta_sa[e].mean() for e in FOCUS_ALL]
sigs_aa   = [stats_aa.get(e, {}).get("sig", "") for e in FOCUS_ALL]
sigs_sa   = [stats_sa.get(e, {}).get("sig", "") for e in FOCUS_ALL]
pvals_aa  = [stats_aa.get(e, {}).get("p", None) for e in FOCUS_ALL]
pvals_sa  = [stats_sa.get(e, {}).get("p", None) for e in FOCUS_ALL]

def to_plot_height(d):
    if abs(d) < MIN_VIS:
        return MIN_VIS if d >= 0 else -MIN_VIS
    return d

deltas_aa_plot = [to_plot_height(d) for d in deltas_aa]
deltas_sa_plot = [to_plot_height(d) for d in deltas_sa]

fig, ax = plt.subplots(figsize=(13, 7))   # FIX 3: wider figure for bigger fonts

# FIX 1: labels → "Teenager + Anger" / "Teenager + Sadness"
bars_aa = ax.bar(x - width / 2, deltas_aa_plot, width,
                 color=PALETTE["anger_age"], alpha=0.85,
                 edgecolor="white", linewidth=0.8,
                 label="Teenager + Anger", zorder=3)
bars_sa = ax.bar(x + width / 2, deltas_sa_plot, width,
                 color=PALETTE["sadness_age"], alpha=0.85,
                 edgecolor="white", linewidth=0.8,
                 label="Teenager + Sadness", zorder=3)

def annotate_bars(bars, raw_deltas, sigs, pvals, color):
    for bar, raw_d, sig, p in zip(bars, raw_deltas, sigs, pvals):
        plot_h = bar.get_height()
        x_ctr  = bar.get_x() + bar.get_width() / 2
        ypos   = plot_h + 0.00018 if plot_h >= 0 else plot_h - 0.00018
        va     = "bottom" if plot_h >= 0 else "top"

        if p is None or sig == "n.s.":
            ax.text(x_ctr, ypos, "n.s.", ha="center", va=va,
                    fontsize=13, color="gray", zorder=5)
        else:
            # FIX 2: p< thresholds in label
            label = f"{sig}\n{fmt_p(p)}"
            fc    = "#FFF3F0" if color == PALETTE["anger_age"] else "#F5F0FF"
            ax.text(x_ctr, ypos, label, ha="center", va=va,
                    fontsize=13, fontweight="bold",    # FIX 3: bigger annotation font
                    bbox=dict(boxstyle="round,pad=0.25", fc=fc,
                              ec=color, alpha=0.92, lw=0.8),
                    zorder=5)

annotate_bars(bars_aa, deltas_aa, sigs_aa, pvals_aa, PALETTE["anger_age"])
annotate_bars(bars_sa, deltas_sa, sigs_sa, pvals_sa, PALETTE["sadness_age"])

ax.axhline(0, color="black", lw=0.9, ls="--", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([e.capitalize() for e in FOCUS_ALL])
ax.set_ylabel("Token Density(Condition − Baseline)")

y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo - abs(y_lo) * 0.15,
            y_hi + abs(y_hi) * 0.70)

# FIX 1: legend labels updated; FIX 3: legend fontsize inherited from rcParams
anger_patch   = mpatches.Patch(color=PALETTE["anger_age"],   alpha=0.85, label="Teenager + Anger")
sadness_patch = mpatches.Patch(color=PALETTE["sadness_age"], alpha=0.85, label="Teenager + Sadness")
sig_note      = mpatches.Patch(color="none", label="* p<.05  ** p<.01  *** p<.001")
sig_note1      = mpatches.Patch(color="none", label="n.s. = not significant")

ax.legend(handles=[anger_patch, sadness_patch, sig_note, sig_note1],
          framealpha=0.9, edgecolor="lightgray", loc="upper right")

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, f"{model}_delta_bar_grouped.pdf")
plt.savefig(fig1_path)
plt.savefig(fig1_path.replace(".pdf", ".png"))
plt.close()
print(f"\nSaved → {fig1_path}")

# ── FIGURE 2: PER-SCENARIO POSITIVE/NEGATIVE SHIFT ───────────────────────────
def make_sentiment_panel(ax, delta_df, stats, color_neg_up, color_neg_down, title, ylabel):
    d_neg = delta_df["negative"].values
    d_pos = delta_df["positive"].values

    order   = np.argsort(-d_neg)
    d_neg_s = d_neg[order]
    d_pos_s = d_pos[order]
    x_idx   = np.arange(len(d_neg_s))
    bw      = 0.42

    neg_colors = [color_neg_up   if v >= 0 else color_neg_down for v in d_neg_s]
    pos_colors = [PALETTE["pos_down"] if v <= 0 else PALETTE["pos_up"] for v in d_pos_s]

    ax.bar(x_idx - bw / 2, d_neg_s, bw, color=neg_colors, zorder=3)
    ax.bar(x_idx + bw / 2, d_pos_s, bw, color=pos_colors, zorder=3)
    ax.axhline(0, color="black", lw=0.9, ls="--", zorder=2)

    ax.axhline(d_neg_s.mean(), color=color_neg_up,       lw=1.5, ls=":")
    ax.axhline(d_pos_s.mean(), color=PALETTE["baseline"], lw=1.5, ls=":")

    p_neg = stats.get("negative", {}).get("p", 1.0)
    s_neg = stats.get("negative", {}).get("sig", "n.s.")
    p_pos = stats.get("positive", {}).get("p", 1.0)
    s_pos = stats.get("positive", {}).get("sig", "n.s.")

    # FIX 2: use p< thresholds in Wilcoxon annotation box too
    ax.text(0.02, 0.97,
            f"Wilcoxon signed-rank\n"
            f"Δ Negative: {fmt_p(p_neg)}  {s_neg}\n"
            f"Δ Positive:   {fmt_p(p_pos)}  {s_pos}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=16, linespacing=1.6,            # FIX 3: bigger annotation text
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="gray", alpha=0.93, lw=0.9),
            zorder=6)

    tick_step = max(1, len(x_idx) // 10)
    ax.set_xticks(x_idx[::tick_step])
    ax.set_xticklabels(x_idx[::tick_step] + 1)
    ax.set_xlabel("Scenario index (sorted by Δ negative, descending)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    neg_up   = mpatches.Patch(color=color_neg_up,         label=f"Neg ↑ ({np.sum(d_neg > 0)} scenarios)")
    neg_down = mpatches.Patch(color=color_neg_down,        label=f"Neg ↓ ({np.sum(d_neg < 0)} scenarios)")
    pos_down = mpatches.Patch(color=PALETTE["pos_down"],   label=f"Pos ↓ ({np.sum(d_pos < 0)} scenarios)")
    pos_up   = mpatches.Patch(color=PALETTE["pos_up"],     label=f"Pos ↑ ({np.sum(d_pos > 0)} scenarios)")

    ax.legend(handles=[neg_up, neg_down, pos_down, pos_up],
              ncol=2, framealpha=0.9, loc="lower right", edgecolor="lightgray")

fig, axes = plt.subplots(2, 1, figsize=(18, 14))   # FIX 3: larger figure

make_sentiment_panel(
    axes[0], delta_aa, stats_aa,
    color_neg_up=PALETTE["neg_up_aa"],
    color_neg_down=PALETTE["neg_down_aa"],
    title="Per-Scenario Sentiment Shift — Teenager + Anger vs. Baseline",   # FIX 1
    ylabel="Δ Token Density (Teenager+Anger − Baseline)"
)
make_sentiment_panel(
    axes[1], delta_sa, stats_sa,
    color_neg_up=PALETTE["neg_up_sa"],
    color_neg_down=PALETTE["neg_down_sa"],
    title="Per-Scenario Sentiment Shift — Teenager + Sadness vs. Baseline",  # FIX 1
    ylabel="Δ Token Density (Teenager+Sadness − Baseline)"
)

plt.tight_layout(h_pad=4)
fig2_path = os.path.join(OUTPUT_DIR, f"{model}_sentiment_per_scenario.pdf")
plt.savefig(fig2_path)
plt.savefig(fig2_path.replace(".pdf", ".png"))
plt.close()
print(f"Saved → {fig2_path}")

# ── SAVE SCORES ───────────────────────────────────────────────────────────────
out_aa = pd.concat([
    df_aa[["Justification", "AngerAge_Justification"]].reset_index(drop=True),
    base_vecs_aa[FOCUS_ALL].add_prefix("base_"),
    aa_vecs[FOCUS_ALL].add_prefix("aa_"),
    delta_aa[FOCUS_ALL].add_prefix("delta_aa_")
], axis=1)
out_aa.to_csv(os.path.join(OUTPUT_DIR, f"{model}_lexical_scores_anger_age.csv"), index=False)

out_sa = pd.concat([
    df_sa[["Justification", "SadnessAge_Justification"]].reset_index(drop=True),
    base_vecs_sa[FOCUS_ALL].add_prefix("base_"),
    sa_vecs[FOCUS_ALL].add_prefix("sa_"),
    delta_sa[FOCUS_ALL].add_prefix("delta_sa_")
], axis=1)
out_sa.to_csv(os.path.join(OUTPUT_DIR, f"{model}_lexical_scores_sadness_age.csv"), index=False)

print(f"\n✓ All outputs saved to {OUTPUT_DIR}")