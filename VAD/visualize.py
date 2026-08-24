import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker
import matplotlib.colors as mcolors


# ── Config ──────────────────────────────────────────────────────────────

MODEL_DIRS = {
    "GPT-oss": "results_gptoss",
    "Llama 4": "results_llama4",
}
MODEL_ORDER = ["GPT-oss", "Llama 4"]

OUT_DIR = "figures_all"

DELTA_FILES = {
    ("anger", "reader"): "delta_anger_reader.csv",
    ("anger", "writer"): "delta_anger_writer.csv",
    ("sadness", "reader"): "delta_sadness_reader.csv",
    ("sadness", "writer"): "delta_sadness_writer.csv",
}
DELTA_COLS = {"V": "delta_valence", "A": "delta_arousal", "D": "delta_dominance"}

VAD_FILE_PATTERN = {
    "neutral": "vad_neutral_{persp}.csv",
    "anger": "vad_anger_{persp}.csv",
    "sadness": "vad_sadness_{persp}.csv",
}
VAD_COLS = {"V": "valence", "A": "arousal", "D": "dominance"}

DIMS = ["V", "A", "D"]
DIM_NAMES = {"V": "Valence", "A": "Arousal", "D": "Dominance"}
CONDITIONS = ["anger", "sadness"]
PERSPECTIVES = ["reader", "writer"]


# ── Style ───────────────────────────────────────────────────────────────

FONT_FAMILY = "serif"
FONT_SIZE = 9
LABEL_SIZE = 9
TICK_SIZE = 7.5
LEGEND_SIZE = 8
DPI = 300
FIG_WIDTH = 7.2

PERSP_COLORS = {"reader": "#4477AA", "writer": "#AA3377"}
MODEL_HATCH = {"GPT-oss": "", "Llama 4": "//"}
NEUTRAL_MARKER_COLOR = "#555555"

matplotlib.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "patch.linewidth": 0.6,
})

# Bar layout shared by fig1 and fig4 (identical panel geometry so the
# two figures read as a matched pair in the paper).
BAR_WIDTH = 0.34
WITHIN_MODEL_GAP = 0.02
BETWEEN_MODEL_GAP = 0.30

_X0 = 0.0
_X1 = _X0 + BAR_WIDTH + WITHIN_MODEL_GAP
_X2 = _X1 + BAR_WIDTH + BETWEEN_MODEL_GAP
_X3 = _X2 + BAR_WIDTH + WITHIN_MODEL_GAP
BAR_POSITIONS = [_X0, _X1, _X2, _X3]
BAR_SPECS = [
    ("GPT-oss", "reader"),
    ("GPT-oss", "writer"),
    ("Llama 4", "reader"),
    ("Llama 4", "writer"),
]


# ── Loaders ─────────────────────────────────────────────────────────────

def load_deltas_for_model(results_dir: str) -> dict:
    """Returns {(condition, perspective): DataFrame[V, A, D]}."""
    data = {}

    for key, fname in DELTA_FILES.items():
        path = os.path.join(results_dir, fname)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        df = pd.read_csv(path)
        missing = [c for c in DELTA_COLS.values() if c not in df.columns]
        if missing:
            raise ValueError(f"{fname}: missing columns {missing}")

        renamed = (
            df[list(DELTA_COLS.values())]
            .rename(columns={v: k for k, v in DELTA_COLS.items()})
            .dropna()
            .reset_index(drop=True)
        )
        data[key] = renamed
        print(f"  delta  {results_dir}/{fname}: {len(renamed)} rows")

    return data


def load_vad_for_model(results_dir: str) -> dict:
    """Returns {(condition, perspective): DataFrame[V, A, D]}."""
    data = {}

    for cond, pattern in VAD_FILE_PATTERN.items():
        for persp in PERSPECTIVES:
            fname = pattern.format(persp=persp)
            path = os.path.join(results_dir, fname)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")

            df = pd.read_csv(path)
            missing = [c for c in VAD_COLS.values() if c not in df.columns]
            if missing:
                raise ValueError(f"{fname}: missing columns {missing}")

            renamed = (
                df[list(VAD_COLS.values())]
                .rename(columns={v: k for k, v in VAD_COLS.items()})
                .dropna()
                .reset_index(drop=True)
            )
            data[(cond, persp)] = renamed
            print(f"  vad    {results_dir}/{fname}: {len(renamed)} rows")

    return data


def load_all_models() -> dict:
    """Returns {model_name: {"delta": {...}, "vad": {...}}}."""
    all_data = {}

    for model_name, results_dir in MODEL_DIRS.items():
        print(f"Loading {model_name} from {results_dir}/ ...")
        all_data[model_name] = {
            "delta": load_deltas_for_model(results_dir),
            "vad": load_vad_for_model(results_dir),
        }

    return all_data


# ── Stats helpers ──────────────────────────────────────────────────────

def mean_se_ci(arr: np.ndarray):
    n = len(arr)
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n)
    ci = se * stats.t.ppf(0.975, df=n - 1)
    return m, ci, n


def wilcoxon_p(arr: np.ndarray) -> float:
    d = arr[arr != 0.0]
    if len(d) < 10:
        return float("nan")
    _, p = stats.wilcoxon(d, alternative="two-sided")
    return float(p)


def sig_label(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def summarise_deltas(all_data: dict) -> pd.DataFrame:
    """One row per model x condition x perspective x dimension."""
    rows = []

    for model_name, bundle in all_data.items():
        delta_data = bundle["delta"]

        for (cond, persp), df in delta_data.items():
            for dim in DIMS:
                arr = df[dim].values
                m, ci, n = mean_se_ci(arr)
                std = arr.std(ddof=1)
                p = wilcoxon_p(arr)

                rows.append({
                    "model": model_name,
                    "condition": cond,
                    "perspective": persp,
                    "dimension": DIM_NAMES[dim],
                    "n": n,
                    "mean": m,
                    "std": std,
                    "ci95": ci,
                    "p": p,
                    "sig": sig_label(p),
                })

    return pd.DataFrame(rows)


def _save(fig, out_dir: str, fname_stem: str):
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{fname_stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig 1 — Merged ΔVAD bar chart (both models)
# ─────────────────────────────────────────────────────────────────────

def fig1_delta_bars_merged(summary: pd.DataFrame, out_dir: str):
    """
    3 rows (Valence / Arousal / Dominance) x 2 cols (Anger / Sadness).
    Each panel: 4 bars -- GPT-oss/Reader, GPT-oss/Writer,
                          Llama4/Reader, Llama4/Writer.
    Bars are ΔVAD relative to neutral, so the reference is a plain
    zero line (no per-bar neutral marker needed here, unlike Fig. 4,
    since the delta itself already encodes that comparison).
    """

    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, 6.9), sharex=True, sharey=False)

    for row, dim in enumerate(DIMS):
        dim_name = DIM_NAMES[dim]

        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]

            panel_lo, panel_hi = np.inf, -np.inf

            for xpos, (model_name, persp) in zip(BAR_POSITIONS, BAR_SPECS):
                r = summary[
                    (summary.model == model_name)
                    & (summary.condition == cond)
                    & (summary.perspective == persp)
                    & (summary.dimension == dim_name)
                ].iloc[0]

                m, ci, sig = r["mean"], r["ci95"], r["sig"]
                color = PERSP_COLORS[persp]
                hatch = MODEL_HATCH[model_name]

                ax.bar(
                    xpos, m, BAR_WIDTH,
                    color=color, alpha=0.85, hatch=hatch,
                    edgecolor="white", linewidth=0.5, zorder=2,
                )
                ax.errorbar(
                    xpos, m, yerr=ci,
                    fmt="none", color="#333333",
                    capsize=2.5, linewidth=0.8, zorder=4,
                )

                va = "bottom" if m >= 0 else "top"
                label_y = (m + ci + 0.05 * max(abs(m), 0.02)) if m >= 0 else (m - ci - 0.05 * max(abs(m), 0.02))
                label_text = f"{m:+.3f}"
                if sig:
                    label_text += f"\n{sig}"

                ax.text(
                    xpos, label_y, label_text,
                    ha="center", va=va,
                    fontsize=6.3, fontweight="bold",
                    color="#222222", linespacing=1.05, zorder=5,
                )

                panel_lo = min(panel_lo, m - ci, label_y)
                panel_hi = max(panel_hi, m + ci, label_y)

            ax.axhline(0, color="black", linewidth=0.8, zorder=1)

            pad = max((panel_hi - panel_lo) * 0.22, 0.01)
            ax.set_ylim(panel_lo - pad, panel_hi + pad)

            mid_gap_x = (_X1 + _X2) / 2
            ax.axvline(mid_gap_x, color="#dddddd", linewidth=0.7, zorder=0)

            ax.set_xlim(_X0 - BAR_WIDTH, _X3 + BAR_WIDTH)
            ax.set_xticks([(_X0 + _X1) / 2, (_X2 + _X3) / 2])
            ax.set_xticklabels(["GPT-oss", "Llama 4"], fontsize=TICK_SIZE + 1)

            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            ax.tick_params(axis="y", labelsize=TICK_SIZE)

            if col == 0:
                ax.set_ylabel(
                    dim_name, fontsize=LABEL_SIZE + 2,
                    fontweight="bold", labelpad=6,
                )

            if row == 0:
                ax.text(
                    0.5, 1.10, cond.capitalize(),
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#333333",
                )

    legend_handles = [
        mpatches.Patch(facecolor=PERSP_COLORS["reader"], edgecolor="white", label="Reader"),
        mpatches.Patch(facecolor=PERSP_COLORS["writer"], edgecolor="white", label="Writer"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#333333", hatch="", label="GPT-oss"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#333333", hatch="//", label="Llama 4"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="* p<.05  ** p<.01  *** p<.001"),
    ]
    fig.suptitle("Mean ΔVAD relative to neutral", y=1.06, fontsize=12, fontweight="bold")
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
        ncol=3, frameon=True, framealpha=0.95, fontsize=LEGEND_SIZE,
        edgecolor="#dddddd", columnspacing=1.3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    _save(fig, out_dir, "fig1_delta_bars_merged")


# ─────────────────────────────────────────────────────────────────────
# Fig 3 — Side-by-side heatmaps (both models, shared color scale)
# ─────────────────────────────────────────────────────────────────────

def fig3_heatmap_merged(summary: pd.DataFrame, out_dir: str):
    row_keys = [
        ("anger", "reader"), ("anger", "writer"),
        ("sadness", "reader"), ("sadness", "writer"),
    ]
    row_labels = ["Anger / Reader", "Anger / Writer", "Sadness / Reader", "Sadness / Writer"]
    col_labels = [DIM_NAMES[d] for d in DIMS]

    mats, pmats = {}, {}
    global_max = 0.0

    for model_name in MODEL_ORDER:
        mat = np.zeros((4, 3))
        pmat = np.zeros((4, 3))

        for ri, (cond, persp) in enumerate(row_keys):
            for ci, dim_name in enumerate(col_labels):
                r = summary[
                    (summary.model == model_name)
                    & (summary.condition == cond)
                    & (summary.perspective == persp)
                    & (summary.dimension == dim_name)
                ].iloc[0]
                mat[ri, ci] = r["mean"]
                pmat[ri, ci] = r["p"]

        mats[model_name] = mat
        pmats[model_name] = pmat
        global_max = max(global_max, np.abs(mat).max())

    vmax = global_max * 1.15
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    im = None
    for ax, model_name in zip(axes, MODEL_ORDER):
        mat = mats[model_name]
        pmat = pmats[model_name]

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(3))
        ax.set_xticklabels(col_labels, fontsize=TICK_SIZE, rotation=0, ha="center")
        ax.tick_params(axis="x", pad=5)

        ax.set_yticks(range(4))
        if model_name == MODEL_ORDER[0]:
            ax.set_yticklabels(row_labels, fontsize=TICK_SIZE)
        else:
            ax.set_yticklabels([])

        for ri in range(4):
            for ci in range(3):
                v = mat[ri, ci]
                sig = sig_label(pmat[ri, ci])
                txt = f"{v:+.4f}\n{sig}" if sig else f"{v:+.4f}"
                ax.text(
                    ci, ri, txt, ha="center", va="center",
                    fontsize=6.5, color="#000000", linespacing=1.05,
                )

        ax.set_title(model_name, fontsize=10, fontweight="bold", pad=6)

    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cb.set_label("Mean ΔVAD", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.text(
        0.5, -0.06, "* p < .05, ** p < .01, *** p < .001  (color scale shared across both panels)",
        ha="center", va="top", fontsize=8, color="#333333",
    )

    _save(fig, out_dir, "fig3_heatmap_merged")


# ─────────────────────────────────────────────────────────────────────
# Fig 4 — Merged absolute VAD profiles (paired neutral markers)
# ─────────────────────────────────────────────────────────────────────

def fig4_vad_profiles_merged(all_data: dict, out_dir: str):
    """
    3 rows (Valence / Arousal / Dominance) x 2 cols (Anger / Sadness).
    Each panel: 4 bars -- GPT-oss/Reader, GPT-oss/Writer,
                          Llama4/Reader, Llama4/Writer.
    Each bar carries its OWN paired neutral marker (dotted line + CI
    whisker) rather than a single pooled reader+writer line, so the
    reference is fair for that exact model/perspective.
    """

    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, 6.9), sharex=True, sharey=False)

    for row, dim in enumerate(DIMS):
        dim_name = DIM_NAMES[dim]

        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]

            panel_lo, panel_hi = np.inf, -np.inf

            for xpos, (model_name, persp) in zip(BAR_POSITIONS, BAR_SPECS):
                vad_data = all_data[model_name]["vad"]

                cond_arr = vad_data[(cond, persp)][dim].values
                neutral_arr = vad_data[("neutral", persp)][dim].values

                m, ci, n = mean_se_ci(cond_arr)
                nm, nci, nn = mean_se_ci(neutral_arr)

                color = PERSP_COLORS[persp]
                hatch = MODEL_HATCH[model_name]

                ax.bar(
                    xpos, m, BAR_WIDTH,
                    color=color, alpha=0.85, hatch=hatch,
                    edgecolor="white", linewidth=0.5, zorder=2,
                )
                ax.errorbar(
                    xpos, m, yerr=ci,
                    fmt="none", color="#333333",
                    capsize=2.5, linewidth=0.8, zorder=4,
                )

                # per-bar paired neutral marker
                ax.plot(
                    [xpos - BAR_WIDTH / 2, xpos + BAR_WIDTH / 2], [nm, nm],
                    color=NEUTRAL_MARKER_COLOR, linewidth=1.1,
                    linestyle=(0, (2, 1.5)), zorder=3,
                )
                ax.errorbar(
                    xpos, nm, yerr=nci,
                    fmt="none", color=NEUTRAL_MARKER_COLOR,
                    alpha=0.6, capsize=1.6, linewidth=0.7, zorder=3,
                )

                # label cleared above the higher of (bar+CI) and (neutral+CI)
                label_y = max(m + ci, nm + nci) + 0.004 * max(abs(m), 1)
                ax.text(
                    xpos, label_y, f"{m:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                    color="#222222", zorder=5,
                )

                panel_lo = min(panel_lo, m - ci, nm - nci)
                panel_hi = max(panel_hi, m + ci, nm + nci, label_y)

            pad = max((panel_hi - panel_lo) * 0.35, 0.05)
            ax.set_ylim(panel_lo - pad, panel_hi + pad * 0.4)

            mid_gap_x = (_X1 + _X2) / 2
            ax.axvline(mid_gap_x, color="#dddddd", linewidth=0.7, zorder=0)

            ax.set_xlim(_X0 - BAR_WIDTH, _X3 + BAR_WIDTH)
            ax.set_xticks([(_X0 + _X1) / 2, (_X2 + _X3) / 2])
            ax.set_xticklabels(["GPT-oss", "Llama 4"], fontsize=TICK_SIZE + 1)

            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis="y", labelsize=TICK_SIZE)

            if col == 0:
                ax.set_ylabel(
                    dim_name, fontsize=LABEL_SIZE + 2,
                    fontweight="bold", labelpad=6,
                )

            if row == 0:
                ax.text(
                    0.5, 1.10, cond.capitalize(),
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#333333",
                )

    legend_handles = [
        mpatches.Patch(facecolor=PERSP_COLORS["reader"], edgecolor="white", label="Reader"),
        mpatches.Patch(facecolor=PERSP_COLORS["writer"], edgecolor="white", label="Writer"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#333333", hatch="", label="GPT-oss"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#333333", hatch="//", label="Llama 4"),
        mlines.Line2D(
            [0], [0], color=NEUTRAL_MARKER_COLOR, linewidth=1.1,
            linestyle=(0, (2, 1.5)), label="Neutral mean ± 95% CI",
        ),
    ]
    fig.suptitle("Absolute VAD levels by condition", y=1.06, fontsize=12, fontweight="bold")
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
        ncol=3, frameon=True, framealpha=0.95, fontsize=LEGEND_SIZE,
        edgecolor="#dddddd", columnspacing=1.3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    _save(fig, out_dir, "fig4_vad_profiles_merged")


# ─────────────────────────────────────────────────────────────────────
# Table 1 — Combined LaTeX + Markdown (both models, Model column)
# ─────────────────────────────────────────────────────────────────────

def table1_merged(summary: pd.DataFrame, out_dir: str):
    order = [
        ("anger", "reader"), ("sadness", "reader"),
        ("anger", "writer"), ("sadness", "writer"),
    ]
    dim_order = ["Valence", "Arousal", "Dominance"]

    def p_str_md(p):
        if np.isnan(p):
            return "---"
        if p < 0.001:
            return "< .001***"
        if p < 0.01:
            return f"{p:.3f}**"
        if p < 0.05:
            return f"{p:.3f}*"
        return f"{p:.3f}"

    md = [
        "| Model | Condition | Perspective | Dimension | Mean Δ | SD | 95 % CI | *p* |",
        "|-------|-----------|-------------|-----------|-------:|---:|--------:|-----|",
    ]

    for model_name in MODEL_ORDER:
        for cond, persp in order:
            sub = summary[
                (summary.model == model_name)
                & (summary.condition == cond)
                & (summary.perspective == persp)
            ]

            for dim in dim_order:
                r = sub[sub.dimension == dim].iloc[0]
                m, sd, ci, p = r["mean"], r["std"], r["ci95"], r["p"]

                md.append(
                    f"| {model_name} | {cond.capitalize()} | {persp.capitalize()} | "
                    f"{dim} | {m:+.4f} | {sd:.4f} | ±{ci:.4f} | {p_str_md(p)} |"
                )

    md_path = os.path.join(out_dir, "table1_delta_summary_merged.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved: {md_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data for both models ...")
    all_data = load_all_models()

    print("\nBuilding combined summary statistics ...")
    summary = summarise_deltas(all_data)

    print("\nGenerating figures ...")
    fig1_delta_bars_merged(summary, OUT_DIR)
    fig3_heatmap_merged(summary, OUT_DIR)
    fig4_vad_profiles_merged(all_data, OUT_DIR)

    print("\nGenerating table ...")
    table1_merged(summary, OUT_DIR)

    print(f"\n{'=' * 55}")
    print(f"All outputs saved to: {OUT_DIR}/")
    print(f"{'=' * 55}")

    print("\nSummary (mean ΔVAD):")
    print(
        summary[["model", "condition", "perspective", "dimension", "mean", "std", "ci95", "p", "sig"]]
        .to_string(index=False, float_format=lambda x: f"{x:+.4f}")
    )


if __name__ == "__main__":
    main()