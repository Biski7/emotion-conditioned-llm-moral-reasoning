"""
infer.py — Predict VAD and compute delta-VAD across affect conditions.

This script loads trained reader/writer VAD MLP models and applies them to
LLM-generated justifications under three conditions:

    1. neutral
    2. anger
    3. sadness

It then computes paired delta-VAD:

    anger_delta   = anger   - neutral
    sadness_delta = sadness - neutral

For each condition x perspective x VAD dimension, it reports:

    mean delta
    std delta
    median delta
    Wilcoxon signed-rank p-value
    FDR-corrected p-value
    effect size

Expected trained files:

    vad_models/
        vad_mlp_reader.pt
        x_scaler_reader.joblib
        y_scaler_reader.joblib

        vad_mlp_writer.pt
        x_scaler_writer.joblib
        y_scaler_writer.joblib

Example:
    python infer.py

Outputs:
    results/
        vad_predictions_long.csv
        vad_neutral_reader.csv
        vad_anger_reader.csv
        vad_sadness_reader.csv
        vad_neutral_writer.csv
        vad_anger_writer.csv
        vad_sadness_writer.csv

        delta_item_level.csv
        delta_anger_reader.csv
        delta_sadness_reader.csv
        delta_anger_writer.csv
        delta_sadness_writer.csv

        delta_summary.csv
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy import stats
from sentence_transformers import SentenceTransformer
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_DIR = "vad_models"
OUTPUT_DIR = "results"
CACHE_DIR = "vad_infer_cache"

USE_EMBEDDING_CACHE = True

# Shared ID column. Strongly recommended.
# This prevents accidentally comparing the wrong neutral/anger/sadness rows.
ID_COL = "ID"

# NEUTRAL_CSV = "gpt-oss_Commonsense_Base0-shot_Test_CS.csv"
# ANGER_CSV   = "anger_emotion_results_full_gpt_oss_1.csv"
# SADNESS_CSV = "sadness_emotion_results_full_gpt_oss_1.csv"


NEUTRAL_CSV = "scout_Commonsense_Base0-shot_Test_CS.csv"
ANGER_CSV   = "anger_emotion_results_full_llama4_1.csv"
SADNESS_CSV = "sadness_emotion_results_full_llama4_1.csv"

# CSV paths and the justification column name in each file
NEUTRAL_COL = "Justification"
# Justification, AngerAge_Justification, SadnessAge_Justification, Anger_Justification, SadnessAge_Justification
ANGER_COL   = "Anger_Justification"
SADNESS_COL = "SadnessAge_Justification"       

PERSPECTIVES = ["reader", "writer"]
CONDITIONS = ["neutral", "anger", "sadness"]
DELTA_CONDITIONS = ["anger", "sadness"]

DIMS = ["valence", "arousal", "dominance"]
DIM_LABELS = ["Valence", "Arousal", "Dominance"]


# ---------------------------------------------------------------------
# Model
# Must match train.py architecture
# ---------------------------------------------------------------------

class VADMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def ensure_dirs() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def check_file(path: str | Path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def text_hash(texts: List[str]) -> str:
    """
    Stable hash for a list of texts, used for embedding cache names.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8", errors="ignore"))
        h.update(b"\n---ROW---\n")
    return h.hexdigest()[:16]


def safe_wilcoxon(values: np.ndarray) -> float:
    """
    scipy.stats.wilcoxon fails if all values are zero.
    In that degenerate case, return p=1.0.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan

    if np.allclose(values, 0.0):
        return 1.0

    try:
        return float(stats.wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def cohens_d_paired(values: np.ndarray) -> float:
    """
    Paired Cohen's d for difference scores:
        d_z = mean(diff) / std(diff)

    This is a simple effect size for paired deltas.
    """
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return np.nan

    sd = values.std(ddof=1)

    if np.isclose(sd, 0.0):
        return 0.0

    return float(values.mean() / sd)


def median_over_sd(values: np.ndarray) -> float:
    """
    Robust-ish effect indicator:
        median(diff) / std(diff)
    """
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return np.nan

    sd = values.std(ddof=1)

    if np.isclose(sd, 0.0):
        return 0.0

    return float(np.median(values) / sd)


# ---------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------

def load_condition_data() -> pd.DataFrame:
    """
    Loads neutral, anger, and sadness CSVs and aligns them by ID_COL.

    Required output columns:
        id
        neutral_text
        anger_text
        sadness_text
    """
    check_file(NEUTRAL_CSV, "Neutral CSV")
    check_file(ANGER_CSV, "Anger CSV")
    check_file(SADNESS_CSV, "Sadness CSV")

    neutral_df = pd.read_csv(NEUTRAL_CSV)
    anger_df = pd.read_csv(ANGER_CSV)
    sadness_df = pd.read_csv(SADNESS_CSV)

    required = [
        (neutral_df, NEUTRAL_CSV, [ID_COL, NEUTRAL_COL]),
        (anger_df, ANGER_CSV, [ID_COL, ANGER_COL]),
        (sadness_df, SADNESS_CSV, [ID_COL, SADNESS_COL]),
    ]

    for df, path, cols in required:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

    neutral_df = neutral_df[[ID_COL, NEUTRAL_COL]].rename(
        columns={NEUTRAL_COL: "neutral_text"}
    )

    anger_df = anger_df[[ID_COL, ANGER_COL]].rename(
        columns={ANGER_COL: "anger_text"}
    )

    sadness_df = sadness_df[[ID_COL, SADNESS_COL]].rename(
        columns={SADNESS_COL: "sadness_text"}
    )

    before_counts = {
        "neutral": len(neutral_df),
        "anger": len(anger_df),
        "sadness": len(sadness_df),
    }

    df = (
        neutral_df
        .merge(anger_df, on=ID_COL, how="inner")
        .merge(sadness_df, on=ID_COL, how="inner")
        .reset_index(drop=True)
    )

    for col in ["neutral_text", "anger_text", "sadness_text"]:
        df[col] = df[col].fillna("").astype(str)

    after_count = len(df)

    print("Loaded condition files:")
    for k, v in before_counts.items():
        print(f"  {k:8s}: {v}")

    print(f"Aligned examples by '{ID_COL}': {after_count}")

    if after_count == 0:
        raise ValueError(
            "No aligned examples after merging by ID. "
            "Check that all three CSVs share the same ID values."
        )

    dropped = {k: v - after_count for k, v in before_counts.items()}
    if any(v > 0 for v in dropped.values()):
        print("Warning: some rows were dropped during ID alignment:")
        for k, v in dropped.items():
            print(f"  {k:8s}: dropped {v}")

    return df


# ---------------------------------------------------------------------
# Loading models
# ---------------------------------------------------------------------

def load_vad_model(
    perspective: str,
    device: torch.device,
) -> Tuple[VADMLP, object, object, str]:
    """
    Loads trained MLP and scalers for a given perspective.
    """
    if perspective not in PERSPECTIVES:
        raise ValueError(f"Unknown perspective: {perspective}")

    ckpt_path = Path(MODEL_DIR) / f"vad_mlp_{perspective}.pt"
    x_scaler_path = Path(MODEL_DIR) / f"x_scaler_{perspective}.joblib"
    y_scaler_path = Path(MODEL_DIR) / f"y_scaler_{perspective}.joblib"

    check_file(ckpt_path, f"{perspective} model checkpoint")
    check_file(x_scaler_path, f"{perspective} input scaler")
    check_file(y_scaler_path, f"{perspective} output scaler")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = VADMLP(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    embedding_model = ckpt.get("embedding_model")
    if embedding_model is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain 'embedding_model'. "
            "Please retrain with the updated train.py."
        )

    return model, x_scaler, y_scaler, embedding_model


# ---------------------------------------------------------------------
# Embedding and prediction
# ---------------------------------------------------------------------

def embedding_cache_path(
    embedding_model_name: str,
    condition: str,
    texts: List[str],
) -> Path:
    safe_model_name = embedding_model_name.replace("/", "__")
    h = text_hash(texts)
    return Path(CACHE_DIR) / f"{safe_model_name}_{condition}_{len(texts)}_{h}.npy"


def encode_condition_texts(
    encoder: SentenceTransformer,
    embedding_model_name: str,
    condition: str,
    texts: List[str],
) -> np.ndarray:
    """
    Encodes texts with sentence-transformer, optionally using cache.
    """
    path = embedding_cache_path(embedding_model_name, condition, texts)

    if USE_EMBEDDING_CACHE and path.exists():
        print(f"Loading cached embeddings for {condition}: {path}")
        return np.load(path)

    print(f"Encoding {condition}: {len(texts)} texts")
    embeddings = encoder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=16,
    )
    encoder.max_seq_length = 1024  # Ensure max sequence length is set

    if USE_EMBEDDING_CACHE:
        np.save(path, embeddings)
        print(f"Saved embeddings cache: {path}")

    return embeddings


@torch.no_grad()
def predict_vad_from_embeddings(
    embeddings: np.ndarray,
    model: VADMLP,
    x_scaler,
    y_scaler,
    device: torch.device,
) -> np.ndarray:
    """
    Returns VAD predictions in original 1–5 scale.

    Shape:
        (n_examples, 3)
    """
    X = x_scaler.transform(embeddings)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    preds_scaled = model(X).cpu().numpy()
    preds = y_scaler.inverse_transform(preds_scaled)

    # EmoBank VAD labels are on a 1–5 scale.
    preds = np.clip(preds, 1.0, 5.0)

    return preds


# ---------------------------------------------------------------------
# Saving prediction outputs
# ---------------------------------------------------------------------

def save_condition_prediction_csv(
    ids: List,
    texts: List[str],
    preds: np.ndarray,
    condition: str,
    perspective: str,
) -> None:
    out = pd.DataFrame({
        ID_COL: ids,
        "text": texts,
        "valence": preds[:, 0],
        "arousal": preds[:, 1],
        "dominance": preds[:, 2],
    })

    out_path = Path(OUTPUT_DIR) / f"vad_{condition}_{perspective}.csv"
    out.to_csv(out_path, index=False)


def make_long_prediction_df(
    df: pd.DataFrame,
    vad_store: Dict[Tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows = []

    ids = df[ID_COL].tolist()

    for condition in CONDITIONS:
        texts = df[f"{condition}_text"].tolist()

        for perspective in PERSPECTIVES:
            preds = vad_store[(condition, perspective)]

            for i in range(len(df)):
                rows.append({
                    ID_COL: ids[i],
                    "condition": condition,
                    "perspective": perspective,
                    "text": texts[i],
                    "valence": preds[i, 0],
                    "arousal": preds[i, 1],
                    "dominance": preds[i, 2],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Delta analysis
# ---------------------------------------------------------------------

def make_delta_item_level_df(
    df: pd.DataFrame,
    vad_store: Dict[Tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    """
    Creates one wide item-level delta table.

    Columns include:
        delta_anger_reader_valence
        delta_anger_reader_arousal
        ...
    """
    delta_df = df[[ID_COL]].copy()

    # Include texts for easier inspection.
    delta_df["neutral_text"] = df["neutral_text"]
    delta_df["anger_text"] = df["anger_text"]
    delta_df["sadness_text"] = df["sadness_text"]

    for condition in DELTA_CONDITIONS:
        for perspective in PERSPECTIVES:
            baseline = vad_store[("neutral", perspective)]
            affected = vad_store[(condition, perspective)]
            delta = affected - baseline

            for i, dim in enumerate(DIMS):
                delta_df[f"delta_{condition}_{perspective}_{dim}"] = delta[:, i]

    return delta_df


def save_pairwise_delta_csvs(
    df: pd.DataFrame,
    vad_store: Dict[Tuple[str, str], np.ndarray],
) -> None:
    """
    Saves separate human-readable delta files:
        delta_anger_reader.csv
        delta_sadness_reader.csv
        delta_anger_writer.csv
        delta_sadness_writer.csv
    """
    ids = df[ID_COL].tolist()

    for condition in DELTA_CONDITIONS:
        for perspective in PERSPECTIVES:
            baseline = vad_store[("neutral", perspective)]
            affected = vad_store[(condition, perspective)]
            delta = affected - baseline

            out = pd.DataFrame({
                ID_COL: ids,
                "neutral_text": df["neutral_text"].tolist(),
                f"{condition}_text": df[f"{condition}_text"].tolist(),

                "neutral_valence": baseline[:, 0],
                f"{condition}_valence": affected[:, 0],
                "delta_valence": delta[:, 0],

                "neutral_arousal": baseline[:, 1],
                f"{condition}_arousal": affected[:, 1],
                "delta_arousal": delta[:, 1],

                "neutral_dominance": baseline[:, 2],
                f"{condition}_dominance": affected[:, 2],
                "delta_dominance": delta[:, 2],
            })

            out_path = Path(OUTPUT_DIR) / f"delta_{condition}_{perspective}.csv"
            out.to_csv(out_path, index=False)


def summarize_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics and tests for all:
        condition x perspective x VAD dimension
    """
    rows = []

    for condition in DELTA_CONDITIONS:
        for perspective in PERSPECTIVES:
            for dim, dim_label in zip(DIMS, DIM_LABELS):
                col = f"delta_{condition}_{perspective}_{dim}"
                values = delta_df[col].dropna().values.astype(float)

                row = {
                    "condition": condition,
                    "perspective": perspective,
                    "dimension": dim_label,
                    "n": int(len(values)),
                    "mean_delta": float(np.mean(values)),
                    "std_delta": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median_delta": float(np.median(values)),
                    "min_delta": float(np.min(values)),
                    "max_delta": float(np.max(values)),
                    "wilcoxon_p": safe_wilcoxon(values),
                    "cohens_d_paired": cohens_d_paired(values),
                    "effect_median_over_sd": median_over_sd(values),
                    "mean_abs_delta": float(np.mean(np.abs(values))),
                    "percent_positive": float(np.mean(values > 0) * 100.0),
                    "percent_negative": float(np.mean(values < 0) * 100.0),
                    "percent_zero": float(np.mean(np.isclose(values, 0.0)) * 100.0),
                }

                rows.append(row)

    summary = pd.DataFrame(rows)

    # FDR correction across all tested condition x perspective x dimension pairs.
    valid_mask = summary["wilcoxon_p"].notna()

    summary["wilcoxon_p_fdr_bh"] = np.nan
    summary.loc[valid_mask, "wilcoxon_p_fdr_bh"] = multipletests(
        summary.loc[valid_mask, "wilcoxon_p"].values,
        method="fdr_bh",
    )[1]

    summary["significant_raw_0.05"] = summary["wilcoxon_p"] < 0.05
    summary["significant_fdr_0.05"] = summary["wilcoxon_p_fdr_bh"] < 0.05

    return summary


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "─" * 110)
    print(
        f"{'Condition':<10} "
        f"{'Perspective':<12} "
        f"{'Dim':<10} "
        f"{'N':>6} "
        f"{'Mean Δ':>10} "
        f"{'Median Δ':>10} "
        f"{'Std':>10} "
        f"{'p':>12} "
        f"{'p_FDR':>12} "
        f"{'Sig':>6}"
    )
    print("─" * 110)

    for _, row in summary.iterrows():
        sig = "yes" if row["significant_fdr_0.05"] else ""

        print(
            f"{row['condition']:<10} "
            f"{row['perspective']:<12} "
            f"{row['dimension']:<10} "
            f"{int(row['n']):>6} "
            f"{row['mean_delta']:>+10.4f} "
            f"{row['median_delta']:>+10.4f} "
            f"{row['std_delta']:>10.4f} "
            f"{row['wilcoxon_p']:>12.4g} "
            f"{row['wilcoxon_p_fdr_bh']:>12.4g} "
            f"{sig:>6}"
        )

    print("─" * 110)


# ---------------------------------------------------------------------
# Optional run metadata
# ---------------------------------------------------------------------

def save_run_metadata(
    df: pd.DataFrame,
    embedding_model_name: str,
    device: torch.device,
) -> None:
    metadata = {
        "model_dir": MODEL_DIR,
        "output_dir": OUTPUT_DIR,
        "cache_dir": CACHE_DIR,
        "use_embedding_cache": USE_EMBEDDING_CACHE,
        "id_col": ID_COL,
        "neutral_csv": NEUTRAL_CSV,
        "neutral_col": NEUTRAL_COL,
        "anger_csv": ANGER_CSV,
        "anger_col": ANGER_COL,
        "sadness_csv": SADNESS_CSV,
        "sadness_col": SADNESS_COL,
        "n_aligned_examples": int(len(df)),
        "perspectives": PERSPECTIVES,
        "conditions": CONDITIONS,
        "delta_conditions": DELTA_CONDITIONS,
        "dims": DIMS,
        "embedding_model": embedding_model_name,
        "device": str(device),
    }

    out_path = Path(OUTPUT_DIR) / "infer_metadata.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_condition_data()

    # Load one model first to determine the embedding model name.
    _, _, _, embedding_model_name = load_vad_model("reader", device)

    # Make sure writer uses same embedding model.
    _, _, _, writer_embedding_model = load_vad_model("writer", device)
    if writer_embedding_model != embedding_model_name:
        raise ValueError(
            "Reader and writer checkpoints use different embedding models:\n"
            f"  reader: {embedding_model_name}\n"
            f"  writer: {writer_embedding_model}\n"
            "Use the same encoder for both perspectives for clean comparison."
        )

    print(f"Embedding model: {embedding_model_name}")
    encoder = SentenceTransformer(embedding_model_name)

    # Encode each condition once.
    embeddings_by_condition: Dict[str, np.ndarray] = {}

    for condition in CONDITIONS:
        texts = df[f"{condition}_text"].tolist()
        embeddings_by_condition[condition] = encode_condition_texts(
            encoder=encoder,
            embedding_model_name=embedding_model_name,
            condition=condition,
            texts=texts,
        )

    # Predict VAD for every condition x perspective.
    vad_store: Dict[Tuple[str, str], np.ndarray] = {}

    ids = df[ID_COL].tolist()

    for perspective in PERSPECTIVES:
        print(f"\nPredicting perspective: {perspective}")

        model, x_scaler, y_scaler, _ = load_vad_model(perspective, device)

        for condition in CONDITIONS:
            print(f"  Condition: {condition}")

            preds = predict_vad_from_embeddings(
                embeddings=embeddings_by_condition[condition],
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                device=device,
            )

            vad_store[(condition, perspective)] = preds

            save_condition_prediction_csv(
                ids=ids,
                texts=df[f"{condition}_text"].tolist(),
                preds=preds,
                condition=condition,
                perspective=perspective,
            )

        del model
        torch.cuda.empty_cache()

    # Save long-form prediction table.
    pred_long = make_long_prediction_df(df, vad_store)
    pred_long_path = Path(OUTPUT_DIR) / "vad_predictions_long.csv"
    pred_long.to_csv(pred_long_path, index=False)

    # Delta item-level analysis.
    delta_df = make_delta_item_level_df(df, vad_store)
    delta_item_path = Path(OUTPUT_DIR) / "delta_item_level.csv"
    delta_df.to_csv(delta_item_path, index=False)

    # Separate readable delta CSVs.
    save_pairwise_delta_csvs(df, vad_store)

    # Summary statistics and Wilcoxon tests.
    summary = summarize_delta(delta_df)
    summary_path = Path(OUTPUT_DIR) / "delta_summary.csv"
    summary.to_csv(summary_path, index=False)

    print_summary(summary)

    save_run_metadata(df, embedding_model_name, device)

    print("\nSaved outputs:")
    print(f"  {pred_long_path}")
    print(f"  {delta_item_path}")
    print(f"  {summary_path}")
    print(f"  {OUTPUT_DIR}/vad_<condition>_<perspective>.csv")
    print(f"  {OUTPUT_DIR}/delta_<condition>_<perspective>.csv")
    print(f"  {OUTPUT_DIR}/infer_metadata.json")


if __name__ == "__main__":
    main()