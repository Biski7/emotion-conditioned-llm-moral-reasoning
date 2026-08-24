# Corpus-Generation/Data

    Result outputs and scripts for the Corpus Generation stage.

    ## Subfolders
    - `age_results`, `anger_age_results`, `anger_results`, `middle_age_results`, `old_age_results`, `sadness_age_results`, `sadness_results` — condition-specific outputs
    - `models` — model artifacts
    - `neutral_baseline` — baseline (no emotion/age conditioning) results



# EmoBank_Dataset

    VAD (Valence-Arousal-Dominance) emotion annotations for text, from the [EmoBank](https://github.com/JULIELab/EmoBank) corpus, split by annotator perspective and merged into one file for downstream use.

    ## Files

    | File | Rows | Columns | Description |
    |---|---|---|---|
    | `emobank.csv` | 10,062 | `id, split, V, A, D, text` | Base EmoBank corpus — text with averaged VAD scores (reader + writer combined) and its `train`/`dev`/`test` split. |
    | `reader.csv` | 10,325 | `id, V, A, D, stdV, stdA, stdD, N` | VAD scores from the reader's perspective, with per-dimension standard deviation and annotator count `N`. |
    | `writer.csv` | 10,279 | `id, V, A, D, stdV, stdA, stdD, N` | VAD scores from the writer's perspective, same column layout as `reader.csv`. |
    | `emobank_merged.csv` | 10,062 | `id, split, text, V_reader, A_reader, D_reader, V_writer, A_writer, D_writer` | Output of `merge.py` — `emobank.csv` joined with `reader.csv` and `writer.csv` on `id`, keeping only text/split plus the perspective-specific VAD scores. |

    ## merge.py

    Merges `emobank.csv`, `reader.csv`, and `writer.csv` on `id` into `emobank_merged.csv`, renaming each source's `V`/`A`/`D` columns to `V_reader`/`A_reader`/`D_reader` and `V_writer`/`A_writer`/`D_writer` so both perspectives coexist in one row per text.

    ```bash
    python merge.py
    ```

    Reads the three CSVs from the current directory and writes `emobank_merged.csv`, printing row counts, the resulting columns, a sample, and a missing-value check.


    # NRC
        # ablation_nrc_analysis.py

        Lexical emotion analysis comparing Teenager + Anger and Teenager + Sadness prompt conditions against a neutral baseline, using the NRC Emotion Lexicon. Part of the ETHICS commonsense morality evaluation framework.

        For each condition (Anger+Age, Sadness+Age), the script:

        >> Computes token-level NRC emotion densities for baseline and conditioned justifications across five dimensions: anger, fear, sadness, positive sentiment, negative sentiment
        >> Calculates per-scenario deltas (condition − baseline)
        >> Runs Wilcoxon signed-rank tests to determine whether median shifts are statistically significant
        >> Produces two figures and saves per-scenario lexical scores to CSV

        # decision_label.py
        Computes the three decision-transition patterns from the emotion-conditioned moral evaluation framework, for Anger+Age and Sadness+Age conditions.

        Patterns

        Moral Recovery --> 1→0→1, 0→1→0 --> Baseline error corrected by emotion framing

        Unchanged Decision Rate --> 0→0→0, 1→1→1, 0→1→1, 1→0→0 --> Decision unchanged after conditioning

        Harmful Disruption --> 0→0→1, 1→1→0 --> Baseline-correct decision broken by emotion framing

# VAD/visualize.py

Generates all VAD figures for GPT-oss and Llama 4, side by side, in one run.

    ## Setup

    Edit `MODEL_DIRS` at the top of the script if your folder names differ:

    ```python
    MODEL_DIRS = {
        "GPT-oss": "results_gptoss",
        "Llama 4": "results_llama4",
    }
    ```

    Each folder needs:

    - `delta_{anger,sadness}_{reader,writer}.csv` — columns: `delta_valence, delta_arousal, delta_dominance`
    - `vad_{neutral,anger,sadness}_{reader,writer}.csv` — columns: `text, valence, arousal, dominance`

    ## Run

    python # VAD/visualize.py


    ## Output (`figures_all/`)

    | File | What it shows |
    |---|---|
    | `fig1_delta_bars_merged.png/.pdf` | Mean ΔVAD (condition − neutral) per model × perspective, with Wilcoxon significance stars |
    | `fig3_heatmap_merged.png/.pdf` | Two heatmaps (GPT-oss, Llama 4) side by side on a shared color scale |
    | `fig4_vad_profiles_merged.png/.pdf` | Absolute VAD levels, each bar with its own paired neutral reference (dotted line ± 95% CI) |
    | `table1_delta_summary_merged.md` | Combined summary table (mean, SD, 95% CI, *p*) with a Model column |

    ## Visual convention (consistent across all three figures)

    - **Color** = perspective — blue = Reader, maroon = Writer
    - **Hatch** = model — solid = GPT-oss, `//` = Llama 4
    - **Column** = affect condition (Anger / Sadness)
    - **Row** = VAD dimension (Valence / Arousal / Dominance)

    ## Note on Fig. 4's neutral reference

    Each bar has its own dotted neutral marker (that model's, that perspective's, own neutral mean ± CI) rather than one pooled reader+writer line — this keeps the reference fair and matches what the paired Wilcoxon tests in Fig. 1 / Table 1 actually compare against.



# Dataset attribution: If you use this dataset, please cite the original dataset publication and comply with its licensing and attribution requirements.