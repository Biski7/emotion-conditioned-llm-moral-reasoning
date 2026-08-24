import csv
import os
import pandas as pd
from tqdm import tqdm
from prompts import generate_age_prompt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL  = "llama4"   # gpt_oss 

# Use scenarios from csv
if MODEL == "gpt_oss":
    INPUT  = '...'
elif MODEL == "llama4":
    INPUT  = '...'
run = 2

OUTPUT     = f"age_results/age_emotion_results_full_{MODEL}_{run}.csv"
OUTPUT_TXT = f"age_results/age_emotion_results_full_{MODEL}.txt"

# ── Model selection ───────────────────────────────────────────────────────────
if MODEL == "gpt_oss":
    from models.gpt_oss import query_gptoss_ollama as query_fn
elif MODEL == "llama4":
    from models.llama4 import query_ollama4 as query_fn
else:
    raise ValueError(f"Unknown model: {MODEL}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_label(raw) -> str:
    for ch in str(raw or ""):
        if ch in ("0", "1"):
            return ch
    return ""

def to_int(val) -> int:
    try:
        return int(float(str(val).strip()))
    except:
        return -1

def append_txt(path: str, lines: list[str]) -> None:
    """Append a block of lines to the txt results file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ── Step 1: Load full dataset ─────────────────────────────────────────────────
df = pd.read_csv(INPUT, encoding="utf-8")
df.columns = [c.strip() for c in df.columns]
df = df.reset_index(drop=True)

print(f"Total scenarios to process: {len(df)}")

# ── Step 2: Run anger+age inference on ALL scenarios (resumable) ──────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
file_exists  = os.path.isfile(OUTPUT)
already_done = set(pd.read_csv(OUTPUT)["ID"].astype(str).tolist()) if file_exists else set()
fieldnames   = ["ID", "Label", "Scenario", "Base0-shot", "Justification",
                "Age_Label", "Age_Justification", "Age_Full_Response"]

with open(OUTPUT, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Age inference (all)"):
        if str(row["ID"]) in already_done:
            continue

        prompt = generate_age_prompt(str(row["Scenario"]))
        full_response, raw_score, justification = query_fn(prompt)
        age_label = parse_label(raw_score)

        writer.writerow({
            "ID": row["ID"], "Label": row["Label"], "Scenario": row["Scenario"],
            "Base0-shot": row["Base0-shot"], "Justification": row["Justification"],
            "Age_Label": age_label, "Age_Justification": justification,
            "Age_Full_Response": full_response,
        })
        f.flush()

        log_line = f"ID {row['ID']} | GT: {row['Label']} | Original: {row['Base0-shot']} | Age: {age_label}"
        print(log_line)
        # append_txt(OUTPUT_TXT, [log_line])

# ── Step 3: Compute accuracy metrics on the FULL set ─────────────────────────
result_df = pd.read_csv(OUTPUT, encoding="utf-8")
labels    = result_df["Label"].apply(to_int)
base0     = result_df["Base0-shot"].apply(to_int)
age = result_df["Age_Label"].apply(to_int)
total     = len(result_df)

orig_correct      = (labels == base0)
age_correct = (labels == age)

orig_acc      = orig_correct.sum() / total * 100
age_acc = age_correct.sum() / total * 100

# Transition counts
correct_to_correct = ( orig_correct &  age_correct).sum()
correct_to_wrong   = ( orig_correct & ~age_correct).sum()
wrong_to_correct   = (~orig_correct &  age_correct).sum()
wrong_to_wrong     = (~orig_correct & ~age_correct).sum()

# Directional 1→0→1 pattern
mask_1_0_1 = (labels == 1) & (base0 == 0) & (age == 1)
mask_0_1_0 = (labels == 0) & (base0 == 1) & (age == 0)

summary_lines = [
    f"\n{'='*60}",
    f"FULL-SET EVALUATION ({MODEL}) — Age",
    f"{'='*60}",
    f"Total evaluated                  : {total}",
    f"Original accuracy                : {orig_acc:.2f}%  ({orig_correct.sum()}/{total})",
    f"Age accuracy                     : {age_acc:.2f}%  ({age_correct.sum()}/{total})",
    f"Delta (Age − Original)           : {age_acc - orig_acc:+.2f}%",
    f"{'-'*60}",
    f"Correct → Correct (kept)         : {correct_to_correct}",
    f"Correct → Wrong   (broke)        : {correct_to_wrong}",
    f"Wrong   → Correct (helped)       : {wrong_to_correct}",
    f"Wrong   → Wrong   (no help)      : {wrong_to_wrong}",
    f"{'-'*60}",
    f"1→0→1 corrections (morally-wrong recovered): {mask_1_0_1.sum()}",
    f"0→1→0 flips        (morally-ok corrupted)  : {mask_0_1_0.sum()}  [excluded from narrative]",
    f"{'-'*60}",
    f"Results saved to                 : {OUTPUT}",
    f"TXT log saved to                 : {OUTPUT_TXT}",
]

for line in summary_lines:
    print(line)

append_txt(OUTPUT_TXT, summary_lines)