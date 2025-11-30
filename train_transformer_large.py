import json, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

# ========= USER CHOICE: which file to train on =========
# Set ONE of these:
DATA_FILE = Path("data/dataset.csv")         # option A
# DATA_FILE = Path("data/symptom_description.csv") # option B
# =======================================================

assert DATA_FILE.exists(), f"Could not find: {DATA_FILE} (check path/filename)"

print(f"✅ Loading: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print("Columns:", list(df.columns)[:30])
print("Head:\n", df.head(3))

# ---- 1) AUTO-DETECT SCHEMA & BUILD TEXT/LABEL ----
text_col = None
label_col = None

# common label guesses
label_candidates = ["prognosis", "disease", "Disease", "label", "Label"]
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    raise ValueError("Could not find a label column. Expected one of: 'prognosis','disease','Disease','label'.")

# case A: wide symptom matrix or Symptom_1..N
symptom_like_cols = [c for c in df.columns if "symptom" in c.lower() or c.lower().startswith("symptom_")]
# also detect binary symptom columns (0/1) if present
binary_symptom_cols = []
for c in df.columns:
    if c == label_col: 
        continue
    # if column has only 0/1 or NaN → likely a binary symptom flag
    vals = pd.Series(df[c]).dropna().unique()
    if len(vals) > 0 and set(pd.Series(vals).astype(str)) <= set(["0","1"]):
        binary_symptom_cols.append(c)

# case B: prose description column
description_candidates = ["description", "Description", "desc", "Desc"]
desc_col = None
for c in description_candidates:
    if c in df.columns:
        desc_col = c
        break

def row_to_text_from_binary(row, cols):
    # join column names where value==1
    syms = [c for c in cols if str(row[c]) == "1"]
    return ", ".join(syms) if syms else ""

def row_to_text_from_symptomN(row, cols):
    # join Symptom_1..N that are non-empty
    vals = []
    for c in cols:
        v = str(row.get(c, "")).strip()
        if v and v.lower() != "nan" and v != "0":
            vals.append(v)
    return ", ".join(vals)

if desc_col:
    # use descriptions as text
    print(f" Using '{desc_col}' as input text; label = '{label_col}'")
    text_series = df[desc_col].astype(str).fillna("")
elif symptom_like_cols:
    # if we have Symptom_1..N style columns
    print(f" Using Symptom_* columns as input text; label = '{label_col}'")
    text_series = df.apply(lambda r: row_to_text_from_symptomN(r, symptom_like_cols), axis=1)
elif binary_symptom_cols:
    # binary symptom matrix
    print(f" Using binary symptom columns as input text; label = '{label_col}'")
    text_series = df.apply(lambda r: row_to_text_from_binary(r, binary_symptom_cols), axis=1)
else:
    raise ValueError("Could not detect text features. Need either a Description column, Symptom_1..N columns, or many 0/1 symptom columns.")

# final tidy dataframe
work_df = pd.DataFrame({
    "text": text_series.fillna("").astype(str),
    "label_text": df[label_col].astype(str)
})

# remove empties
work_df = work_df[work_df["text"].str.strip() != ""].copy()
work_df.dropna(subset=["label_text"], inplace=True)

# encode labels
labels = {d: i for i, d in enumerate(sorted(work_df["label_text"].unique()))}
id2label = {v: k for k, v in labels.items()}
work_df["label"] = work_df["label_text"].map(labels)

print(f" Classes: {len(labels)}")
print("Example rows:\n", work_df.head(5)[["text","label_text","label"]])

# ---- 2) TRAIN/TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    work_df["text"], work_df["label"], test_size=0.2, random_state=42, stratify=work_df["label"]
)

train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df  = pd.DataFrame({"text": X_test,  "label": y_test})

# ---- 3) HF DATASETS + TOKENIZER ----
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tok(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=96)

train_ds = Dataset.from_pandas(train_df).map(tok, batched=True)
test_ds  = Dataset.from_pandas(test_df).map(tok, batched=True)

train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
test_ds.set_format("torch",  columns=["input_ids","attention_mask","label"])

# ---- 4) MODEL ----
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=labels,
)

# ---- 5) TRAINING ARGS (kept simple for max compat) ----
args = TrainingArguments(
    output_dir="./results_large",
    num_train_epochs=6,                 # bigger data → learn better
    per_device_train_batch_size=8,      # lower if OOM
    learning_rate=2e-5,
    logging_steps=50,
    save_total_limit=1,
    gradient_accumulation_steps=1,      # set 2/4 if you hit RAM issues
)

# ---- 6) METRICS ----
def compute_metrics(eval_pred):
    logits, labels_np = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,   # deprecation warning is fine
    compute_metrics=compute_metrics,
)

print("\n Starting training...")
trainer.train()

print("\n Evaluating...")
metrics = trainer.evaluate()
print("Metrics:", metrics)

# ---- 7) SAVE (avoid safetensors to reduce memory issues on Windows) ----
save_dir = "trained_model_large"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print(f"\n Done. Model saved to '{save_dir}/'. Label map saved to 'label_map.json'.")
