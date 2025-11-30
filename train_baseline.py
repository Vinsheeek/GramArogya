import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load data
symptoms = pd.read_csv("diseases_symptoms.csv")
precautions = pd.read_csv("precautions.csv")

print(" Data loaded successfully!")

# Example: Print first rows
print(symptoms.head())
print(precautions.head())


# train_baseline.py
# Step 2: just verify files, load data, and build a label map. No training yet.

import json
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()

# 1) verify files exist right next to this script
need = ["diseases_symptoms.csv", "precautions.csv"]
for f in need:
    p = ROOT / f
    if not p.exists():
        raise FileNotFoundError(f"Expected '{f}' in {ROOT}, but it wasn't found.")

print("✅ Found CSV files:", need)

# 2) load the CSVs
symptoms_df = pd.read_csv("diseases_symptoms.csv")
precautions_df = pd.read_csv("precautions.csv")

print("\n Loaded 'diseases_symptoms.csv'")
print(symptoms_df.head(3))
print("\n Loaded 'precautions.csv'")
print(precautions_df.head(3))

# 3) build a disease label map (text label -> integer id)
if "disease" not in symptoms_df.columns or "symptoms" not in symptoms_df.columns:
    raise ValueError("The CSV must have 'symptoms' and 'disease' columns.")

diseases = sorted(symptoms_df["disease"].astype(str).unique().tolist())
label2id = {d: i for i, d in enumerate(diseases)}
id2label = {i: d for d, i in label2id.items()}

# 4) save mapping for later steps
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)

print("\n✅ Created label map (disease -> id):")
print(label2id)
print("\n✅ Wrote label_map.json in your project folder.")
print("\n Step 2 complete: data is readable and labels are mapped.")

from datasets import Dataset
from transformers import AutoTokenizer

# 5) Convert pandas dataframe to Hugging Face Dataset
dataset = Dataset.from_pandas(symptoms_df[["symptoms", "disease"]].rename(columns={"symptoms": "text"}))
print("\n✅ Converted to Hugging Face Dataset:")
print(dataset)

# 6) Add label ids
dataset = dataset.map(lambda ex: {"label": label2id[ex["disease"]]})
print("\n✅ Added labels (first 3):")
print(dataset[:3])

# 7) Split train/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 8) Load tokenizer
model_name = "distilbert-base-uncased"   # light model for CPU
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 9) Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 10) Format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print("\n Step 4 complete: data is tokenized and ready for training.")

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# 11) Load model for classification 
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# 12) Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=5,
)

#13) Define Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

#14) Train 
print("\n training finished. Now saving model ...")
trainer.train()

#15) Save model + tokenizer
model.save_pretrained("trained_model", safe_serialization=False)
tokenizer.save_pretrained("trained_model")

print("\n Model and tokenizer saved in 'trained_model/' folder without safetensors.")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=9)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("trained_model", safe_serialization=False)
tokenizer.save_pretrained("trained_model")
