import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch, json
from pathlib import Path

# 1 Load dataset
file_path = Path("diseases_symptoms.csv")
df = pd.read_csv(file_path)

print ("Dataset loaded:")
print(df.head())

df = df.rename(columns={"Disease": "disease", "Symptom": "symptoms"})
df["symptoms"] = df["symptoms"].astype(str)

# 2 Encode labels 
labels = {d: i for i, d in enumerate(df["disease"].unique())}
inv_labels = {v: k for k, v in labels.items()}
df["label"] = df["disease"].map(labels)
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print("\n Label map created:")
print(labels)

# 3 split train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["symptoms"], df["label"], test_size=0.2, random_state=42
)
train_df = pd.DataFrame({"text": train_texts, "label" : train_labels})
test_df = pd.DataFrame({"text": test_texts, "label": test_labels})

#Convert to Hugging face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 4 Tokenization 
model_name = "distilbert-base-uncased"
tokenizer =  AutoTokenizer.from_pretrained(model_name)

def tokenize (batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 5 Load model 
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels)
)

# 6 Training setup 
training_args = TrainingArguments(
     output_dir="./results_one_dataset",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1
)

# 7 Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 8 Train 
print("n\ Starting training...")
trainer.train()

# 9 Save model 
model.save_pretrained("trained_model_one")
tokenizer.save_pretrained("trained_model_one")

print("\n Training complete. Model saved in 'trained_model_one/'")
