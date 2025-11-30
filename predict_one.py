import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd

# 1) Load model + tokenizer
model = AutoModelForSequenceClassification.from_pretrained("trained_model_one")
tokenizer = AutoTokenizer.from_pretrained("trained_model_one")

# 2) Load label map
with open("label_map.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# 3) Load precautions
precautions_df = pd.read_csv("data/precaution.csv")

# Check if precautions file has multiple columns (Precaution_1, Precaution_2, etc.)
if "precautions" in precautions_df.columns:
    # case 1: single precautions column
    prec_map = dict(zip(precautions_df["disease"], precautions_df["precautions"]))
else:
    # case 2: multiple precaution columns
    prec_map = {}
    for _, row in precautions_df.iterrows():
        disease = row["Disease"] if "Disease" in row else row["disease"]
        steps = [str(val) for col, val in row.items() if "Precaution" in col and pd.notna(val)]
        prec_map[disease] = "; ".join(steps)

# 4) Prediction function
def predict(symptom_text):
    inputs = tokenizer(symptom_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    disease = id2label.get(pred_id, "Unknown")
    precautions = prec_map.get(disease, "No precautions available")

    return disease, precautions   # ✅ Always return two values


# 5) Interactive loop
print("🩺 Healthcare Assistant — type symptoms like 'fever,cough' (or 'quit' to exit)\n")

while True:
    text = input("Enter symptoms: ").strip().lower()

    if text in ["quit", "exit", "q"]:
        print("Exiting. Stay healthy! ✨")
        break

    disease, precautions = predict(text)
    print(f"\nPredicted Disease: {disease}")
    print(f"Precautions: {precautions}\n")
