import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load model + tokenizer
model = AutoModelForSequenceClassification.from_pretrained("trained_model")
tokenizer = AutoTokenizer.from_pretrained("trained_model")

# Load label map
with open("label_map.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# Prediction function
def predict(symptom_text):
    inputs = tokenizer(symptom_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
    return id2label[pred_id]

# --- Test ---
while True:
    text = input("Enter symptoms (comma-separated, or 'quit' to exit): ")
    if text.lower() == "quit":
        print("Exiting prediction loop. Goodbye!")
        break
    prediction = predict(text)
    print(f"Predicted Disease: {prediction}\n")