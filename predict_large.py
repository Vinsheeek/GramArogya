import torch, json, pandas as pd
import numpy as np 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1 load model/tokenizer/labels
model_dir = "trained_model_large"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

with open("label_map.json","r",encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {int(v): k for k, v in label2id.items()}  # ensure int keys if needed

# 2) load precautions flexibly
prec_df = pd.read_csv("data/precaution.csv")
if "precautions" in prec_df.columns and "disease" in prec_df.columns:
    prec_map = dict(zip(prec_df["disease"], prec_df["precautions"]))
else:
    # multi-column style: Disease, Precaution_1..N
    disease_col = "Disease" if "Disease" in prec_df.columns else "disease"
    prec_cols = [c for c in prec_df.columns if "Precaution" in c]
    prec_map = {}
    for _, r in prec_df.iterrows():
        steps = [str(r[c]) for c in prec_cols if pd.notna(r[c])]
        prec_map[str(r[disease_col])] = "; ".join(steps) if steps else "No precautions available"

def predict_topk(symptom_text, k=3):
    inputs = tokenizer(symptom_text, return_tensors="pt", padding=True, truncation=True, max_length=96)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    probs = np.exp(logits - logits.max())  # softmax stable trick
    probs = probs / probs.sum()
    classes = np.arange(len(probs))
    topk_idx = classes[np.argsort(probs)[::-1][:k]]
    result = [(id2label[int(i)], float(probs[int(i)])) for i in topk_idx]
    return result

print("🩺 Type symptoms like 'fever, cough, headache' (or 'quit' to exit)\n")
while True:
    text = input("Enter symptoms: ").strip()
    if text.lower() in ["quit", "exit", "q"]:
        print("Exiting. Stay healthy! ✨")
        break

    top3 = predict_topk(text, k=3)
    print("\nTop predictions:")
    for disease, p in top3:
        pct = round(p*100, 1)
        print(f" - {disease} ({pct}%)")
        print(f"   Precautions: {prec_map.get(disease,'No precautions available')}")
    print()