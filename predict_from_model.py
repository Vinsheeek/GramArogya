# predict_from_model.py
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----- CONFIG -----
MODEL_DIR = Path("trained_model_clean")
LABEL_MAP = Path("label_map_clean.json")   # created during training
PRECAUTIONS_CSV = Path("data/precaution.csv")        # optional
SEVERITY_CSV = Path("data/Symptom-severity.csv")     # optional
TOPK = 3
MAX_LEN = 96
# ------------------

if not MODEL_DIR.exists():
    print("Model folder not found:", MODEL_DIR); sys.exit(1)
if not LABEL_MAP.exists():
    print("Label map not found:", LABEL_MAP); sys.exit(1)

# load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
id2label = {int(v): k for k, v in label_map.items()}

# load precautions map (best-effort)
prec_map = {}
if PRECAUTIONS_CSV.exists():
    dfp = pd.read_csv(PRECAUTIONS_CSV)
    cols = [c.lower() for c in dfp.columns]
    if "disease" in cols and "precautions" in cols:
        dcol = dfp.columns[cols.index("disease")]
        pcol = dfp.columns[cols.index("precautions")]
        prec_map = dict(zip(dfp[dcol].astype(str), dfp[pcol].astype(str)))
    else:
        # try common variants
        if "disease" in dfp.columns:
            prec_map = dict(zip(dfp["disease"].astype(str), dfp[dfp.columns[-1]].astype(str)))

# load symptom severity weights (optional)
sev_map = {}
if SEVERITY_CSV.exists():
    sdf = pd.read_csv(SEVERITY_CSV)
    # try to find 'symptom' and 'weight' columns
    cols = [c.lower() for c in sdf.columns]
    if "symptom" in cols and ("weight" in cols or "severity" in cols):
        s_col = sdf.columns[cols.index("symptom")]
        w_col = sdf.columns[cols.index("weight")] if "weight" in cols else sdf.columns[cols.index("severity")]
        sev_map = dict(zip(sdf[s_col].astype(str).str.lower(), sdf[w_col].astype(float)))

def predict_topk(symptoms_text, k=TOPK):
    inputs = tokenizer(symptoms_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    idx = np.argsort(probs)[::-1][:k]
    return [(id2label[int(i)], float(probs[int(i)])) for i in idx]

def compute_severity(symptoms_text):
    syms = [s.strip().lower() for s in symptoms_text.split(",") if s.strip()]
    score = sum(sev_map.get(s, 0) for s in syms)
    if score < 5:
        level = "🟢 Mild (Home care)"
    elif score < 10:
        level = "🟡 Moderate (Doctor visit suggested)"
    else:
        level = "🔴 Severe (Immediate referral)"
    return score, level

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter symptoms (comma-separated): ").strip()
    if not text:
        print("No input given. Exiting."); sys.exit(0)

    topk = predict_topk(text)
    score, level = compute_severity(text)

    print("\nTop predictions:")
    for disease, p in topk:
        print(f" - {disease} ({p*100:.1f}%)")
        print("   Precautions:", prec_map.get(disease, "No precautions available"))
    print(f"\nSeverity score: {score} -> {level}")
