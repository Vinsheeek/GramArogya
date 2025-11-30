import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import sys
from rapidfuzz import process, fuzz

MODEL_DIR = Path("trained_model_large")
LABEL_MAP = Path("label_map.json")
PREC_FILE = Path("data/precaution.csv")
SEV_FILE = Path("data/symptom-severity.csv")

# --- sanity checks ---
if not MODEL_DIR.exists():
    print("❌ Model folder 'trained_model_large' not found. Run training first.")
    sys.exit(1)
if not LABEL_MAP.exists():
    print("❌ label_map.json not found. Run training first.")
    sys.exit(1)

# --- load model & tokenizer ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- load labels ---
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {int(v): str(k) for k, v in label2id.items()}

# --- load precautions ---
prec_map = {}
if PREC_FILE.exists():
    prec_df = pd.read_csv(PREC_FILE)
    cols = [c.lower() for c in prec_df.columns]
    col_map = {c.lower(): c for c in prec_df.columns}

    if "disease" in cols and "precautions" in cols:
        dcol = col_map["disease"]
        pcol = col_map["precautions"]
        prec_map = dict(zip(prec_df[dcol].astype(str), prec_df[pcol].astype(str)))
    else:
        dcol = col_map["disease"] if "disease" in cols else "Disease"
        pcols = [c for c in prec_df.columns if "precaution" in c.lower()]
        for _, r in prec_df.iterrows():
            steps = [str(r[c]) for c in pcols if pd.notna(r[c])]
            prec_map[str(r[dcol])] = "; ".join(steps) if steps else "No precautions available"

# --- load severity ---
sev_map = {}
if SEV_FILE.exists():
    sev_df = pd.read_csv(SEV_FILE)
    sev_df.columns = [c.lower().strip() for c in sev_df.columns]
    if "symptom" in sev_df.columns and "weight" in sev_df.columns:
        sev_map = dict(zip(sev_df["symptom"].astype(str).str.lower(), sev_df["weight"]))

# --- fuzzy-match helper for symptom suggestions ---
# create a list of all known symptom names
known_symptoms = sorted(list(sev_map.keys()))

def suggest_symptom(word: str, score_cutoff: int = 70):
    """
    Finds the closest known symptom name to a misspelled word.
    Example: 'headach' -> 'headache'
    """
    if not known_symptoms:
        return None
    match = process.extractOne(word, known_symptoms, scorer=fuzz.WRatio)
    if match and match[1] >= score_cutoff:  # 70% similarity or higher
        return match[0]
    return None

# --- prediction ---
def predict_topk(symptom_text: str, k: int = 3):
    inputs = tokenizer(symptom_text, return_tensors="pt", padding=True, truncation=True, max_length=96)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    idx = np.argsort(probs)[::-1][:k]
    return [(id2label[int(i)], float(probs[int(i)])) for i in idx]

# --- severity scoring with unknown handling ---
def compute_severity(symptom_text: str):
    """
    Returns: (score: int or None, level: str, unknown: List[str])
    """
    if not sev_map:
        return None, "⚠️ Severity data not available.", []

    syms = [s.strip().lower() for s in symptom_text.split(",") if s.strip()]
    score = 0
    unknown = []

    for s in syms:
        if s in sev_map:
            score += sev_map[s]
        else:
            unknown.append(s)

    if score < 5:
        level = "🟢 Mild (Home care is enough)"
    elif score < 10:
        level = "🟡 Moderate (Doctor visit suggested)"
    else:
        level = "🔴 Severe (Immediate referral recommended)"

    return score, level, unknown

# --- interactive loop ---
print("🩺 Advanced Healthcare Assistant — type symptoms like 'fever, cough' (or 'quit' to exit)\n")

while True:
    text = input("Enter symptoms: ").strip()
    if text.lower() in ["quit", "exit", "q"]:
        print("Exiting. Stay healthy! ✨")
        break
    if not text:
        print("Please type some symptoms.")
        continue

    top3 = predict_topk(text, k=3)
    score, level, unknown = compute_severity(text)

    print("\nTop predictions:")
    for disease, p in top3:
        pct = round(p * 100, 1)
        advice = prec_map.get(disease, "No precautions available")
        print(f" - {disease} ({pct}%)")
        print(f"   Precautions: {advice}")

    if score is not None:
        print(f"\n⚖️ Severity Score: {score}")
        print(f"Recommendation: {level}")

    # 🔥 Always show unknowns if any
    if unknown:
        print(f"⚠️ Unrecognized symptoms ignored: {', '.join(unknown)}\n")
    else:
        print()

while true:
def compute_severity(symptom_text: str, suggestion_cutoff: int = 70):
    """
    Returns:
      score: int or None
      level: str
      unknown: List[str]
      suggestions: Dict[str, str]  (unknown -> suggested_symptom)
    """
    if not sev_map:
        return None, "⚠️ Severity data not available.", [], {}

    # split user text into symptom tokens (lowercased, trimmed)
    syms = [s.strip().lower() for s in symptom_text.split(",") if s.strip()]
    score = 0
    unknown = []
    suggestions = {}

    for s in syms:
        if s in sev_map:
            score += sev_map[s]
        else:
            unknown.append(s)
            # try to find a fuzzy suggestion (may return None)
            sug = suggest_symptom(s, score_cutoff=suggestion_cutoff)
            if sug:
                suggestions[s] = sug

    # severity bucket
    if score < 5:
        level = "🟢 Mild (Home care is enough)"
    elif score < 10:
        level = "🟡 Moderate (Doctor visit suggested)"
    else:
        level = "🔴 Severe (Immediate referral recommended)"

    return score, level, unknown, suggestions
