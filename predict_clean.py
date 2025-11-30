# predict_clean.py
# Simple inference script for the trained classifier (trained_model_clean)
# Usage:
#   python predict_clean.py            -> interactive prompt
#   python predict_clean.py "fever, cough"  -> single-shot prediction

import sys
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("trained_model_clean")
LABEL_MAP_FILE = Path("label_map_clean.json")
MAX_LEN = 96

if not MODEL_DIR.exists():
    print(f"ERROR: trained model folder not found: {MODEL_DIR}")
    sys.exit(1)
if not LABEL_MAP_FILE.exists():
    print(f"ERROR: label map file not found: {LABEL_MAP_FILE}")
    sys.exit(1)

# load model + tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    print("Failed to load model or tokenizer:", e)
    sys.exit(1)

with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
    label_map = json.load(f)
# label_map: {label_text: id}
inv_label_map = {int(v): k for k, v in label_map.items()}


def predict_once(text):
    text = str(text).strip()
    if not text:
        return None
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    best_idx = int(probs.argmax())
    best_label = inv_label_map.get(best_idx, str(best_idx))
    best_conf = float(probs[best_idx])

    # prepare top-3
    topk_idx = list(reversed(probs.argsort()))[:3]
    topk = [(inv_label_map.get(int(i), str(i)), float(probs[int(i)])) for i in topk_idx]

    return best_label, best_conf, topk


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        res = predict_once(text)
        if res is None:
            print("No input provided.")
            sys.exit(0)
        best_label, best_conf, topk = res
        print(f"\nPredicted: {best_label} ({best_conf*100:.1f}% confidence)")
        print("Top predictions:")
        for d, p in topk:
            print(f" - {d}: {p*100:.1f}%")
        sys.exit(0)

    print("\n🩺 Predict Clean Model — interactive\n(type symptoms like: fever, cough  OR 'quit' to exit)\n")
    while True:
        try:
            text = input("Enter symptoms: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not text:
            continue
        if text.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        res = predict_once(text)
        if res is None:
            print("No valid input.")
            continue
        best_label, best_conf, topk = res
        print(f"\nPredicted disease: {best_label} ({best_conf*100:.1f}% confidence)")
        print("Top predictions:")
        for d, p in topk:
            print(f" - {d}: {p*100:.1f}%")
        print()
