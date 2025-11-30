# prep_dataset.py
import pandas as pd
import re
from pathlib import Path

DATA = Path("data/dataset.csv")
OUT = Path("data/clean_dataset.csv")

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9, ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

print("\n🔄 Loading dataset.csv...")
df = pd.read_csv(DATA)

# find symptom columns
sym_cols = [c for c in df.columns if c.lower().startswith("symptom")]
print(f"✔ Found {len(sym_cols)} symptom columns")

# combine symptoms into text
def join_symptoms(row):
    parts = []
    for col in sym_cols:
        val = row.get(col)
        if pd.isna(val):
            continue
        cleaned = clean_text(val)
        if cleaned:
            parts.append(cleaned)
    # remove duplicates
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)

print("🔧 Cleaning rows...")
df["text"] = df.apply(join_symptoms, axis=1)
df = df.rename(columns={"Disease": "disease"})

# remove empty rows
df = df[df["text"].str.strip() != ""]

# save output
df[["text", "disease"]].to_csv(OUT, index=False)

print("\n🎉 CLEAN DATASET CREATED!")
print(f"Saved to: {OUT}")
print(f"Total rows: {len(df)}\n")
