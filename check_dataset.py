import pandas as pd

df = pd.read_csv("data/dataset.csv")

print("🔍 First 5 rows:")
print(df.head())

print("\n📋 Column names:")
print(df.columns.tolist())

print("\n📊 Number of rows:", len(df))
