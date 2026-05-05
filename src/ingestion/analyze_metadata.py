import pandas as pd
import numpy as np
import ast

# === CONFIG ===
CSV_PATH = "./data/raw/html_samples_v7/metadata.csv"

# === LOAD ===
df = pd.read_csv(CSV_PATH, dtype={"internal_id": "string"})

# Converti query_params da string a dict
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}

df["query_params"] = df["query_params"].apply(safe_eval)

# === CONVERSIONE SIZE IN KB ===
df["size_kb"] = df["size"] / 1024

print("\n==============================")
print(" ANALISI DATASET")
print("==============================\n")

# === 1. STATISTICHE GENERALI ===
print(" - Numero totale documenti:", len(df))
print(" -  Numero domini unici:", df["domain"].nunique())
print(" -  Numero internal_id unici:", df["internal_id"].nunique())
print(" -  Numero base_path unici:", df["base_path"].nunique())

# === 2. DOCUMENTI PER DOMINIO ===
print("\n==============================")
print(" DOCUMENTI PER DOMINIO")
print("==============================\n")

print(df["domain"].value_counts().head(10))

# === 3. DISTRIBUZIONE DEPTH ===
print("\n==============================")
print(" DISTRIBUZIONE DEPTH")
print("==============================\n")

print(df["depth"].value_counts().sort_index())

# === 4. INTERNAL ID ===
print("\n==============================")
print(" TOP INTERNAL_ID")
print("==============================\n")

print(df["internal_id"].value_counts().head(10))

# === 5. ANALISI DIMENSIONI FILE ===
print("\n==============================")
print(" ANALISI DIMENSIONI FILE (KB)")
print("==============================\n")

percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

size_stats = df["size_kb"].describe(percentiles=percentiles)
print(size_stats)

print("\n- Media (KB):", round(df["size_kb"].mean(), 2))
print("- Mediana (KB):", round(df["size_kb"].median(), 2))
print("- Std Dev (KB):", round(df["size_kb"].std(), 2))

# === 6. DISTRIBUZIONE PER INTERVALLI ===
print("\n==============================")
print(" DISTRIBUZIONE DIMENSIONI (BINNING)")
print("==============================\n")

bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf]
labels = [
    "0-1 KB", "1-2 KB", "2-3 KB", "3-4 KB", "4-5 KB",
    "5-10 KB", "10-20 KB", "20-50 KB", "50-100 KB", ">100 KB"
]

df["size_bin"] = pd.cut(df["size_kb"], bins=bins, labels=labels)

bin_counts = df["size_bin"].value_counts().sort_index()
bin_percent = df["size_bin"].value_counts(normalize=True).sort_index() * 100

print("Conteggi:")
print(bin_counts)

print("\nPercentuali:")
print(bin_percent.round(2).astype(str) + " %")

# === 7. OUTLIER ===
print("\n==============================")
print(" FILE PIÙ GRANDI")
print("==============================\n")

largest = df.sort_values(by="size_kb", ascending=False).head(10)
print(largest[["doc_id", "domain", "size_kb"]])

# === 8. QUERY PARAMS ===
print("\n==============================")
print(" ANALISI QUERY PARAMS")
print("==============================\n")

param_counts = {}

for params in df["query_params"]:
    for key in params:
        param_counts[key] = param_counts.get(key, 0) + 1

sorted_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)

for k, v in sorted_params[:10]:
    print(f"{k}: {v}")

# === 9. CROSS ANALYSIS ===
print("\n==============================")
print(" CROSS ANALYSIS (domain vs depth)")
print("==============================\n")

pivot = pd.pivot_table(
    df,
    index="domain",
    columns="depth",
    values="doc_id",
    aggfunc="count",
    fill_value=0
)

print(pivot.head(10))

# === 10. INSIGHT ML ===
print("\n==============================")
print(" INSIGHT ")
print("==============================\n")

print("Distribuzione depth (normalizzata):")
print(df["depth"].value_counts(normalize=True).sort_index())

print("\nTop domini (%):")
print(df["domain"].value_counts(normalize=True).head(5))

print("\nDone.")