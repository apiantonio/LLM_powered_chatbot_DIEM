import os
import re
import pandas as pd

# Cartella contenente i file
FOLDER_PATH = "data/raw/html_samples_cleaned"

data = []

# Pattern principale
pattern = re.compile(r'^doc(\d+)_depth(\d+)_(.+)$')

for filename in os.listdir(FOLDER_PATH):
    filepath = os.path.join(FOLDER_PATH, filename)

    if not os.path.isfile(filepath):
        continue

    match = pattern.match(filename)
    if not match:
        continue

    doc_id = int(match.group(1))
    depth = int(match.group(2))
    rest = match.group(3)

    # --- STEP 1: dominio + resto ---
    parts = rest.split('-', 1)
    domain = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    # Rimuove estensione
    path = path.replace(".html", "")

    # --- STEP 2: split intelligente ---
    tokens = path.split('-')

    internal_id = None
    base_path_parts = []
    query_params = {}

    for token in tokens:
        # Caso ID numerico (tipo 041789)
        if token.isdigit() and internal_id is None:
            internal_id = token

        # Caso parametro tipo key=value
        elif "=" in token:
            pairs = token.split("&")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    query_params[key] = value

        # Parte del path
        else:
            base_path_parts.append(token)

    base_path = "-".join(base_path_parts)

    # Dimensione file
    size = os.path.getsize(filepath)

    data.append({
        "doc_id": doc_id,
        "depth": depth,
        "domain": domain,
        "internal_id": internal_id,
        "base_path": base_path,
        "query_params": query_params,
        "size": size
    })

# --- DataFrame ---
df = pd.DataFrame(data)

# Ordina per size (opzionale)
df = df.sort_values(by="size")

# Salva CSV
samples_dir = os.path.basename(FOLDER_PATH)
output_path = os.path.join(f"data/evaluation/", f"metadata_{samples_dir}.csv")
df.to_csv(output_path, index=False, encoding="utf-8")


print(f"CSV generato: {output_path}")