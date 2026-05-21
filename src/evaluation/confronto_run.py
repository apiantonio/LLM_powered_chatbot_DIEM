import json, pandas as pd

files = ["results_summary_run1.json", "results_summary_run2.json"]
rows = []
for f in files:
    data = json.load(open(f))
    row = {"run_id": data["run_id"], **data["aggregate_scores"]}
    if "run_metadata" in data:
        row.update(data["run_metadata"])
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_markdown(index=False))