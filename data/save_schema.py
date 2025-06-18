# One-time step: Run this where merged_data is available
import pandas as pd

# Load original and new data
merged_data = pd.read_csv("data/original_merged_data.csv")

schema = {
    "columns": merged_data.columns.tolist(),
    "dtypes": {col: str(merged_data[col].dtype) for col in merged_data.columns}
}

import json
with open("data/merged_schema.json", "w") as f:
    json.dump(schema, f, indent=4)