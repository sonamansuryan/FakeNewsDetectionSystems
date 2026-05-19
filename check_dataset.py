import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict

base = Path('data/processed')
MAX_PER_DATASET = 5000  # մաքսիմում յուրաքանչյուր dataset-ից

all_dfs = []

for ds_path in sorted(base.iterdir()):
    if not ds_path.is_dir():
        continue

    csv_file = ds_path / 'train' / 'data-00000-of-00001.csv'
    if not csv_file.exists():
        continue

    df = pd.read_csv(csv_file)

    label_col = next((c for c in ['label', 'labels', 'binary_label'] if c in df.columns), None)
    if not label_col:
        continue

    df = df.rename(columns={label_col: 'label'})
    df['source_dataset'] = ds_path.name

    # Balance real/fake within dataset
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]

    min_class = min(len(class_0), len(class_1), MAX_PER_DATASET // 2)

    balanced = pd.concat([
        class_0.sample(n=min_class, random_state=42),
        class_1.sample(n=min_class, random_state=42)
    ]).sample(frac=1, random_state=42)

    all_dfs.append(balanced)

    print(f"{ds_path.name}: {len(balanced)} samples (0:{min_class}, 1:{min_class})")

final_df = pd.concat(all_dfs, ignore_index=True).sample(frac=1, random_state=42)

print(f"\nTotal: {len(final_df)} samples")
print(f"Label 0: {(final_df['label']==0).sum()}")
print(f"Label 1: {(final_df['label']==1).sum()}")