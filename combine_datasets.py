import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

BASE_DIR = Path('data/processed')
OUTPUT_DIR = Path('data/combined/combined_dataset')
MAX_PER_DATASET = 8000
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_STATE = 42

DATASET_CONFIGS = {
    'covid_19': {
        'text_col': 'text',
        'label_col': 'label',
    },
    'fakenewsnet': {
        'text_col': ['title', 'text'],
        'label_col': 'label',
    },
    'fever': {
        'text_col': 'text',
        'label_col': 'label',
    },
    'liar': {
        'text_col': 'text',
        'label_col': 'label',
    },
}


def load_and_balance_dataset(ds_path: Path, config: dict) -> pd.DataFrame:
    csv_file = ds_path / 'train' / 'data-00000-of-00001.csv'
    if not csv_file.exists():
        print(f"  ⚠️  Skipping {ds_path.name}: CSV not found")
        return None

    df = pd.read_csv(csv_file)

    # --- Text column ---
    text_col = config['text_col']
    if isinstance(text_col, list):
        # fakenewsnet: title + text միացնել
        df['text'] = df[text_col[0]].fillna('') + ' ' + df[text_col[1]].fillna('')
        df['text'] = df['text'].str.strip()
    else:
        df = df.rename(columns={text_col: 'text'})

    # --- Label column ---
    label_col = config['label_col']
    df = df.rename(columns={label_col: 'label'})

    # --- Keep only needed columns ---
    df = df[['text', 'label']].copy()

    # --- Drop empty texts ---
    df = df[df['text'].str.strip().str.len() > 10].reset_index(drop=True)

    # --- Drop NaN ---
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

    # --- Ensure label is int ---
    df['label'] = df['label'].astype(int)

    # --- Balance: equal real/fake ---
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    per_class = min(len(class_0), len(class_1), MAX_PER_DATASET // 2)

    balanced = pd.concat([
        class_0.sample(n=per_class, random_state=RANDOM_STATE),
        class_1.sample(n=per_class, random_state=RANDOM_STATE)
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    balanced['source_dataset'] = ds_path.name

    print(f"  ✅ {ds_path.name}: {len(balanced)} samples "
          f"(0: {per_class}, 1: {per_class})")

    return balanced


def create_splits(df: pd.DataFrame):

    # Train+Val vs Test
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df['label'],
        random_state=RANDOM_STATE
    )

    # Train vs Val
    adjusted_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val['label'],
        random_state=RANDOM_STATE
    )

    return train, val, test


def print_stats(name: str, df: pd.DataFrame):
    total = len(df)
    c0 = (df['label'] == 0).sum()
    c1 = (df['label'] == 1).sum()
    print(f"  {name:12s}: {total:6d} samples | "
          f"real(0): {c0} ({c0/total*100:.1f}%) | "
          f"fake(1): {c1} ({c1/total*100:.1f}%)")


def main():
    print("=" * 60)
    print("DATASET COMBINER")
    print("=" * 60)

    # --- Load & balance each dataset ---
    print("\n📂 Loading datasets...")
    all_dfs = []

    for ds_name, config in DATASET_CONFIGS.items():
        ds_path = BASE_DIR / ds_name
        if not ds_path.exists():
            print(f"  ⚠️  {ds_name} not found, skipping")
            continue
        df = load_and_balance_dataset(ds_path, config)
        if df is not None:
            all_dfs.append(df)

    # --- Combine ---
    final_df = pd.concat(all_dfs, ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"\n📊 Combined total: {len(final_df)} samples")
    print(f"   Label 0 (real): {(final_df['label']==0).sum()}")
    print(f"   Label 1 (fake): {(final_df['label']==1).sum()}")

    # --- Source distribution ---
    print("\n📊 Source distribution:")
    for src, cnt in final_df['source_dataset'].value_counts().items():
        print(f"   {src}: {cnt}")

    # --- Create splits ---
    print("\n✂️  Creating splits...")
    train, val, test = create_splits(final_df)

    print("\n📊 Split statistics:")
    print_stats("train", train)
    print_stats("validation", val)
    print_stats("test", test)

    # --- Save as HuggingFace DatasetDict ---
    print(f"\n💾 Saving to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val.reset_index(drop=True)),
        'test': Dataset.from_pandas(test.reset_index(drop=True)),
    })

    dataset_dict.save_to_disk(str(OUTPUT_DIR))

    print("\n✅ Done! Dataset saved successfully.")
    print("=" * 60)
    print(f"Train:      {len(train)} samples")
    print(f"Validation: {len(val)} samples")
    print(f"Test:       {len(test)} samples")
    print(f"Total:      {len(final_df)} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()