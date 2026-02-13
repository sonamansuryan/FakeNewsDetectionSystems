"""Dataset combination strategies."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk, ClassLabel, Features, Value
from sklearn.model_selection import train_test_split
import json

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetCombiner:
    """Combine multiple datasets using different strategies."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.datasets = {}
        self.combined_dataset = None

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_datasets(self, datasets: Optional[Dict[str, DatasetDict]] = None,
                     base_dir: str = 'data/processed'):
        """
        Load datasets specifically from the processed Arrow files.
        """
        if datasets is not None:
            self.datasets = datasets
            return

        logger.info(f"Loading processed datasets from {base_dir}...")
        self.datasets = {}
        base_path = Path(base_dir)

        if not base_path.exists():
            raise FileNotFoundError(f"Processed data directory not found: {base_dir}")

        # Iterate over directories in data/processed
        for dataset_path in base_path.iterdir():
            if dataset_path.is_dir():
                try:
                    is_dataset = False
                    if (dataset_path / "dataset_dict.json").exists():
                        is_dataset = True
                    elif (dataset_path / "dataset_info.json").exists():
                        is_dataset = True
                    elif (dataset_path / "state.json").exists():
                        is_dataset = True
                    elif list(dataset_path.rglob("*.arrow")):
                        is_dataset = True

                    if is_dataset:
                        logger.info(f"Loading {dataset_path.name}...")
                        ds = load_from_disk(str(dataset_path))
                        self.datasets[dataset_path.name] = ds
                    else:
                        logger.debug(f"Skipping {dataset_path.name}: not identified as a dataset")

                except Exception as e:
                    logger.warning(f"Could not load {dataset_path.name}: {e}")

        if not self.datasets:
            found_dirs = [d.name for d in base_path.iterdir() if d.is_dir()]
            raise ValueError(f"No valid datasets found in {base_dir}. Found directories: {found_dirs}")

        logger.info(f"Successfully loaded {len(self.datasets)} datasets: {list(self.datasets.keys())}")

    def harmonize_labels(self, datasets: Dict[str, DatasetDict]) -> Dict[str, DatasetDict]:
        """Harmonize labels across datasets to ensure consistency."""
        logger.info("Harmonizing labels across datasets...")
        harmonized_datasets = {}

        for dataset_name, dataset_dict in datasets.items():
            harmonized_dict = {}
            for split, dataset in dataset_dict.items():
                df = dataset.to_pandas()

                if 'label' in df.columns:
                    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
                    df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)
                else:
                    logger.warning(f"Dataset {dataset_name} split {split} has no 'label' column. Assigning 0.")
                    df['label'] = 0

                if 'source' not in df.columns:
                    df['source'] = dataset_name

                # Standardize columns
                keep_cols = ['text', 'label', 'source']
                for col in ['title', 'date']:
                    if col in df.columns: keep_cols.append(col)

                existing_cols = [c for c in keep_cols if c in df.columns]
                df = df[existing_cols]

                harmonized_dict[split] = Dataset.from_pandas(df, preserve_index=False)

            harmonized_datasets[dataset_name] = DatasetDict(harmonized_dict)

        return harmonized_datasets

    def combine_concatenate(self, split: str = 'train') -> Dataset:
        """Simple concatenation of all datasets."""
        logger.info(f"Combining datasets using concatenation strategy ({split} split)...")
        datasets_to_combine = []

        for name, dataset_dict in self.datasets.items():
            if split in dataset_dict:
                datasets_to_combine.append(dataset_dict[split])
            else:
                logger.warning(f"Split '{split}' not found in {name}")

        if not datasets_to_combine:
            raise ValueError(f"No datasets found for split: {split}")

        combined = concatenate_datasets(datasets_to_combine)
        logger.info(f"Combined dataset size: {len(combined)} samples")
        return combined

    def combine_balanced(self, split: str = 'train',
                        samples_per_dataset: Optional[int] = None) -> Dataset:
        """Balanced sampling from each dataset."""
        logger.info(f"Combining datasets using balanced sampling strategy ({split} split)...")

        if samples_per_dataset is None:
            dataset_sizes = []
            for name, dataset_dict in self.datasets.items():
                if split in dataset_dict:
                    size = len(dataset_dict[split])
                    dataset_sizes.append(size)
                    logger.info(f"  {name}: {size} samples")

            if not dataset_sizes:
                raise ValueError(f"No valid datasets found for split '{split}'")

            samples_per_dataset = min(dataset_sizes)
            logger.info(f"Minimum dataset size is {samples_per_dataset}. Using this as balance limit.")

        logger.info(f"Sampling {samples_per_dataset} samples from each dataset...")
        balanced_datasets = []

        for name, dataset_dict in self.datasets.items():
            if split in dataset_dict:
                dataset = dataset_dict[split]
                dataset = dataset.shuffle(seed=42)

                if len(dataset) < samples_per_dataset:
                    indices = np.random.choice(len(dataset), samples_per_dataset, replace=True)
                    sampled_dataset = dataset.select(indices)
                else:
                    sampled_dataset = dataset.select(range(samples_per_dataset))

                balanced_datasets.append(sampled_dataset)

        combined = concatenate_datasets(balanced_datasets)
        logger.info(f"Combined balanced dataset size: {len(combined)} samples")
        return combined

    def combine_stratified(self, split: str = 'train') -> Dataset:
        """Stratified combination ensuring class balance within each dataset."""
        logger.info(f"Combining datasets using stratified strategy ({split} split)...")
        stratified_datasets = []

        for name, dataset_dict in self.datasets.items():
            if split in dataset_dict:
                dataset = dataset_dict[split]
                df = dataset.to_pandas()

                if 'label' not in df.columns:
                    stratified_datasets.append(dataset)
                    continue

                class_counts = df['label'].value_counts()
                min_class_count = class_counts.min()

                logger.info(f"  {name} min class count: {min_class_count}")

                stratified_dfs = []
                for label in class_counts.index:
                    class_df = df[df['label'] == label]
                    if len(class_df) > min_class_count:
                        sampled_df = class_df.sample(n=min_class_count, random_state=42)
                    else:
                        sampled_df = class_df
                    stratified_dfs.append(sampled_df)

                stratified_df = pd.concat(stratified_dfs, ignore_index=True)
                stratified_dataset = Dataset.from_pandas(stratified_df, preserve_index=False)
                stratified_datasets.append(stratified_dataset)

        combined = concatenate_datasets(stratified_datasets)
        logger.info(f"Combined stratified dataset size: {len(combined)} samples")
        return combined

    def create_splits(self, dataset: Dataset,
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> DatasetDict:
        """Create train/validation/test splits."""
        logger.info("Creating train/validation/test splits...")

        # FIX: Cast 'label' column to ClassLabel to allow stratification
        # This tells HuggingFace that this column contains categories (Real/Fake)
        try:
            logger.info("Casting 'label' column to ClassLabel...")
            dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["real", "fake"]))
        except Exception as e:
            logger.warning(f"Could not cast label to ClassLabel: {e}. Stratification might fail if column is not integer.")

        # First split: Train+Val vs Test
        splits_1 = dataset.train_test_split(test_size=test_size, seed=42, stratify_by_column="label")
        test_dataset = splits_1['test']
        train_val_dataset = splits_1['train']

        # Adjust validation size relative to the remaining set
        adjusted_val_size = val_size / (1 - test_size)

        # Second split: Train vs Val
        splits_2 = train_val_dataset.train_test_split(test_size=adjusted_val_size, seed=42, stratify_by_column="label")
        val_dataset = splits_2['test']
        train_dataset = splits_2['train']

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        logger.info(f"Split sizes - Train: {len(dataset_dict['train'])}, "
                   f"Validation: {len(dataset_dict['validation'])}, "
                   f"Test: {len(dataset_dict['test'])}")

        return dataset_dict

    def combine_datasets(self, strategy: str = 'balanced',
                        create_splits: bool = True) -> DatasetDict:
        """Combine datasets using specified strategy."""
        logger.info(f"Combining datasets using '{strategy}' strategy...")

        harmonized_datasets = self.harmonize_labels(self.datasets)
        self.datasets = harmonized_datasets

        if strategy == 'concatenate':
            combined = self.combine_concatenate(split='train')
        elif strategy == 'balanced':
            combined = self.combine_balanced(split='train')
        elif strategy == 'stratified':
            combined = self.combine_stratified(split='train')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        combined = combined.shuffle(seed=42)

        if create_splits:
            dataset_dict = self.create_splits(combined)
            self.combined_dataset = dataset_dict
        else:
            self.combined_dataset = DatasetDict({'train': combined})

        self._print_combination_stats()
        return self.combined_dataset

    def _print_combination_stats(self):
        """Print statistics about the combined dataset."""
        print("\n" + "="*80)
        print("COMBINED DATASET STATISTICS")
        print("="*80)

        for split, dataset in self.combined_dataset.items():
            print(f"\n{split.upper()} Split: {len(dataset)} samples")

            sample_size = min(len(dataset), 10000)
            if sample_size > 0:
                df = dataset.select(range(sample_size)).to_pandas()

                if 'label' in df.columns:
                    label_counts = df['label'].value_counts().sort_index()
                    print(f"  Label distribution (sampled):")
                    for label, count in label_counts.items():
                        print(f"    Label {label}: {count}")

                if 'source' in df.columns:
                    source_counts = df['source'].value_counts()
                    print(f"  Source distribution (sampled):")
                    for source, count in source_counts.items():
                        print(f"    {source}: {count}")
        print("="*80 + "\n")

    def save_combined_dataset(self, output_dir: str = "data/combined",
                             save_format: str = 'arrow'):
        """Save combined dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving combined dataset to {output_dir}...")

        if save_format == 'arrow':
            self.combined_dataset.save_to_disk(str(output_path / 'combined_dataset'))
            logger.info(f"Saved in Arrow format")

        self._save_metadata(output_path)

    def _save_metadata(self, output_path: Path):
        """Save metadata about the combined dataset."""
        metadata = {
            'source_datasets': list(self.datasets.keys()),
            'splits': {}
        }
        for split, dataset in self.combined_dataset.items():
            metadata['splits'][split] = {'num_samples': len(dataset)}

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--strategy', type=str, default='balanced')
    parser.add_argument('--output-dir', type=str, default='data/combined')
    parser.add_argument('--save-format', type=str, default='arrow')
    args = parser.parse_args()

    combiner = DatasetCombiner(config_path=args.config)
    combiner.load_datasets()
    combiner.combine_datasets(strategy=args.strategy)
    combiner.save_combined_dataset(output_dir=args.output_dir, save_format=args.save_format)

    print(f"\nDataset combination complete! Strategy: {args.strategy}")

if __name__ == "__main__":
    main()