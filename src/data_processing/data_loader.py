"""Dataset loading utilities (multi-format version)."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
import pandas as pd
import json
import glob
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetLoader:
    """Load and manage multiple datasets from local files (CSV, TSV, JSONL, Arrow)."""

    def __init__(self, config_path: str = "configs/config.yaml", base_dir: str = "data/raw"):
        self.config_path = config_path
        self.config = self._load_config()
        self.base_dir = Path(base_dir)
        self.datasets = {}

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_liar_dataset(self) -> DatasetDict:
        """Load LIAR dataset from TSV files."""
        logger.info("Loading LIAR dataset from local TSV files...")
        dataset_dir = self.base_dir / "liar"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"LIAR dataset directory not found: {dataset_dir}")

        try:
            splits = {}
            column_names = [
                'id', 'label', 'statement', 'subjects', 'speaker',
                'speaker_job', 'state_info', 'party_affiliation',
                'barely_true_counts', 'false_counts', 'half_true_counts',
                'mostly_true_counts', 'pants_on_fire_counts', 'context'
            ]

            label_map = {
                'pants-fire': 1, 'false': 1, 'barely-true': 1,
                'half-true': 0, 'mostly-true': 0, 'true': 0
            }

            for split in ["train", "valid", "test"]:
                file_path = dataset_dir / f"{split}.tsv"
                if not file_path.exists():
                    continue

                df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
                df['label'] = df['label'].map(label_map).fillna(0).astype(int)
                df['text'] = df['statement']
                df['source'] = 'liar'
                df = df[['text', 'label', 'source', 'speaker', 'subjects', 'context']]

                split_name = 'validation' if split == 'valid' else split
                splits[split_name] = Dataset.from_pandas(df, preserve_index=False)

            return DatasetDict(splits)

        except Exception as e:
            logger.error(f"Error loading LIAR dataset: {str(e)}")
            raise

    def load_covid19_dataset(self) -> DatasetDict:
        """
        Load COVID-19 dataset from multiple CSV files.
        Detects 'Fake' or 'Real' in filenames to assign labels.
        """
        logger.info("Loading COVID-19 dataset from local CSV files...")

        # Handle directory naming (user has 'covid19', code might expect 'covid_19')
        dataset_dir = self.base_dir / "covid19"
        if not dataset_dir.exists():
            dataset_dir = self.base_dir / "covid_19"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"COVID-19 dataset directory not found at {dataset_dir}")

        try:
            all_dfs = []

            # Iterate over all CSV files in the directory
            csv_files = list(dataset_dir.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {dataset_dir}")

            logger.info(f"Found {len(csv_files)} CSV files for COVID-19. Processing...")

            for file_path in csv_files:
                filename = file_path.name.lower()

                # Determine label based on filename
                if "fake" in filename:
                    label = 1
                elif "real" in filename:
                    label = 0
                else:
                    # Skip files where label is ambiguous
                    continue

                try:
                    # Read CSV (handle potential encoding errors)
                    df = pd.read_csv(file_path, index_col=None)

                    # Identify text column
                    text_col = None
                    if 'title' in df.columns and 'content' in df.columns:
                        df['text'] = df['title'].fillna('') + " " + df['content'].fillna('')
                        text_col = 'text'
                    elif 'text' in df.columns:
                        text_col = 'text'
                    elif 'tweet' in df.columns:
                        text_col = 'tweet'
                    elif 'title' in df.columns:
                        text_col = 'title'

                    if text_col:
                        # Standardize DataFrame
                        temp_df = pd.DataFrame()
                        temp_df['text'] = df[text_col].astype(str)
                        temp_df['label'] = label
                        temp_df['source'] = 'covid_19'

                        # Add metadata if available
                        if 'publish_date' in df.columns:
                            temp_df['date'] = df['publish_date']

                        all_dfs.append(temp_df)
                except Exception as e:
                    logger.warning(f"Skipping file {filename}: {str(e)}")

            if not all_dfs:
                raise ValueError("Could not load any valid data from COVID-19 CSV files.")

            # Combine all dataframes
            full_df = pd.concat(all_dfs, ignore_index=True)

            # Remove duplicates and empty texts
            full_df = full_df.dropna(subset=['text'])
            full_df = full_df[full_df['text'].str.strip().astype(bool)]

            logger.info(f"Total loaded COVID-19 samples: {len(full_df)}")

            # Convert to HuggingFace Dataset
            full_dataset = Dataset.from_pandas(full_df, preserve_index=False)

            # Create splits (Train: 80%, Val: 10%, Test: 10%)
            splits = full_dataset.train_test_split(test_size=0.2, seed=42)
            test_val = splits['test'].train_test_split(test_size=0.5, seed=42)

            dataset_dict = DatasetDict({
                'train': splits['train'],
                'validation': test_val['train'],
                'test': test_val['test']
            })

            return dataset_dict

        except Exception as e:
            logger.error(f"Error loading COVID-19 dataset: {str(e)}")
            raise

    def load_fakenewsnet_dataset(self) -> DatasetDict:
        """Load FakeNewsNet dataset."""
        logger.info("Loading FakeNewsNet dataset...")
        dataset_dir = self.base_dir / "fakenewsnet"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"FakeNewsNet dir not found: {dataset_dir}")

        try:
            # Check if Arrow format exists (load fast)
            if (dataset_dir / "train").exists() and (dataset_dir / "dataset_dict.json").exists():
                logger.info("Loading FakeNewsNet from Arrow format...")
                dataset = load_from_disk(str(dataset_dir))
            else:
                # Load from CSV files
                logger.info("Loading FakeNewsNet from CSV files...")
                splits = {}
                for split_name in ["train", "validation", "test"]:
                    file_path = dataset_dir / f"{split_name}.csv"
                    if not file_path.exists(): continue

                    df = pd.read_csv(file_path)
                    if 'label' in df.columns:
                        df['label'] = df['label'].astype(int)
                    else:
                        df['label'] = 0

                    if 'text' not in df.columns:
                        if 'title' in df.columns:
                            df['text'] = df['title']
                        else:
                            continue  # Skip if no text

                    df['source'] = 'fakenewsnet'
                    splits[split_name] = Dataset.from_pandas(df, preserve_index=False)

                dataset = DatasetDict(splits)

            # Ensure consistency
            def process_fakenews(example):
                example['label'] = int(example['label']) if 'label' in example else 0
                if 'text' not in example:
                    example['text'] = example.get('title', example.get('content', ''))
                example['source'] = 'fakenewsnet'
                return example

            dataset = dataset.map(process_fakenews)
            return dataset

        except Exception as e:
            logger.error(f"Error loading FakeNewsNet dataset: {str(e)}")
            raise

    def load_fever_dataset(self) -> DatasetDict:
        """
        Load FEVER dataset from JSONL files using robust line-by-line reading.
        Avoids pandas read_json error with nested mixed types.
        """
        logger.info("Loading FEVER dataset from local JSONL files...")
        dataset_dir = self.base_dir / "fever_data"  # User provided this path in prompt

        if not dataset_dir.exists():
            # Try fallback
            dataset_dir = self.base_dir / "fever"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"FEVER dataset directory not found.")

        try:
            splits = {}

            # Map filenames to splits
            file_map = {
                "train": ["fever_train.jsonl", "train.jsonl"],
                "validation": ["fever_dev.jsonl", "dev.jsonl", "valid.jsonl"],
                "test": ["fever_test.jsonl", "test.jsonl"]
            }

            label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}

            for split_name, possible_filenames in file_map.items():
                file_path = None
                for fname in possible_filenames:
                    if (dataset_dir / fname).exists():
                        file_path = dataset_dir / fname
                        break

                if not file_path:
                    logger.warning(f"FEVER {split_name} file not found.")
                    continue

                logger.info(f"Reading {split_name} from {file_path.name}...")

                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)

                            # Skip if label is NOT ENOUGH INFO (we want binary)
                            # Or if label is missing
                            lbl_str = obj.get('label', None)
                            if lbl_str == 'NOT ENOUGH INFO' or lbl_str is None:
                                continue

                            label = label_map.get(lbl_str, 2)
                            if label == 2: continue  # Double check

                            claim = obj.get('claim', obj.get('text', ''))
                            if not claim: continue

                            # We intentionally ignore 'evidence' to avoid type errors
                            data.append({
                                'text': claim,
                                'label': label,
                                'source': 'fever'
                            })
                        except json.JSONDecodeError:
                            continue

                if not data:
                    logger.warning(f"No valid binary samples found in {split_name}")
                    continue

                # Create DataFrame and then Dataset
                df = pd.DataFrame(data)
                splits[split_name] = Dataset.from_pandas(df, preserve_index=False)

            if not splits:
                raise ValueError("No valid splits found for FEVER dataset")

            return DatasetDict(splits)

        except Exception as e:
            logger.error(f"Error loading FEVER dataset: {str(e)}")
            raise

    def load_all_datasets(self, dataset_names: Optional[List[str]] = None) -> Dict[str, DatasetDict]:
        """Load all configured datasets."""
        if dataset_names is None:
            dataset_names = ['liar', 'covid_19', 'fakenewsnet', 'fever']

        logger.info(f"Loading datasets: {', '.join(dataset_names)}...")
        datasets = {}

        for name in dataset_names:
            try:
                if name.lower() == 'liar':
                    datasets['liar'] = self.load_liar_dataset()
                elif name.lower() in ['covid_19', 'covid19']:
                    datasets['covid_19'] = self.load_covid19_dataset()
                elif name.lower() == 'fakenewsnet':
                    datasets['fakenewsnet'] = self.load_fakenewsnet_dataset()
                elif name.lower() == 'fever':
                    datasets['fever'] = self.load_fever_dataset()
            except Exception as e:
                logger.error(f"Failed to load {name}: {str(e)}")

        self.datasets = datasets
        return datasets

    def save_datasets(self, output_dir: str = None):
        """Save datasets to disk."""
        if output_dir is None: output_dir = "data/processed"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, dataset in self.datasets.items():
            dataset_path = output_path / name
            if dataset_path.exists():
                import shutil
                shutil.rmtree(dataset_path)
            dataset.save_to_disk(str(dataset_path))
            logger.info(f"Saved {name} to {dataset_path}")

        logger.info("All datasets saved successfully")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Load datasets for fine-tuning from local files")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--base-dir', type=str, default='data/raw',
                        help='Base directory containing dataset folders')
    parser.add_argument('--datasets', nargs='+',
                        default=['liar', 'covid_19', 'fakenewsnet', 'fever'],
                        help='Datasets to load')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Output directory for datasets')
    parser.add_argument('--save', action='store_true',
                        help='Save datasets after loading')

    args = parser.parse_args()

    # Create loader
    loader = DatasetLoader(config_path=args.config, base_dir=args.base_dir)

    # Load datasets
    datasets = loader.load_all_datasets(dataset_names=args.datasets)

    # Save datasets if requested
    if args.save:
        loader.save_datasets(output_dir=args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET LOADING SUMMARY")
    print("=" * 80)
    for name, dataset in datasets.items():
        print(f"\n{name.upper()}:")
        for split, data in dataset.items():
            print(f"  {split}: {len(data)} samples")
    print("=" * 80)


if __name__ == "__main__":
    main()