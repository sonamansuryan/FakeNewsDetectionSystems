import re
import yaml
from typing import Dict, List, Optional
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:

    def __init__(self, config_path: str = "configs/config.yaml"):

        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text, flags=re.MULTILINE)

        text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)

        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def preprocess_dataset(self, dataset: Dataset,
                           clean: bool = True) -> Dataset:

        logger.info("Preprocessing dataset...")

        def preprocess_function(examples):
            if clean and 'text' in examples:
                examples['text'] = [self.clean_text(text) for text in examples['text']]
            return examples

        processed = dataset.map(preprocess_function, batched=True)

        logger.info("Preprocessing complete")
        return processed

    def setup_tokenizer(self, model_name: str):

        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_dataset(self, dataset: Dataset,
                         max_length: Optional[int] = None,
                         text_column: str = 'text') -> Dataset:

        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")

        if max_length is None:
            max_length = self.config.get('data_processing', {}).get('max_length', 512)

        possible_label_cols = ['label', 'labels', 'binary_label', 'target', 'class']
        label_col = next((col for col in possible_label_cols if col in dataset.column_names), None)

        if label_col is None:
            raise ValueError(f"Label column not found. Available: {dataset.column_names}")

        if label_col != 'labels':
            dataset = dataset.rename_column(label_col, 'labels')

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
            tokenized['labels'] = examples['labels']
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        logger.info("Tokenization complete. Columns: " + str(tokenized_dataset.column_names))
        return tokenized_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess text data")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='roberta-base',
                        help='Model name for tokenizer')

    args = parser.parse_args()

    preprocessor = TextPreprocessor(config_path=args.config)
    preprocessor.setup_tokenizer(args.model)

    # Test with sample text
    sample_text = "This is a test! http://example.com Check it out."
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")


if __name__ == "__main__":
    main()