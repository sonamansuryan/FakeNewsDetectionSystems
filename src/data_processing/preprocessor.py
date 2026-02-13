"""Text preprocessing utilities."""

import re
import yaml
from typing import Dict, List, Optional
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    """Preprocess text data for model training."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize text preprocessor.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra whitespace, etc.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def preprocess_dataset(self, dataset: Dataset,
                          clean: bool = True) -> Dataset:
        """
        Preprocess a dataset.

        Args:
            dataset: Input dataset
            clean: Whether to clean text

        Returns:
            Preprocessed dataset
        """
        logger.info("Preprocessing dataset...")

        def preprocess_function(examples):
            if clean and 'text' in examples:
                examples['text'] = [self.clean_text(text) for text in examples['text']]
            return examples

        processed = dataset.map(preprocess_function, batched=True)

        logger.info("Preprocessing complete")
        return processed

    def setup_tokenizer(self, model_name: str):
        """
        Set up tokenizer for the model.

        Args:
            model_name: Name of the pre-trained model
        """
        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_dataset(self, dataset: Dataset,
                        max_length: Optional[int] = None,
                        text_column: str = 'text') -> Dataset:
        """
        Tokenize a dataset.

        Args:
            dataset: Input dataset
            max_length: Maximum sequence length
            text_column: Name of the text column

        Returns:
            Tokenized dataset
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer() first.")

        if max_length is None:
            max_length = self.config.get('data_processing', {}).get('max_length', 512)

        logger.info(f"Tokenizing dataset with max_length={max_length}...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col not in ['label', 'labels']]
        )

        # Rename label column if needed
        if 'label' in tokenized.column_names and 'labels' not in tokenized.column_names:
            tokenized = tokenized.rename_column('label', 'labels')

        logger.info("Tokenization complete")
        return tokenized


def main():
    """Main function for testing."""
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