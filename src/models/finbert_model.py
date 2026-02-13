"""FinBERT model wrapper."""

import torch
import yaml
from typing import Dict, Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FinBERTModel:
    """Wrapper for FinBERT model for sequence classification."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize FinBERT model.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_device(self) -> str:
        """Get the device to use for training."""
        device_config = self.config.get('hardware', {}).get('device', 'cuda')

        if device_config == 'cuda' and torch.cuda.is_available():
            device = 'cuda'
        elif device_config == 'mps' and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        logger.info(f"Using device: {device}")
        return device

    def load_model(self, num_labels: Optional[int] = None,
                   pretrained: bool = True):
        """
        Load FinBERT model.

        Args:
            num_labels: Number of output labels
            pretrained: Whether to load pretrained weights
        """
        model_config = self.config.get('models', {}).get('finbert', {})
        model_name = model_config.get('name', 'ProsusAI/finbert')

        if num_labels is None:
            num_labels = model_config.get('num_labels', 2)

        logger.info(f"Loading FinBERT model: {model_name}")
        logger.info(f"Number of labels: {num_labels}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_config(config)

        # Move model to device
        self.model.to(self.device)

        logger.info("FinBERT model loaded successfully")

    def get_model(self):
        """Get the model instance."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer instance."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer

    def save_model(self, output_dir: str):
        """
        Save model and tokenizer.

        Args:
            output_dir: Output directory
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving FinBERT model to {output_dir}")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info("Model saved successfully")

    def load_from_checkpoint(self, checkpoint_dir: str):
        """
        Load model from a checkpoint.

        Args:
            checkpoint_dir: Checkpoint directory
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)

        logger.info("Model loaded from checkpoint successfully")

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get the number of model parameters.

        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

    def freeze_base_model(self):
        """Freeze the base model parameters (for transfer learning)."""
        if self.model is None:
            raise ValueError("Model not loaded")

        logger.info("Freezing base model parameters...")

        # Freeze all parameters except the classifier
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

        params = self.get_num_parameters()
        logger.info(f"Trainable parameters: {params['trainable']:,}")

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        if self.model is None:
            raise ValueError("Model not loaded")

        logger.info("Unfreezing all parameters...")

        for param in self.model.parameters():
            param.requires_grad = True

        params = self.get_num_parameters()
        logger.info(f"Trainable parameters: {params['trainable']:,}")


def main():
    """Main function for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Test FinBERT model")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Create model
    model_wrapper = FinBERTModel(config_path=args.config)

    # Load model
    model_wrapper.load_model(num_labels=2)

    # Print parameter counts
    params = model_wrapper.get_num_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")


if __name__ == "__main__":
    main()