"""Model training pipeline."""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datasets import load_from_disk, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from datetime import datetime
import json

from src.models.finbert_model import FinBERTModel
from src.models.roberta_model import RoBERTaModel
from src.data_processing.preprocessor import TextPreprocessor
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics

logger = setup_logger(__name__)


class MetricsCallback(TrainerCallback):
    """Callback to log metrics during training."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.training_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Save logs to history."""
        if logs:
            self.training_history.append({
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            })

    def get_history(self):
        """Get training history."""
        return self.training_history


class ModelTrainer:
    """Train language models for misinformation detection."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize model trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model_wrapper = None
        self.preprocessor = None
        self.trainer = None
        self.training_history = []

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def setup_model(self, model_type: str = 'finbert', num_labels: int = 2):
        """
        Set up the model for training.

        Args:
            model_type: Type of model ('finbert' or 'roberta')
            num_labels: Number of output labels
        """
        logger.info(f"Setting up {model_type} model...")

        if model_type.lower() == 'finbert':
            self.model_wrapper = FinBERTModel(config_path=self.config_path)
        elif model_type.lower() == 'roberta':
            self.model_wrapper = RoBERTaModel(config_path=self.config_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_wrapper.load_model(num_labels=num_labels)

        # Print model info
        params = self.model_wrapper.get_num_parameters()
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {params['total']:,}")
        logger.info(f"  Trainable: {params['trainable']:,}")

    def setup_preprocessor(self):
        """Set up text preprocessor."""
        logger.info("Setting up preprocessor...")

        self.preprocessor = TextPreprocessor(config_path=self.config_path)
        self.preprocessor.tokenizer = self.model_wrapper.get_tokenizer()

    def load_dataset(self, dataset_path: str) -> DatasetDict:
        """
        Load dataset from disk.

        Args:
            dataset_path: Path to dataset

        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset from {dataset_path}...")

        dataset = load_from_disk(dataset_path)

        logger.info(f"Dataset loaded:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} samples")

        return dataset

    def prepare_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare dataset for training (preprocessing and tokenization).

        Args:
            dataset: Input dataset

        Returns:
            Prepared dataset
        """
        logger.info("Preparing dataset...")

        # Preprocess text
        processed_dataset = {}
        for split, data in dataset.items():
            processed_dataset[split] = self.preprocessor.preprocess_dataset(data, clean=True)

        # Tokenize
        tokenized_dataset = {}
        max_length = self.config.get('data_processing', {}).get('max_length', 512)

        for split, data in processed_dataset.items():
            tokenized_dataset[split] = self.preprocessor.tokenize_dataset(
                data,
                max_length=max_length
            )

        logger.info("Dataset preparation complete")
        return DatasetDict(tokenized_dataset)

    def get_training_args(self, output_dir: str) -> TrainingArguments:
        """
        Get training arguments from config.

        Args:
            output_dir: Output directory for checkpoints

        Returns:
            TrainingArguments object
        """
        train_config = self.config.get('training', {})

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get logging configuration
        logging_config = train_config.get('logging', {})
        report_to = logging_config.get('report_to', ['tensorboard'])

        # Get checkpointing configuration
        checkpoint_config = train_config.get('checkpointing', {})

        training_args = TrainingArguments(
            output_dir=str(output_path),

            # Training parameters
            num_train_epochs=train_config.get('num_epochs', 3),
            per_device_train_batch_size=train_config.get('batch_size', 16),
            per_device_eval_batch_size=train_config.get('batch_size', 16),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),

            # Optimization
            learning_rate=train_config.get('learning_rate', 2e-5),
            weight_decay=train_config.get('weight_decay', 0.01),
            warmup_ratio=train_config.get('warmup_ratio', 0.1),
            max_grad_norm=train_config.get('max_grad_norm', 1.0),

            # Learning rate scheduler
            lr_scheduler_type=train_config.get('lr_scheduler_type', 'linear'),

            # Evaluation
            eval_strategy=train_config.get('evaluation', {}).get('strategy', 'steps'),
            eval_steps=train_config.get('evaluation', {}).get('steps', 500),

            # Logging
            logging_steps=logging_config.get('steps', 100),
            logging_strategy='steps',
            report_to=report_to,

            # Checkpointing
            save_strategy='steps',
            save_steps=checkpoint_config.get('save_steps', 500),
            save_total_limit=checkpoint_config.get('save_total_limit', 3),

            # Metrics
            load_best_model_at_end=checkpoint_config.get('save_best_only', True),
            metric_for_best_model=checkpoint_config.get('metric_for_best_model', 'eval_f1'),
            greater_is_better=checkpoint_config.get('greater_is_better', True),

            # Mixed precision
            fp16=train_config.get('fp16', False),

            # Other settings
            seed=train_config.get('seed', 42),
            data_seed=train_config.get('seed', 42),
            remove_unused_columns=True,
            push_to_hub=False,
        )

        return training_args

    def compute_metrics_wrapper(self, eval_pred):
        """
        Wrapper for compute_metrics function.

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred

        # Get predicted classes
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        # Compute metrics
        average = self.config.get('evaluation', {}).get('average', 'weighted')
        metrics = compute_metrics(predictions, labels, average=average)

        return metrics

    def train(self, train_dataset, eval_dataset, output_dir: str):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory for model and checkpoints
        """
        logger.info("Starting training...")

        # Get training arguments
        training_args = self.get_training_args(output_dir)

        # Create callbacks
        callbacks = []

        # Early stopping
        if self.config.get('training', {}).get('early_stopping', {}).get('enabled', True):
            early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_config.get('patience', 3),
                    early_stopping_threshold=early_stopping_config.get('min_delta', 0.001)
                )
            )

        # Create trainer
        self.trainer = Trainer(
            model=self.model_wrapper.get_model(),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics_wrapper,
            callbacks=callbacks
        )

        # Add metrics callback
        metrics_callback = MetricsCallback(self.trainer)
        self.trainer.add_callback(metrics_callback)

        # Train
        logger.info("=" * 80)
        logger.info("TRAINING STARTED")
        logger.info("=" * 80)

        train_result = self.trainer.train()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)

        # Save the final model
        self.trainer.save_model(output_dir)
        self.model_wrapper.get_tokenizer().save_pretrained(output_dir)

        # Save metrics
        self.training_history = metrics_callback.get_history()
        self._save_training_metrics(train_result, output_dir)

        logger.info(f"Model saved to {output_dir}")

        return train_result

    def evaluate(self, eval_dataset, output_dir: Optional[str] = None):
        """
        Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset
            output_dir: Output directory for results

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info("Evaluating model...")

        eval_results = self.trainer.evaluate(eval_dataset)

        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")

        if output_dir:
            self._save_evaluation_results(eval_results, output_dir)

        return eval_results

    def _save_training_metrics(self, train_result, output_dir: str):
        """Save training metrics to file."""
        metrics_dir = Path(output_dir) / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save training result
        metrics_file = metrics_dir / 'train_results.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'train_runtime': train_result.metrics.get('train_runtime', 0),
                'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
                'train_steps_per_second': train_result.metrics.get('train_steps_per_second', 0),
                'total_flos': train_result.metrics.get('total_flos', 0),
                'train_loss': train_result.metrics.get('train_loss', 0)
            }, f, indent=2)

        logger.info(f"Training metrics saved to {metrics_file}")

        # Save training history
        history_file = metrics_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Training history saved to {history_file}")

    def _save_evaluation_results(self, eval_results: Dict, output_dir: str):
        """Save evaluation results to file."""
        metrics_dir = Path(output_dir) / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)

        eval_file = metrics_dir / 'eval_results.json'
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Evaluation results saved to {eval_file}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Train language models")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['finbert', 'roberta'],
                        help='Model type to train')
    parser.add_argument('--dataset', type=str, default='data/combined/combined_dataset',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: models/{model_name})')
    parser.add_argument('--num-labels', type=int, default=2,
                        help='Number of output labels')

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'models/{args.model}/{timestamp}'

    # Create trainer
    trainer = ModelTrainer(config_path=args.config)

    # Setup model
    trainer.setup_model(model_type=args.model, num_labels=args.num_labels)

    # Setup preprocessor
    trainer.setup_preprocessor()

    # Load dataset
    dataset = trainer.load_dataset(args.dataset)

    # Prepare dataset
    prepared_dataset = trainer.prepare_dataset(dataset)

    # Train model
    train_result = trainer.train(
        train_dataset=prepared_dataset['train'],
        eval_dataset=prepared_dataset['validation'],
        output_dir=args.output_dir
    )

    # Evaluate on test set
    test_results = trainer.evaluate(
        eval_dataset=prepared_dataset['test'],
        output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()