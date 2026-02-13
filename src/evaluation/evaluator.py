"""Model evaluation utilities."""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from sklearn.metrics import confusion_matrix, classification_report
import json

from src.utils.logger import setup_logger
from src.utils.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_roc_metrics
)

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluate trained models."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize model evaluator.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.evaluation_results = {}

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_device(self) -> str:
        """Get the device to use."""
        device_config = self.config.get('hardware', {}).get('device', 'cuda')

        if device_config == 'cuda' and torch.cuda.is_available():
            device = 'cuda'
        elif device_config == 'mps' and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        logger.info(f"Using device: {device}")
        return device

    def load_model(self, model_path: str):
        """
        Load a trained model.

        Args:
            model_path: Path to the model checkpoint
        """
        logger.info(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def load_dataset(self, dataset_path: str, split: str = 'test') -> Dataset:
        """
        Load dataset for evaluation.

        Args:
            dataset_path: Path to dataset
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset from {dataset_path} ({split} split)...")

        dataset_dict = load_from_disk(dataset_path)

        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in dataset")

        dataset = dataset_dict[split]
        logger.info(f"Loaded {len(dataset)} samples")

        return dataset

    def predict(self, texts: List[str]) -> Dict:
        """
        Make predictions on a list of texts.

        Args:
            texts: List of input texts

        Returns:
            Dictionary containing predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }

    def evaluate_dataset(self, dataset: Dataset) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Dataset to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on dataset...")

        # Get texts and labels
        df = dataset.to_pandas()
        texts = df['text'].tolist()
        true_labels = df['label'].values

        # Make predictions in batches
        batch_size = 32
        all_predictions = []
        all_probabilities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            results = self.predict(batch_texts)
            all_predictions.extend(results['predictions'])
            all_probabilities.extend(results['probabilities'])

        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)

        # Compute metrics
        metrics = compute_metrics(predictions, true_labels, average='weighted')

        # Add confusion matrix
        cm = compute_confusion_matrix(predictions, true_labels)
        metrics['confusion_matrix'] = cm.tolist()

        # Add classification report
        report = get_classification_report(predictions, true_labels)
        metrics['classification_report'] = report

        # Add ROC metrics for binary classification
        if probabilities.shape[1] == 2:
            roc_metrics = compute_roc_metrics(probabilities, true_labels, num_classes=2)
            metrics.update(roc_metrics)

        self.evaluation_results = {
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }

        logger.info("Evaluation complete")
        self._print_evaluation_summary()

        return metrics

    def _print_evaluation_summary(self):
        """Print evaluation summary."""
        metrics = self.evaluation_results['metrics']

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        if 'auc' in metrics:
            print(f"AUC Score: {metrics['auc']:.4f}")

        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("=" * 80 + "\n")

    def plot_confusion_matrix(self, output_dir: str = "reports/figures/evaluation",
                              class_names: Optional[List[str]] = None):
        """
        Plot confusion matrix.

        Args:
            output_dir: Output directory for plot
            class_names: Names of classes
        """
        if 'confusion_matrix' not in self.evaluation_results['metrics']:
            logger.warning("No confusion matrix available")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cm = np.array(self.evaluation_results['metrics']['confusion_matrix'])

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix plot saved to {output_path}")

    def plot_roc_curve(self, output_dir: str = "reports/figures/evaluation"):
        """
        Plot ROC curve.

        Args:
            output_dir: Output directory for plot
        """
        metrics = self.evaluation_results['metrics']

        if 'fpr' not in metrics or 'tpr' not in metrics:
            logger.warning("No ROC data available")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve
        ax.plot(metrics['fpr'], metrics['tpr'], linewidth=2,
                label=f"ROC Curve (AUC = {metrics['auc']:.4f})")

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"ROC curve plot saved to {output_path}")

    def plot_prediction_distribution(self, output_dir: str = "reports/figures/evaluation"):
        """
        Plot distribution of prediction probabilities.

        Args:
            output_dir: Output directory for plot
        """
        probabilities = self.evaluation_results['probabilities']
        true_labels = self.evaluation_results['true_labels']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot for each class
        for class_idx in range(probabilities.shape[1]):
            class_probs = probabilities[:, class_idx]

            # Separate by true label
            for true_label in np.unique(true_labels):
                mask = true_labels == true_label
                axes[class_idx].hist(class_probs[mask], bins=30, alpha=0.5,
                                     label=f'True Label {true_label}', edgecolor='black')

            axes[class_idx].set_xlabel('Probability', fontsize=12)
            axes[class_idx].set_ylabel('Frequency', fontsize=12)
            axes[class_idx].set_title(f'Prediction Probability Distribution - Class {class_idx}',
                                      fontsize=12, fontweight='bold')
            axes[class_idx].legend()
            axes[class_idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Prediction distribution plot saved to {output_path}")

    def save_results(self, output_dir: str = "reports/evaluation"):
        """
        Save evaluation results to files.

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving evaluation results to {output_dir}...")

        # Save metrics
        metrics = self.evaluation_results['metrics'].copy()

        # Remove non-JSON-serializable items
        if 'fpr' in metrics:
            metrics['fpr'] = metrics['fpr'].tolist()
        if 'tpr' in metrics:
            metrics['tpr'] = metrics['tpr'].tolist()
        if 'thresholds' in metrics:
            metrics['thresholds'] = metrics['thresholds'].tolist()

        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': self.evaluation_results['true_labels'],
            'predicted_label': self.evaluation_results['predictions'],
            'probability_class_0': self.evaluation_results['probabilities'][:, 0],
            'probability_class_1': self.evaluation_results['probabilities'][:, 1]
        })

        predictions_file = output_path / 'predictions.csv'
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='data/combined/combined_dataset',
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output-dir', type=str, default='reports/evaluation',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(config_path=args.config)

    # Load model
    evaluator.load_model(args.model_path)

    # Load dataset
    dataset = evaluator.load_dataset(args.dataset, split=args.split)

    # Evaluate
    metrics = evaluator.evaluate_dataset(dataset)

    # Create visualizations
    evaluator.plot_confusion_matrix(output_dir=f"{args.output_dir}/figures")
    evaluator.plot_roc_curve(output_dir=f"{args.output_dir}/figures")
    evaluator.plot_prediction_distribution(output_dir=f"{args.output_dir}/figures")

    # Save results
    evaluator.save_results(output_dir=args.output_dir)

    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()