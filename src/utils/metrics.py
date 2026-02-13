"""Custom metrics for model evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, List, Tuple, Optional
import torch


def compute_metrics(predictions: np.ndarray, labels: np.ndarray,
                   average: str = 'weighted') -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Model predictions
        labels: True labels
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary of metric values
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Get predicted classes
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions

    # Compute metrics
    accuracy = accuracy_score(labels, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_classes, average=average, zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    # Add per-class metrics if not using averaging
    if average is None:
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            metrics[f'precision_class_{i}'] = float(p)
            metrics[f'recall_class_{i}'] = float(r)
            metrics[f'f1_class_{i}'] = float(f)

    return metrics


def compute_confusion_matrix(predictions: np.ndarray,
                            labels: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Model predictions
        labels: True labels

    Returns:
        Confusion matrix
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    return confusion_matrix(labels, predictions)


def get_classification_report(predictions: np.ndarray,
                              labels: np.ndarray,
                              target_names: Optional[List[str]] = None) -> str:
    """
    Generate classification report.

    Args:
        predictions: Model predictions
        labels: True labels
        target_names: List of target class names

    Returns:
        Classification report as string
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    return classification_report(labels, predictions, target_names=target_names)


def compute_roc_metrics(predictions: np.ndarray,
                       labels: np.ndarray,
                       num_classes: int = 2) -> Dict[str, any]:
    """
    Compute ROC curve and AUC score.

    Args:
        predictions: Model prediction probabilities
        labels: True labels
        num_classes: Number of classes

    Returns:
        Dictionary containing ROC curve data and AUC scores
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    results = {}

    if num_classes == 2:
        # Binary classification
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            probs = predictions[:, 1]
        else:
            probs = predictions

        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        results['fpr'] = fpr
        results['tpr'] = tpr
        results['thresholds'] = thresholds
        results['auc'] = float(auc)
    else:
        # Multi-class classification
        from sklearn.preprocessing import label_binarize

        labels_bin = label_binarize(labels, classes=list(range(num_classes)))

        for i in range(num_classes):
            fpr, tpr, thresholds = roc_curve(labels_bin[:, i], predictions[:, i])
            auc = roc_auc_score(labels_bin[:, i], predictions[:, i])

            results[f'fpr_class_{i}'] = fpr
            results[f'tpr_class_{i}'] = tpr
            results[f'thresholds_class_{i}'] = thresholds
            results[f'auc_class_{i}'] = float(auc)

        # Compute micro-average
        fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), predictions.ravel())
        auc_micro = roc_auc_score(labels_bin, predictions, average='micro')

        results['fpr_micro'] = fpr_micro
        results['tpr_micro'] = tpr_micro
        results['auc_micro'] = float(auc_micro)

    return results


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = {
            'train': {},
            'val': {}
        }

    def update(self, metrics: Dict[str, float], split: str = 'train'):
        """
        Update metrics for a given split.

        Args:
            metrics: Dictionary of metric values
            split: Data split ('train' or 'val')
        """
        for key, value in metrics.items():
            if key not in self.metrics_history[split]:
                self.metrics_history[split][key] = []
            self.metrics_history[split][key].append(value)

    def get_best(self, metric: str, split: str = 'val',
                 mode: str = 'max') -> Tuple[float, int]:
        """
        Get best metric value and epoch.

        Args:
            metric: Metric name
            split: Data split
            mode: 'max' or 'min'

        Returns:
            Tuple of (best_value, best_epoch)
        """
        values = self.metrics_history[split].get(metric, [])
        if not values:
            return None, None

        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:
            best_value = min(values)
            best_epoch = values.index(best_value)

        return best_value, best_epoch

    def get_latest(self, metric: str, split: str = 'val') -> Optional[float]:
        """
        Get latest metric value.

        Args:
            metric: Metric name
            split: Data split

        Returns:
            Latest metric value
        """
        values = self.metrics_history[split].get(metric, [])
        return values[-1] if values else None

    def get_history(self, metric: str, split: str = 'val') -> List[float]:
        """
        Get complete history of a metric.

        Args:
            metric: Metric name
            split: Data split

        Returns:
            List of metric values
        """
        return self.metrics_history[split].get(metric, [])