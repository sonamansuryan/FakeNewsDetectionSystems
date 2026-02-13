"""Custom training callbacks."""

import os
import json
from pathlib import Path
from typing import Dict, List
from transformers import TrainerCallback, TrainerState, TrainerControl
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PlottingCallback(TrainerCallback):
    """Callback to plot training metrics."""

    def __init__(self, output_dir: str = "reports/figures/training"):
        """
        Initialize plotting callback.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.eval_losses = []
        self.eval_metrics = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are available."""
        if logs is None:
            return

        # Store training loss
        if 'loss' in logs:
            self.train_losses.append({
                'step': state.global_step,
                'loss': logs['loss']
            })

        # Store evaluation metrics
        if 'eval_loss' in logs:
            self.eval_losses.append({
                'step': state.global_step,
                'loss': logs['eval_loss']
            })

            for key, value in logs.items():
                if key.startswith('eval_') and key != 'eval_loss':
                    metric_name = key.replace('eval_', '')
                    if metric_name not in self.eval_metrics:
                        self.eval_metrics[metric_name] = []
                    self.eval_metrics[metric_name].append({
                        'step': state.global_step,
                        'value': value
                    })

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info("Generating training plots...")
        self._plot_losses()
        self._plot_metrics()
        logger.info(f"Training plots saved to {self.output_dir}")

    def _plot_losses(self):
        """Plot training and evaluation losses."""
        if not self.train_losses and not self.eval_losses:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot training loss
        if self.train_losses:
            steps = [x['step'] for x in self.train_losses]
            losses = [x['loss'] for x in self.train_losses]
            ax.plot(steps, losses, label='Training Loss', linewidth=2)

        # Plot evaluation loss
        if self.eval_losses:
            steps = [x['step'] for x in self.eval_losses]
            losses = [x['loss'] for x in self.eval_losses]
            ax.plot(steps, losses, label='Validation Loss', linewidth=2, marker='o')

        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics(self):
        """Plot evaluation metrics."""
        if not self.eval_metrics:
            return

        num_metrics = len(self.eval_metrics)
        if num_metrics == 0:
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (metric_name, values) in enumerate(self.eval_metrics.items()):
            if idx >= 4:  # Only plot first 4 metrics
                break

            steps = [x['step'] for x in values]
            metric_values = [x['value'] for x in values]

            axes[idx].plot(steps, metric_values, linewidth=2, marker='o', markersize=6)
            axes[idx].set_xlabel('Training Steps', fontsize=11)
            axes[idx].set_ylabel(metric_name.capitalize(), fontsize=11)
            axes[idx].set_title(f'{metric_name.capitalize()} over Training',
                                fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(num_metrics, 4):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


class CheckpointCallback(TrainerCallback):
    """Callback to manage model checkpoints."""

    def __init__(self, save_best_only: bool = True,
                 metric_name: str = 'eval_f1',
                 mode: str = 'max'):
        """
        Initialize checkpoint callback.

        Args:
            save_best_only: Whether to save only the best model
            metric_name: Metric to monitor for best model
            mode: 'max' or 'min' for the metric
        """
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None or self.metric_name not in metrics:
            return

        current_metric = metrics[self.metric_name]

        is_better = (
                (self.mode == 'max' and current_metric > self.best_metric) or
                (self.mode == 'min' and current_metric < self.best_metric)
        )

        if is_better:
            self.best_metric = current_metric
            self.best_checkpoint = state.global_step
            logger.info(f"New best {self.metric_name}: {self.best_metric:.4f} "
                        f"at step {self.best_checkpoint}")


class LearningRateLoggerCallback(TrainerCallback):
    """Callback to log learning rate during training."""

    def __init__(self):
        """Initialize learning rate logger."""
        self.learning_rates = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log learning rate."""
        if logs is not None and 'learning_rate' in logs:
            self.learning_rates.append({
                'step': state.global_step,
                'lr': logs['learning_rate']
            })

    def on_train_end(self, args, state, control, **kwargs):
        """Save learning rate history."""
        if not self.learning_rates:
            return

        output_dir = Path(args.output_dir) / 'metrics'
        output_dir.mkdir(parents=True, exist_ok=True)

        lr_file = output_dir / 'learning_rate_history.json'
        with open(lr_file, 'w') as f:
            json.dump(self.learning_rates, f, indent=2)

        logger.info(f"Learning rate history saved to {lr_file}")


class ProgressCallback(TrainerCallback):
    """Callback to print progress updates."""

    def __init__(self, print_every: int = 100):
        """
        Initialize progress callback.

        Args:
            print_every: Print progress every N steps
        """
        self.print_every = print_every
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        import time
        self.start_time = time.time()
        logger.info(f"Training started with {state.max_steps} steps")

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.print_every == 0:
            import time
            elapsed = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0

            logger.info(
                f"Step {state.global_step}/{state.max_steps} | "
                f"Epoch {state.epoch:.2f} | "
                f"Speed: {steps_per_sec:.2f} steps/s"
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        import time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        logger.info(f"Training completed in {hours}h {minutes}m {seconds}s")


def create_callbacks(config: Dict, output_dir: str) -> List[TrainerCallback]:
    """
    Create training callbacks based on configuration.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for plots and logs

    Returns:
        List of callbacks
    """
    callbacks = []

    # Add plotting callback
    plotting_callback = PlottingCallback(output_dir=f"{output_dir}/plots")
    callbacks.append(plotting_callback)

    # Add checkpoint callback
    checkpoint_config = config.get('training', {}).get('checkpointing', {})
    checkpoint_callback = CheckpointCallback(
        save_best_only=checkpoint_config.get('save_best_only', True),
        metric_name=checkpoint_config.get('metric_for_best_model', 'eval_f1'),
        mode='max' if checkpoint_config.get('greater_is_better', True) else 'min'
    )
    callbacks.append(checkpoint_callback)

    # Add learning rate logger
    lr_callback = LearningRateLoggerCallback()
    callbacks.append(lr_callback)

    # Add progress callback
    logging_config = config.get('training', {}).get('logging', {})
    progress_callback = ProgressCallback(
        print_every=logging_config.get('steps', 100)
    )
    callbacks.append(progress_callback)

    return callbacks