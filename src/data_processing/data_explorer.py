"""Dataset exploration and analysis utilities."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import Dataset, DatasetDict
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DatasetLoader
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetExplorer:
    """Explore and analyze datasets."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize dataset explorer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.datasets = {}
        self.analysis_results = {}

        # Set up plotting style
        plt.style.use(self.config.get('visualization', {}).get('style', 'seaborn-v0_8-darkgrid'))
        sns.set_palette(self.config.get('visualization', {}).get('palette', 'husl'))

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_datasets(self, datasets: Optional[Dict[str, DatasetDict]] = None):
        """
        Load datasets for exploration.

        Args:
            datasets: Pre-loaded datasets. If None, will load from scratch.
        """
        if datasets is None:
            loader = DatasetLoader(config_path=self.config_path)
            self.datasets = loader.load_all_datasets()
        else:
            self.datasets = datasets

        logger.info(f"Loaded {len(self.datasets)} datasets for exploration")

    def analyze_dataset(self, dataset_name: str, split: str = 'train') -> Dict:
        """
        Perform comprehensive analysis on a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split to analyze

        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing {dataset_name} ({split} split)...")

        dataset = self.datasets[dataset_name][split]

        # Convert to pandas for easier analysis
        df = dataset.to_pandas()

        analysis = {
            'dataset_name': dataset_name,
            'split': split,
            'basic_stats': self._get_basic_stats(df),
            'label_distribution': self._get_label_distribution(df),
            'text_statistics': self._get_text_statistics(df),
            'missing_values': self._get_missing_values(df)
        }

        logger.info(f"Analysis complete for {dataset_name}")
        return analysis

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Get basic statistics about the dataset."""
        stats = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        return stats

    def _get_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Get label distribution statistics."""
        if 'label' not in df.columns:
            return {}

        label_counts = df['label'].value_counts().to_dict()
        label_percentages = (df['label'].value_counts(normalize=True) * 100).to_dict()

        return {
            'counts': label_counts,
            'percentages': label_percentages,
            'num_classes': df['label'].nunique(),
            'class_balance_ratio': max(label_counts.values()) / min(label_counts.values()) if label_counts else 1.0
        }

    def _get_text_statistics(self, df: pd.DataFrame) -> Dict:
        """Get text-related statistics."""
        if 'text' not in df.columns:
            return {}

        # Calculate text lengths
        text_lengths = df['text'].astype(str).apply(len)
        word_counts = df['text'].astype(str).apply(lambda x: len(x.split()))

        stats = {
            'char_length': {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'std': float(text_lengths.std()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'quantiles': {
                    '25%': float(text_lengths.quantile(0.25)),
                    '75%': float(text_lengths.quantile(0.75)),
                    '95%': float(text_lengths.quantile(0.95))
                }
            },
            'word_count': {
                'mean': float(word_counts.mean()),
                'median': float(word_counts.median()),
                'std': float(word_counts.std()),
                'min': int(word_counts.min()),
                'max': int(word_counts.max()),
                'quantiles': {
                    '25%': float(word_counts.quantile(0.25)),
                    '75%': float(word_counts.quantile(0.75)),
                    '95%': float(word_counts.quantile(0.95))
                }
            }
        }

        return stats

    def _get_missing_values(self, df: pd.DataFrame) -> Dict:
        """Get missing value statistics."""
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()

        return {
            'counts': missing_counts,
            'percentages': missing_percentages,
            'total_missing': sum(missing_counts.values())
        }

    def analyze_all_datasets(self) -> Dict[str, Dict]:
        """
        Analyze all loaded datasets.

        Returns:
            Dictionary containing analysis results for all datasets
        """
        logger.info("Analyzing all datasets...")

        all_results = {}

        for dataset_name in self.datasets.keys():
            all_results[dataset_name] = {}

            for split in self.datasets[dataset_name].keys():
                analysis = self.analyze_dataset(dataset_name, split)
                all_results[dataset_name][split] = analysis

        self.analysis_results = all_results
        return all_results

    def visualize_dataset(self, dataset_name: str, split: str = 'train',
                         output_dir: str = "reports/figures"):
        """
        Create visualizations for a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split
            output_dir: Output directory for figures
        """
        logger.info(f"Creating visualizations for {dataset_name} ({split})...")

        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        dataset = self.datasets[dataset_name][split]
        df = dataset.to_pandas()

        dpi = self.config.get('visualization', {}).get('dpi', 300)
        fmt = self.config.get('visualization', {}).get('figure_format', 'png')

        # 1. Label distribution
        if 'label' in df.columns:
            self._plot_label_distribution(df, output_path, dataset_name, dpi, fmt)

        # 2. Text length distribution
        if 'text' in df.columns:
            self._plot_text_length_distribution(df, output_path, dataset_name, dpi, fmt)

            # 3. Word cloud
            self._plot_wordcloud(df, output_path, dataset_name, dpi, fmt)

        # 4. Text length by label
        if 'text' in df.columns and 'label' in df.columns:
            self._plot_text_length_by_label(df, output_path, dataset_name, dpi, fmt)

        logger.info(f"Visualizations saved to {output_path}")

    def _plot_label_distribution(self, df: pd.DataFrame, output_path: Path,
                                 dataset_name: str, dpi: int, fmt: str):
        """Plot label distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))

        label_counts = df['label'].value_counts().sort_index()
        ax.bar(label_counts.index, label_counts.values)
        ax.set_xlabel('Label', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Label Distribution - {dataset_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add count labels on bars
        for i, v in enumerate(label_counts.values):
            ax.text(label_counts.index[i], v, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / f'label_distribution.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()

    def _plot_text_length_distribution(self, df: pd.DataFrame, output_path: Path,
                                      dataset_name: str, dpi: int, fmt: str):
        """Plot text length distribution."""
        text_lengths = df['text'].astype(str).apply(len)
        word_counts = df['text'].astype(str).apply(lambda x: len(x.split()))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Character length
        axes[0].hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Character Length', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Character Length Distribution - {dataset_name}',
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Word count
        axes[1].hist(word_counts, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Word Count', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Word Count Distribution - {dataset_name}',
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / f'text_length_distribution.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()

    def _plot_wordcloud(self, df: pd.DataFrame, output_path: Path,
                       dataset_name: str, dpi: int, fmt: str):
        """Generate word cloud."""
        try:
            text = ' '.join(df['text'].astype(str).tolist())

            wordcloud = WordCloud(width=1600, height=800,
                                background_color='white',
                                max_words=100,
                                colormap='viridis').generate(text)

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud - {dataset_name}', fontsize=16, fontweight='bold', pad=20)

            plt.tight_layout()
            plt.savefig(output_path / f'wordcloud.{fmt}', dpi=dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate word cloud: {str(e)}")

    def _plot_text_length_by_label(self, df: pd.DataFrame, output_path: Path,
                                   dataset_name: str, dpi: int, fmt: str):
        """Plot text length distribution by label."""
        df['char_length'] = df['text'].astype(str).apply(len)
        df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Character length by label
        for label in sorted(df['label'].unique()):
            data = df[df['label'] == label]['char_length']
            axes[0].hist(data, bins=30, alpha=0.5, label=f'Label {label}', edgecolor='black')

        axes[0].set_xlabel('Character Length', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Character Length by Label - {dataset_name}',
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Word count by label
        for label in sorted(df['label'].unique()):
            data = df[df['label'] == label]['word_count']
            axes[1].hist(data, bins=30, alpha=0.5, label=f'Label {label}', edgecolor='black')

        axes[1].set_xlabel('Word Count', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Word Count by Label - {dataset_name}',
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / f'text_length_by_label.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close()

    def visualize_all_datasets(self, output_dir: str = "reports/figures"):
        """
        Create visualizations for all datasets.

        Args:
            output_dir: Output directory for figures
        """
        logger.info("Creating visualizations for all datasets...")

        for dataset_name in self.datasets.keys():
            for split in ['train']:  # Usually only visualize train split
                if split in self.datasets[dataset_name]:
                    self.visualize_dataset(dataset_name, split, output_dir)

        logger.info("All visualizations created")

    def save_analysis_report(self, output_dir: str = "reports/analysis"):
        """
        Save analysis results to JSON files.

        Args:
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving analysis reports to {output_dir}...")

        # Save individual dataset reports
        for dataset_name, results in self.analysis_results.items():
            report_file = output_path / f'{dataset_name}_analysis.json'
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved {dataset_name} report to {report_file}")

        # Save combined report
        combined_file = output_path / 'all_datasets_analysis.json'
        with open(combined_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        logger.info(f"Saved combined report to {combined_file}")

        # Create summary CSV
        self._create_summary_csv(output_path)

    def _create_summary_csv(self, output_path: Path):
        """Create a summary CSV of all datasets."""
        summary_data = []

        for dataset_name, results in self.analysis_results.items():
            for split, analysis in results.items():
                row = {
                    'dataset': dataset_name,
                    'split': split,
                    'num_samples': analysis['basic_stats']['num_samples'],
                    'num_classes': analysis['label_distribution'].get('num_classes', 0),
                    'avg_char_length': analysis['text_statistics'].get('char_length', {}).get('mean', 0),
                    'avg_word_count': analysis['text_statistics'].get('word_count', {}).get('mean', 0),
                    'class_balance_ratio': analysis['label_distribution'].get('class_balance_ratio', 1.0)
                }
                summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / 'datasets_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary CSV to {summary_file}")

        # Also print summary
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Explore and analyze datasets")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-figures', type=str, default='reports/figures',
                       help='Output directory for figures')
    parser.add_argument('--output-reports', type=str, default='reports/analysis',
                       help='Output directory for analysis reports')

    args = parser.parse_args()

    # Create explorer
    explorer = DatasetExplorer(config_path=args.config)

    # Load datasets
    explorer.load_datasets()

    # Analyze all datasets
    results = explorer.analyze_all_datasets()

    # Create visualizations
    explorer.visualize_all_datasets(output_dir=args.output_figures)

    # Save reports
    explorer.save_analysis_report(output_dir=args.output_reports)

    print("\nDataset exploration complete!")
    print(f"Figures saved to: {args.output_figures}")
    print(f"Reports saved to: {args.output_reports}")


if __name__ == "__main__":
    main()