#!/usr/bin/env python3
"""
Main pipeline script to run all three tasks sequentially.

This script orchestrates the complete workflow:
1. Dataset Exploration & Analysis
2. Dataset Combination
3. Model Fine-Tuning (FinBERT and RoBERTa)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.data_processing.data_loader import DatasetLoader
from src.data_processing.data_explorer import DatasetExplorer
from src.data_processing.data_combiner import DatasetCombiner
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger

logger = setup_logger('pipeline', log_file='logs/pipeline.log')


def task1_explore_datasets(config_path: str):
    """
    Task 1: Dataset Exploration & Analysis.
    """
    logger.info("=" * 80)
    logger.info("TASK 1: DATASET EXPLORATION & ANALYSIS")
    logger.info("=" * 80)

    # Create loader
    loader = DatasetLoader(config_path=config_path, base_dir='data/raw')

    # Load all datasets
    logger.info("Step 1/3: Loading datasets...")
    datasets = loader.load_all_datasets()

    # Save to processed directory
    logger.info("Step 2/3: Saving processed datasets...")
    loader.save_datasets(output_dir='data/processed')

    # Create explorer
    explorer = DatasetExplorer(config_path=config_path)
    explorer.load_datasets(datasets)

    # Analyze all datasets
    logger.info("Step 3/3: Analyzing datasets...")
    explorer.analyze_all_datasets()

    # Create visualizations
    logger.info("Creating visualizations...")
    explorer.visualize_all_datasets(output_dir='reports/figures')

    # Save reports
    logger.info("Saving analysis reports...")
    explorer.save_analysis_report(output_dir='reports/analysis')

    logger.info("Task 1 completed successfully!")
    logger.info("")

    return datasets


def task2_combine_datasets(config_path: str, datasets=None, strategy='balanced'):
    """
    Task 2: Dataset Combination.
    """
    logger.info("=" * 80)
    logger.info("TASK 2: DATASET COMBINATION")
    logger.info("=" * 80)

    # Create combiner
    combiner = DatasetCombiner(config_path=config_path)

    # Load datasets if not provided
    if datasets is None:
        logger.info("Loading datasets from data/processed...")
        # Օգտագործում ենք նույն DatasetLoader-ը, բայց արդեն processed պանակից կարդալու համար
        # Ուշադրություն. data_combiner.py-ում պետք է լինի load_datasets մեթոդ
        combiner.load_datasets(base_dir='data/processed')
    else:
        logger.info("Using pre-loaded datasets...")
        combiner.datasets = datasets

    # Combine datasets
    logger.info(f"Combining datasets using '{strategy}' strategy...")
    combined_dataset = combiner.combine_datasets(strategy=strategy, create_splits=True)

    # Save combined dataset
    logger.info("Saving combined dataset...")
    combiner.save_combined_dataset(
        output_dir='data/combined',
        save_format='arrow'
    )

    logger.info("Task 2 completed successfully!")
    logger.info("")

    return combined_dataset


def task3_train_models(config_path: str, models_to_train=['finbert', 'roberta']):
    """
    Task 3: Model Fine-Tuning.
    """
    logger.info("=" * 80)
    logger.info("TASK 3: MODEL FINE-TUNING")
    logger.info("=" * 80)

    results = {}

    for model_type in models_to_train:
        logger.info("-" * 80)
        logger.info(f"Training {model_type.upper()} model...")
        logger.info("-" * 80)

        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'models/{model_type}/{timestamp}'

        # Create trainer
        trainer = ModelTrainer(config_path=config_path)

        # Setup model
        logger.info(f"Setting up {model_type} model...")
        trainer.setup_model(model_type=model_type, num_labels=2)

        # Setup preprocessor
        logger.info("Setting up preprocessor...")
        trainer.setup_preprocessor()

        # Load dataset
        logger.info("Loading combined dataset...")
        try:
            dataset = trainer.load_dataset('data/combined/combined_dataset')
        except Exception:
            # Fallback: եթե arrow ֆայլը չգտնի, կփորձի ամբողջ պանակը
            logger.warning("Could not load specific arrow file, trying folder load...")
            dataset = trainer.load_dataset('data/combined')

        # Prepare dataset
        logger.info("Preparing dataset...")
        prepared_dataset = trainer.prepare_dataset(dataset)

        # Train model
        logger.info("Training model...")
        train_result = trainer.train(
            train_dataset=prepared_dataset['train'],
            eval_dataset=prepared_dataset['validation'],
            output_dir=output_dir
        )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(
            eval_dataset=prepared_dataset['test'],
            output_dir=output_dir
        )

        # Store results
        results[model_type] = {
            'output_dir': output_dir,
            'train_result': train_result,
            'test_results': test_results
        }

        logger.info(f"{model_type.upper()} training completed!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("")

    logger.info("Task 3 completed successfully!")
    logger.info("")

    return results


def evaluate_models(config_path: str, model_results: dict):
    """
    Evaluate trained models and generate comprehensive reports.
    """
    logger.info("=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    for model_type, result_data in model_results.items():
        logger.info(f"Evaluating {model_type.upper()}...")

        model_path = result_data['output_dir']
        eval_output_dir = f"reports/evaluation/{model_type}"

        # Create evaluator
        evaluator = ModelEvaluator(config_path=config_path)

        # Load model
        evaluator.load_model(model_path)

        # Load test dataset
        dataset = evaluator.load_dataset('data/combined/combined_dataset', split='test')

        # Evaluate
        evaluator.evaluate_dataset(dataset)

        # Create visualizations directory
        Path(f"{eval_output_dir}/figures").mkdir(parents=True, exist_ok=True)

        # Create plots
        evaluator.plot_confusion_matrix(output_dir=f"{eval_output_dir}/figures")
        evaluator.plot_roc_curve(output_dir=f"{eval_output_dir}/figures")
        evaluator.plot_prediction_distribution(output_dir=f"{eval_output_dir}/figures")

        # Save results
        evaluator.save_results(output_dir=eval_output_dir)

        logger.info(f"{model_type.upper()} evaluation completed!")
        logger.info("")


def print_final_summary(model_results: dict):
    """Print final summary of all results."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for model_type, result_data in model_results.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  Output Directory: {result_data['output_dir']}")

        test_results = result_data['test_results']
        print(f"  Test Results:")
        print(f"    Accuracy:  {test_results.get('eval_accuracy', 0):.4f}")
        print(f"    Precision: {test_results.get('eval_precision', 0):.4f}")
        print(f"    Recall:    {test_results.get('eval_recall', 0):.4f}")
        print(f"    F1 Score:  {test_results.get('eval_f1', 0):.4f}")

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run the complete LLM fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--tasks', nargs='+',
                        choices=['explore', 'combine', 'train', 'evaluate', 'all'],
                        default=['all'],
                        help='Tasks to run')
    parser.add_argument('--models', nargs='+',
                        choices=['finbert', 'roberta'],
                        default=['finbert', 'roberta'],
                        help='Models to train')
    parser.add_argument('--strategy', type=str, default='balanced',
                        choices=['concatenate', 'balanced', 'stratified'],
                        help='Dataset combination strategy')

    args = parser.parse_args()

    # Expand 'all' task
    if 'all' in args.tasks:
        args.tasks = ['explore', 'combine', 'train', 'evaluate']

    logger.info("Starting LLM Fine-Tuning Pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info("")

    datasets = None
    combined_dataset = None
    model_results = None

    try:
        # Task 1: Explore datasets
        if 'explore' in args.tasks:
            datasets = task1_explore_datasets(args.config)

        # Task 2: Combine datasets
        if 'combine' in args.tasks:
            combined_dataset = task2_combine_datasets(
                args.config,
                datasets=datasets,
                strategy=args.strategy
            )

        # Task 3: Train models
        if 'train' in args.tasks:
            model_results = task3_train_models(
                args.config,
                models_to_train=args.models
            )

        # Final evaluation
        if 'evaluate' in args.tasks and model_results:
            evaluate_models(args.config, model_results)

        # Print summary
        if model_results:
            print_final_summary(model_results)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()