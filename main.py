import logging
import sys
from src.utils import logger
from pathlib import Path
import argparse
import torch
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_processing.data_loader import DatasetLoader
from src.data_processing.data_explorer import DatasetExplorer
from src.data_processing.data_combiner import DatasetCombiner
from src.models.roberta_model import RoBERTaModel
from src.models.finbert_model import FinBERTModel

# Create logs directory first
Path("outputs/logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/training.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class CounterNarrativePipeline:
    def __init__(self):
        self.datasets = None
        self.combined_data = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

        # Create necessary directories
        Path("outputs/logs").mkdir(parents=True, exist_ok=True)
        Path("outputs/models").mkdir(parents=True, exist_ok=True)
        Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    def step1_explore_datasets(self):
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATASET EXPLORATION AND ANALYSIS")
        logger.info("="*80)

        # Load datasets
        loader = DatasetLoader(base_dir="data/raw")
        self.datasets = loader.load_all_datasets()

        # Display summary
        summary = loader.get_dataset_summary()
        print("\n📊 Dataset Summary:")
        print(summary.to_string())

        # Explore datasets
        explorer = DatasetExplorer(output_dir="outputs/figures")

        # Create overview visualizations
        logger.info("\n🎨 Creating visualizations...")
        explorer.visualize_dataset_overview(self.datasets)

        # Generate detailed report
        report = explorer.generate_full_report(self.datasets)

        # Analyze each dataset individually
        for name, df in self.datasets.items():
            if len(df) > 0:
                explorer.analyze_text_quality(df, name)

        logger.info("\n✅ Step 1 completed successfully!")
        logger.info(f"   - Loaded {len(self.datasets)} datasets")
        logger.info(f"   - Total samples: {sum(len(df) for df in self.datasets.values()):,}")
        logger.info(f"   - Visualizations saved to outputs/figures/")

        return self.datasets

    def step2_combine_datasets(self, strategy: str = "weighted"):
        if self.datasets is None:
            logger.info("📂 Loading datasets...")
            loader = DatasetLoader(data_root="data/raw")
            self.datasets = loader.load_all_datasets()

            if not self.datasets:
                logger.error("❌ No datasets found!")
                return None

        combiner = DatasetCombiner(output_dir="data/combined")

        logger.info(f"\n🔄 Using '{strategy}' combination strategy...")

        # Apply selected strategy
        if strategy == "simple":
            self.combined_data = combiner.combine_balanced_by_dataset(self.datasets)
        elif strategy == "balanced":
            self.combined_data = combiner.combine_stratified_balanced(self.datasets)
        elif strategy == "weighted":
            # ՈՒՂՂՈՒՄ. Այստեղ ներառված են ԲՈԼՈՐ դատասեթերը, որպեսզի ոչ մեկը 0 չլինի
            weights = {
                'liar_train': 1.0,
                'liar_test': 0.5,
                'liar_valid': 0.5,
                'fakenewsnet_politifact': 1.0,
                'fakenewsnet_gossipcop': 0.8,
                'covid19': 1.5,
                'fever_train': 0.5,
                'fever_dev': 0.2,
                'fever_test': 0.2
            }
            self.combined_data = combiner.combine_weighted_balanced(self.datasets, weights)
        elif strategy == "domain":
            logger.warning("Domain strategy not fully implemented, falling back to weighted")
            self.combined_data = combiner.combine_weighted_balanced(self.datasets)
        else:
            logger.error(f"❌ Unknown strategy: {strategy}")
            return None

        # Create train/val/test splits
        logger.info("\n📂 Creating train/validation/test splits...")
        self.train_df, self.val_df, self.test_df = combiner.create_train_val_test_splits(
            self.combined_data,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )

        # Save datasets
        combiner.save_combined_dataset(self.combined_data, f"combined_{strategy}")
        combiner.save_combined_dataset(self.train_df, "train")
        combiner.save_combined_dataset(self.val_df, "val")
        combiner.save_combined_dataset(self.test_df, "test")

        logger.info("\n✅ Step 2 completed successfully!")
        logger.info(f"   - Combined dataset size: {len(self.combined_data):,}")
        logger.info(f"   - Train: {len(self.train_df):,}")
        logger.info(f"   - Val: {len(self.val_df):,}")
        logger.info(f"   - Test: {len(self.test_df):,}")
        logger.info(f"   - Saved to data/combined/")

        return self.train_df, self.val_df, self.test_df

    def step3_train_models(self, models: list = ['roberta', 'finbert']):
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL FINE-TUNING")
        logger.info("="*80)

        # Check if we have data
        if self.train_df is None or self.val_df is None or self.test_df is None:
            logger.info("📂 Loading pre-split datasets...")
            try:
                self.train_df = pd.read_csv("data/combined/train.csv")
                self.val_df = pd.read_csv("data/combined/val.csv")
                self.test_df = pd.read_csv("data/combined/test.csv")
            except FileNotFoundError:
                logger.error("❌ No split datasets found. Run step2 first!")
                return None

        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️  Using device: {device}")
        if device == "cuda":
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

        results = {}

        # Train each model
        for model_type in models:
            logger.info("\n" + "-"*80)
            logger.info(f"🚀 Training {model_type.upper()} model...")
            logger.info("-"*80)

            try:
                if model_type.lower() == 'roberta':
                    # ՈՒՂՂՎԱԾ Է: Կանչում ենք ճիշտ պարամետրերով
                    trainer = RoBERTaModel(model_type='roberta')
                    trainer.load_model() # Չմոռանանք բեռնել մոդելը
                    trainer.prepare_data(self.train_df, self.val_df, self.test_df)
                    trainer.train()

                    # ՈՒՂՂՎԱԾ Է: test() -ի փոխարեն evaluate() test_loader-ով
                    test_metrics = trainer.evaluate(trainer.test_loader, phase="Test")
                    trainer.save_model(f'final_{model_type}')
                    results['RoBERTa'] = test_metrics

                elif model_type.lower() == 'finbert':
                    # ՈՒՂՂՎԱԾ Է: Նույն տրամաբանությունն ենք կիրառում
                    trainer = FinBERTModel(model_type='finbert') # Համոզվիր, որ FinBERT-ի __init__-ն էլ ես այսպես դրել
                    trainer.load_model()
                    trainer.prepare_data(self.train_df, self.val_df, self.test_df)
                    trainer.train()

                    test_metrics = trainer.evaluate(trainer.test_loader, phase="Test")
                    trainer.save_model(f'final_{model_type}')
                    results['FinBERT'] = test_metrics

                logger.info(f"✅ {model_type.upper()} training completed!")

            except Exception as e:
                logger.error(f"❌ Error training {model_type}: {e}")
                continue

        # Compare models
        if len(results) > 1:
            logger.info("\n" + "="*80)
            logger.info("MODEL COMPARISON")
            logger.info("="*80)

            # Ստուգիր, որ այս կլասերն ունես քո utils-ում
            try:
                evaluator = ModelEvaluator()
                comparison = evaluator.compare_models(results)
                print("\n" + comparison.to_string())

                visualizer = TrainingVisualizer()
                visualizer.plot_model_comparison(results)
            except Exception as e:
                logger.warning(f"Could not run comparison/visualization: {e}")

        logger.info("\n✅ Step 3 completed successfully!")
        logger.info(f"   - Trained {len(results)} models")
        logger.info(f"   - Models saved to outputs/models/")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Counter-Narrative Generation Pipeline - Steps 1-3'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='Steps to run (1, 2, 3 or combination)'
    )
    parser.add_argument(
        '--combination-strategy',
        type=str,
        default='weighted',
        choices=['simple', 'balanced', 'weighted', 'domain'],
        help='Dataset combination strategy'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['roberta', 'finbert'],
        choices=['roberta', 'finbert'],
        help='Models to train'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = CounterNarrativePipeline()

    print("\n" + "="*80)
    print("COUNTER-NARRATIVE GENERATION PIPELINE")
    print("Author: Sona Mansuryan")
    print("="*80)

    # Run selected steps
    if 1 in args.steps:
        pipeline.step1_explore_datasets()

    if 2 in args.steps:
        pipeline.step2_combine_datasets(strategy=args.combination_strategy)

    if 3 in args.steps:
        pipeline.step3_train_models(models=args.models)

    logger.info("\n" + "="*80)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nNext Steps:")
    logger.info("  - Review outputs in outputs/ directory")
    logger.info("  - Check visualizations in outputs/figures/")
    logger.info("  - Fine-tuned models saved in outputs/models/")
    logger.info("  - Proceed to Step 4: Data collection with Wikipedia API")
    logger.info("  - Then Step 5: Counter-narrative generation with Mistral 7B")


if __name__ == "__main__":
    main()