import torch
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    BertTokenizer, BertForSequenceClassification
)
from pathlib import Path
import pandas as pd
from src.utils import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleDetector:

    def __init__(self, roberta_path, bert_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load RoBERTa
        logger.info(f"Loading RoBERTa from {roberta_path}...")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(
            roberta_path,
            ignore_mismatched_sizes=True
        )
        self.roberta_model.to(self.device)
        self.roberta_model.eval()

        # Load BERT
        logger.info(f"Loading BERT from {bert_path}...")
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert_model = BertForSequenceClassification.from_pretrained(
            bert_path,
            ignore_mismatched_sizes=True
        )
        self.bert_model.to(self.device)
        self.bert_model.eval()

        logger.info("✓ Ensemble models loaded")

    def predict_single(self, text, method='average'):

        # RoBERTa prediction
        roberta_encoding = self.roberta_tokenizer(
            text, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            roberta_outputs = self.roberta_model(
                input_ids=roberta_encoding['input_ids'].to(self.device),
                attention_mask=roberta_encoding['attention_mask'].to(self.device)
            )
            roberta_probs = torch.softmax(roberta_outputs.logits, dim=1)[0]

        # BERT prediction
        bert_encoding = self.bert_tokenizer(
            text, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            bert_outputs = self.bert_model(
                input_ids=bert_encoding['input_ids'].to(self.device),
                attention_mask=bert_encoding['attention_mask'].to(self.device)
            )
            bert_probs = torch.softmax(bert_outputs.logits, dim=1)[0]

        # Combine predictions
        if method == 'average':
            combined_probs = (roberta_probs + bert_probs) / 2
        elif method == 'weighted':
            # RoBERTa performed better (0.8078 vs 0.7991)
            combined_probs = roberta_probs * 0.6 + bert_probs * 0.4
        elif method == 'voting':
            roberta_pred = torch.argmax(roberta_probs).item()
            bert_pred = torch.argmax(bert_probs).item()
            if roberta_pred == bert_pred:
                prediction = roberta_pred
            else:
                # If disagree, use the more confident one
                roberta_conf = roberta_probs[roberta_pred].item()
                bert_conf = bert_probs[bert_pred].item()
                prediction = roberta_pred if roberta_conf > bert_conf else bert_pred

            combined_probs = torch.zeros(2)
            combined_probs[prediction] = 1.0

        prediction = torch.argmax(combined_probs).item()
        confidence = combined_probs[prediction].item()

        return {
            'prediction': prediction,
            'label': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': confidence,
            'fake_prob': combined_probs[0].item(),
            'real_prob': combined_probs[1].item(),
            'roberta_probs': roberta_probs.cpu().numpy(),
            'bert_probs': bert_probs.cpu().numpy()
        }

    def evaluate(self, texts, labels, method='weighted'):

        predictions = []

        logger.info(f"Evaluating ensemble ({method} method)...")
        for text in texts:
            pred = self.predict_single(text, method)
            predictions.append(pred['prediction'])

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        cm = confusion_matrix(labels, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }


def evaluate_ensemble():

    logger.info("="*80)
    logger.info("ENSEMBLE MODEL EVALUATION")
    logger.info("="*80)

    # Load test data
    logger.info("\nLoading test data...")
    test_df = pd.read_csv("data/combined/test_final.csv")
    logger.info(f"✓ Test samples: {len(test_df):,}")

    # Initialize ensemble
    roberta_path = Path("outputs/models/final_roberta_enhanced")
    bert_path = Path("outputs/models/final_bert_enhanced")

    ensemble = EnsembleDetector(str(roberta_path), str(bert_path))

    # Test different ensemble methods
    methods = ['average', 'weighted', 'voting']

    results = {}

    # Take a sample for faster testing (or use full test set)
    sample_size = 1000
    test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

    logger.info(f"\nTesting on {len(test_sample)} samples...")

    for method in methods:
        logger.info(f"\nMethod: {method}")
        metrics = ensemble.evaluate(
            test_sample['text'].tolist(),
            test_sample['binary_label'].tolist(),
            method=method
        )

        results[method] = metrics

        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")

    # Compare with individual models
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)

    logger.info(f"\nRoBERTa:           F1 = 0.8078")
    logger.info(f"BERT:              F1 = 0.7991")

    for method, metrics in results.items():
        logger.info(f"Ensemble ({method:8s}): F1 = {metrics['f1']:.4f}")

    # Best method
    best_method = max(results.items(), key=lambda x: x[1]['f1'])
    logger.info(f"\n✓ Best ensemble method: {best_method[0]}")
    logger.info(f"  F1 Score: {best_method[1]['f1']:.4f}")


if __name__ == "__main__":
    try:
        evaluate_ensemble()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)