""" RoBERTa model """

import pandas as pd
import numpy as np
import torch
import yaml
import logging
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from dataclasses import dataclass

# PyTorch Imports
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

# Transformers Imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoConfig
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper Class to convert YAML dict to Object ---
class ConfigObject:
    """Helper to convert dictionary to object attributes for easier access"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

class FakeNewsDataset(Dataset):
    """Custom Dataset for fake news classification"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RoBERTaModel:
    """Fine-tune RoBERTa for fake news detection - OPTIMIZED"""

    def __init__(self, config_path: str = "configs/config.yaml", model_type: str = "roberta"):
        # 1. Load YAML Config
        self.raw_config = self._load_yaml(config_path)

        # 2. Extract settings nicely
        self.model_settings = self.raw_config['models'][model_type]
        self.train_settings = self.raw_config['training']
        self.data_settings = self.raw_config['data_processing']
        self.hw_settings = self.raw_config['hardware']

        # 3. Setup Hardware
        self.device = torch.device(self.hw_settings.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Mixed Precision Setup
        self.use_amp = (self.device.type == 'cuda') and self.train_settings.get('fp16', False)
        self.scaler = GradScaler() if self.use_amp else None

        if self.use_amp:
            logger.info("✓ Mixed Precision Training ENABLED (FP16)")

        # 4. Initialize Tokenizer & Model
        model_name = self.model_settings['name']
        logger.info(f"Loading {model_name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading {model_name} model...")

        # Load Config first to inject Dropout
        hf_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=self.model_settings['num_labels'],
            hidden_dropout_prob=self.model_settings.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=self.model_settings.get('attention_probs_dropout_prob', 0.1)
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=hf_config,
            ignore_mismatched_sizes=True
        ).to(self.device)

        logger.info(f"✓ Model loaded on {self.device}")

        # 5. Prepare Output Directory
        self.output_dir = Path(self.model_settings['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Metrics history
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }

    def _load_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Prepare data loaders with optimized settings"""
        logger.info("Preparing datasets...")

        max_len = self.model_settings.get('max_length', 256)
        batch_size = self.train_settings['batch_size']
        num_workers = self.hw_settings.get('num_workers', 2)

        def create_dataset(df):
            return FakeNewsDataset(
                df['text'].tolist(),
                df['label'].tolist(), # Assumes 'label' column exists
                self.tokenizer,
                max_len
            )

        self.train_loader = DataLoader(
            create_dataset(train_df), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            create_dataset(val_df), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            create_dataset(test_df), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        logger.info(f"✓ Data prepared. Train batches: {len(self.train_loader)}")

    def train(self):
        """Main training loop"""
        epochs = self.train_settings['num_epochs']
        lr = float(self.train_settings['learning_rate'])
        weight_decay = self.train_settings.get('weight_decay', 0.01)

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * self.train_settings.get('warmup_ratio', 0.1))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        logger.info(f"Starting training for {epochs} epochs...")
        best_val_f1 = 0.0

        for epoch in range(epochs):
            # --- TRAIN LOOP ---
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Mixed Precision Forward Pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_settings.get('max_grad_norm', 1.0))
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                progress.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)
            train_acc = correct / total
            logger.info(f"Train Loss: {avg_loss:.4f} | Acc: {train_acc:.4f}")

            # --- VALIDATION LOOP ---
            val_metrics = self.evaluate(self.val_loader)
            logger.info(f"Val F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

            # Early Stopping / Checkpointing
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model("best_model")
                logger.info("✅ Best model saved!")

            # Record History
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['val_f1'].append(val_metrics['f1'])

    def evaluate(self, loader, phase="Validation"):
        self.model.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=phase):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                predictions = torch.argmax(outputs.logits, dim=1)
                preds.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        cm = confusion_matrix(true_labels, preds)

        return {
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "confusion_matrix": cm.tolist()
        }

    def save_model(self, sub_dir: str):
        path = self.output_dir / sub_dir
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save history
        with open(path / 'history.json', 'w') as f:
            json.dump(self.training_history, f)

if __name__ == "__main__":
    # Test Run
    try:
        train_df = pd.read_csv("data/combined/train.csv")
        val_df = pd.read_csv("data/combined/val.csv")
        test_df = pd.read_csv("data/combined/test.csv")

        trainer = RoBERTaModel(model_type="roberta")
        trainer.prepare_data(train_df, val_df, test_df)
        trainer.train()

    except FileNotFoundError:
        print("❌ CSV files not found. Make sure you have run the 'combine' step.")
    except Exception as e:
        print(f"❌ Error: {e}")