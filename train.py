"""
Training script for Fault Distance Estimation CNN.
"""

import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR
from tqdm import tqdm
import logging
from datetime import datetime

from config import Config, get_config
from data.dataset import FaultDataset, DataLoaderFactory
from models.cnn1d import CNN1D, DilatedCNN1D
from models.resnet1d import ResNet1D
from utils.metrics import MetricsCalculator
from utils.logger import TrainingLogger
from utils.plots import plot_training_history, plot_predictions


class Trainer:
    """Full training pipeline for fault distance estimation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self._set_seeds(cfg.SEED)

        # --- Create unique run directory: logs/run_YYYYMMDD_HHMMSS/ ---
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(cfg.LOG_DIR, f'run_{self.run_timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)

        # TrainingLogger writes .log into run_dir
        self.logger = TrainingLogger(output_dir=self.run_dir)
        self.logger.log_config(cfg)

        logging.info("Loading data...")
        self.train_loader, self.test_loader, self.scalers = DataLoaderFactory.create_loaders(
            cfg.DATA_DIR, cfg
        )

        train_n = len(self.train_loader.dataset)
        val_n   = len(self.test_loader.dataset)
        logging.info(f"Dataset        : train={train_n} samples | val={val_n} samples")
        logging.info(f"Device         : {cfg.DEVICE}")
        if self.scalers['distance'] is not None:
            dmin = self.scalers['distance'].data_min_[0]
            dmax = self.scalers['distance'].data_max_[0]
            logging.info(f"Distance range : [{dmin:.2f}, {dmax:.2f}] km (scaler)")

        logging.info(f"Initializing {cfg.MODEL_TYPE} model...")
        self.model = self._build_model().to(self.device)
        self._log_model_info()

        self.criterion = self._get_loss_function()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self):
        if self.cfg.MODEL_TYPE == 'cnn1d':
            return CNN1D(seq_length=self.cfg.SEQ_LENGTH, num_channels=self.cfg.NUM_CHANNELS,
                         num_filters=self.cfg.NUM_FILTERS, kernel_size=self.cfg.KERNEL_SIZE,
                         dropout=self.cfg.DROPOUT)
        elif self.cfg.MODEL_TYPE == 'dilated_cnn1d':
            return DilatedCNN1D(seq_length=self.cfg.SEQ_LENGTH, num_channels=self.cfg.NUM_CHANNELS,
                                num_filters=self.cfg.NUM_FILTERS, kernel_size=self.cfg.KERNEL_SIZE,
                                dropout=self.cfg.DROPOUT)
        elif self.cfg.MODEL_TYPE == 'resnet1d':
            if not hasattr(self.cfg, 'BASE_CHANNELS'): self.cfg.BASE_CHANNELS = self.cfg.NUM_FILTERS
            if not hasattr(self.cfg, 'DEPTH'):          self.cfg.DEPTH = 3
            if not hasattr(self.cfg, 'DROPOUT_RATE'):   self.cfg.DROPOUT_RATE = self.cfg.DROPOUT
            if not hasattr(self.cfg, 'USE_SE_BLOCK'):   self.cfg.USE_SE_BLOCK = True
            if not hasattr(self.cfg, 'TASK'):           self.cfg.TASK = 'regression'
            return ResNet1D(self.cfg)
        raise ValueError(f"Unknown model: {self.cfg.MODEL_TYPE}")

    def _log_model_info(self):
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model        : {self.cfg.MODEL_TYPE}")
        logging.info(f"Channels     : {self.cfg.NUM_CHANNELS}")
        logging.info(f"SeqLength    : {self.cfg.SEQ_LENGTH}")
        logging.info(f"Parameters   : {total:,} total / {trainable:,} trainable")

    def _get_loss_function(self):
        return {'mse': nn.MSELoss(), 'mae': nn.L1Loss(),
                'smooth_l1': nn.SmoothL1Loss()}[self.cfg.LOSS_FUNCTION]

    def _get_optimizer(self):
        if self.cfg.OPTIMIZER == 'adam':
            return Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE,
                        weight_decay=self.cfg.WEIGHT_DECAY)
        elif self.cfg.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LEARNING_RATE,
                                     weight_decay=self.cfg.WEIGHT_DECAY)
        elif self.cfg.OPTIMIZER == 'sgd':
            return SGD(self.model.parameters(), lr=self.cfg.LEARNING_RATE,
                       momentum=0.9, weight_decay=self.cfg.WEIGHT_DECAY)
        raise ValueError(f"Unknown optimizer: {self.cfg.OPTIMIZER}")

    def _get_scheduler(self):
        if self.cfg.SCHEDULER_TYPE == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.NUM_EPOCHS, eta_min=1e-6)
        elif self.cfg.SCHEDULER_TYPE == 'linear':
            return LinearLR(self.optimizer, start_factor=1.0, total_iters=self.cfg.NUM_EPOCHS)
        elif self.cfg.SCHEDULER_TYPE == 'exponential':
            return ExponentialLR(self.optimizer, gamma=0.95)
        return None

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.cfg.NUM_EPOCHS}')
        for signals, distances in pbar:
            signals, distances = signals.to(self.device), distances.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(signals), distances)
            loss.backward()
            if self.cfg.GRADIENT_CLIP:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP)
            self.optimizer.step()
            total_loss += loss.item(); n += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        return total_loss / n

    def validate_epoch(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for signals, distances in tqdm(self.test_loader, desc='Validation'):
                signals, distances = signals.to(self.device), distances.to(self.device)
                preds = self.model(signals)
                total_loss += self.criterion(preds, distances).item(); n += 1
                y_true.extend(distances.cpu().numpy().flatten())
                y_pred.extend(preds.cpu().numpy().flatten())
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        scaler = self.scalers['distance']
        if scaler is not None:
            y_true = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
        mae = np.mean(np.abs(y_true - y_pred))
        self.history['val_loss'].append(total_loss / n)
        self.history['val_mae'].append(mae)
        return total_loss / n, mae, y_true, y_pred

    def save_checkpoint(self, is_best=False):
        fname = 'best_model.pth' if is_best else f'checkpoint_epoch_{self.current_epoch+1}.pth'
        path  = os.path.join(self.cfg.SAVE_DIR, fname)
        torch.save({'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.cfg, 'scalers': self.scalers}, path)
        logging.info(f"Checkpoint saved -> {path}")

    def train(self):
        train_n = len(self.train_loader.dataset)
        val_n   = len(self.test_loader.dataset)
        logging.info("=" * 60)
        logging.info("Starting training")
        logging.info(f"  Train samples : {train_n}")
        logging.info(f"  Val   samples : {val_n}")
        logging.info(f"  Epochs        : {self.cfg.NUM_EPOCHS}")
        logging.info(f"  Batch size    : {self.cfg.BATCH_SIZE}")
        logging.info(f"  Device        : {self.cfg.DEVICE}")
        logging.info(f"  Run dir       : {self.run_dir}")
        logging.info("=" * 60)

        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            val_loss, mae, y_true, y_pred = self.validate_epoch()
            self.logger.log_epoch(epoch, train_loss, val_loss, {'MAE': mae})
            if self.scheduler: self.scheduler.step()
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.cfg.SAVE_BEST_ONLY: self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            if (epoch + 1) % self.cfg.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint()
            if self.cfg.EARLY_STOPPING and self.patience_counter >= self.cfg.PATIENCE:
                logging.info(f"Early stopping after {epoch+1} epochs")
                break
            logging.info(f"Epoch {epoch+1}: train={train_loss:.6f}  val={val_loss:.6f}  MAE={mae:.4f} km")

        # --- Save plots INTO run_dir (same folder as .log) ---
        plot_training_history(self.history,
                              os.path.join(self.run_dir, 'training_history.png'))
        plot_predictions(y_true, y_pred,
                         os.path.join(self.run_dir, 'predictions.png'))

        logging.info(f"Plots saved -> {self.run_dir}")
        logging.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--model',      type=str, default='cnn1d',
                        choices=['cnn1d','dilated_cnn1d','resnet1d'])
    parser.add_argument('--lr',         type=float)
    parser.add_argument('--data-dir',   type=str)
    parser.add_argument('--device',     type=str, choices=['cuda','cpu'])
    args = parser.parse_args()
    kwargs = {}
    if args.epochs:     kwargs['NUM_EPOCHS']    = args.epochs
    if args.batch_size: kwargs['BATCH_SIZE']    = args.batch_size
    if args.model:      kwargs['MODEL_TYPE']    = args.model
    if args.lr:         kwargs['LEARNING_RATE'] = args.lr
    if args.data_dir:   kwargs['DATA_DIR']      = args.data_dir
    if args.device:     kwargs['DEVICE']        = args.device
    cfg = get_config(**kwargs) if kwargs else Config()
    Trainer(cfg).train()

if __name__ == '__main__':
    main()
