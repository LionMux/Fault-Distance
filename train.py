"""
Training script for Fault Distance Estimation CNN.

Usage:
    python train.py --epochs 200 --batch-size 64 --model cnn1d
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR
from tqdm import tqdm
import logging

from config import Config, get_config
from data.dataset import FaultDataset, DataLoaderFactory
from models.cnn1d import CNN1D, DilatedCNN1D
from models.resnet1d import ResNet1D
from utils.metrics import MetricsCalculator
from utils.logger import TrainingLogger
from utils.plots import plot_training_history, plot_predictions


class Trainer:
    """
    Training pipeline for fault distance estimation.
    """
    
    def __init__(self, cfg: Config):
        """
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        
        # Set random seeds for reproducibility
        self._set_seeds(cfg.SEED)
        
        # Initialize logger
        self.logger = TrainingLogger(output_dir=cfg.LOG_DIR)
        self.logger.log_config(cfg)
        
        # Load data
        logging.info("Loading data...")
        self.train_loader, self.test_loader, self.scalers = DataLoaderFactory.create_loaders(
            cfg.CSV_FILE, cfg
        )
        
        # Initialize model
        logging.info(f"Initializing {cfg.MODEL_TYPE} model...")
        self.model = self._build_model().to(self.device)
        self._log_model_info()
        
        # Loss and optimizer
        self.criterion = self._get_loss_function()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Training state
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
    
    def _set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_model(self):
        """Build model based on configuration."""
        if self.cfg.MODEL_TYPE == 'cnn1d':
            return CNN1D(
                seq_length=self.cfg.SEQ_LENGTH,
                num_filters=self.cfg.NUM_FILTERS,
                kernel_size=self.cfg.KERNEL_SIZE,
                dropout=self.cfg.DROPOUT
            )
        elif self.cfg.MODEL_TYPE == 'dilated_cnn1d':
            return DilatedCNN1D(
                seq_length=self.cfg.SEQ_LENGTH,
                num_filters=self.cfg.NUM_FILTERS,
                kernel_size=self.cfg.KERNEL_SIZE,
                dropout=self.cfg.DROPOUT
            )
        elif self.cfg.MODEL_TYPE == 'resnet1d':
            return ResNet1D(
                seq_length=self.cfg.SEQ_LENGTH,
                num_filters=self.cfg.NUM_FILTERS,
                dropout=self.cfg.DROPOUT
            )
        else:
            raise ValueError(f"Unknown model type: {self.cfg.MODEL_TYPE}")
    
    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logging.info(f"\nModel: {self.cfg.MODEL_TYPE}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Model on device: {self.device}\n")
    
    def _get_loss_function(self):
        """Get loss function."""
        if self.cfg.LOSS_FUNCTION == 'mse':
            return nn.MSELoss()
        elif self.cfg.LOSS_FUNCTION == 'mae':
            return nn.L1Loss()
        elif self.cfg.LOSS_FUNCTION == 'smooth_l1':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.cfg.LOSS_FUNCTION}")
    
    def _get_optimizer(self):
        """Get optimizer."""
        if self.cfg.OPTIMIZER == 'adam':
            return Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE, 
                       weight_decay=self.cfg.WEIGHT_DECAY)
        elif self.cfg.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LEARNING_RATE,
                                    weight_decay=self.cfg.WEIGHT_DECAY)
        elif self.cfg.OPTIMIZER == 'sgd':
            return SGD(self.model.parameters(), lr=self.cfg.LEARNING_RATE,
                      momentum=0.9, weight_decay=self.cfg.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.OPTIMIZER}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler."""
        if self.cfg.SCHEDULER_TYPE == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.NUM_EPOCHS,
                                    eta_min=1e-6)
        elif self.cfg.SCHEDULER_TYPE == 'linear':
            return LinearLR(self.optimizer, start_factor=1.0, 
                          total_iters=self.cfg.NUM_EPOCHS)
        elif self.cfg.SCHEDULER_TYPE == 'exponential':
            return ExponentialLR(self.optimizer, gamma=0.95)
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.cfg.NUM_EPOCHS}')
        
        for batch_idx, (signals, distances) in enumerate(pbar):
            signals = signals.to(self.device)
            distances = distances.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(signals)
            loss = self.criterion(predictions, distances)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.cfg.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.cfg.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item():.6f})
            
            # Log batch metrics
            if (batch_idx + 1) % self.cfg.LOG_EVERY_N_BATCHES == 0:
                avg_loss = train_loss / num_batches
                logging.info(f"Batch {batch_idx+1}/{len(self.train_loader)}: "
                           f"Loss = {avg_loss:.6f}")
        
        avg_train_loss = train_loss / num_batches
        return avg_train_loss
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        num_batches = 0
        
        with torch.no_grad():
            for signals, distances in tqdm(self.test_loader, desc='Validation'):
                signals = signals.to(self.device)
                distances = distances.to(self.device)
                
                predictions = self.model(signals)
                loss = self.criterion(predictions, distances)
                
                val_loss += loss.item()
                num_batches += 1
                
                y_true.extend(distances.cpu().numpy().flatten())
                y_pred.extend(predictions.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / num_batches
        
        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Denormalize for MAE calculation
        if self.scalers['distance'] is not None:
            y_true_denorm = self.scalers['distance'].inverse_transform(
                y_true.reshape(-1, 1)
            ).flatten()
            y_pred_denorm = self.scalers['distance'].inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
        else:
            y_true_denorm = y_true
            y_pred_denorm = y_pred
        
        mae = np.mean(np.abs(y_true_denorm - y_pred_denorm))
        
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_mae'].append(mae)
        
        return avg_val_loss, mae, y_true_denorm, y_pred_denorm
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        if is_best:
            path = os.path.join(self.cfg.SAVE_DIR, 'best_model.pth')
        else:
            path = os.path.join(self.cfg.SAVE_DIR, 
                              f'checkpoint_epoch_{self.current_epoch+1}.pth')
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'scalers': self.scalers
        }, path)
        logging.info(f"✓ Checkpoint saved to {path}")
    
    def train(self):
        """Full training loop."""
        logging.info("Starting training...\n")
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, mae, y_true, y_pred = self.validate_epoch()
            
            # Log epoch
            self.logger.log_epoch(epoch, train_loss, val_loss, {'MAE': mae})
            
            # Learning rate scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.cfg.SAVE_BEST_ONLY:
                    self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoints
            if (epoch + 1) % self.cfg.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint()
            
            # Early stopping
            if self.cfg.EARLY_STOPPING and self.patience_counter >= self.cfg.PATIENCE:
                logging.info(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
            
            logging.info(f"Epoch {epoch+1}: "
                       f"Train Loss = {train_loss:.6f}, "
                       f"Val Loss = {val_loss:.6f}, "
                       f"MAE = {mae:.4f} km")
        
        # Plot results
        logging.info("\nGenerating plots...")
        plot_training_history(self.history, 
                            os.path.join(self.cfg.LOG_DIR, 'training_history.png'))
        plot_predictions(y_true, y_pred,
                        os.path.join(self.cfg.LOG_DIR, 'predictions.png'))
        
        logging.info("\n✅ Training completed!")
        logging.info(f"Best model saved to {os.path.join(self.cfg.SAVE_DIR, 'best_model.pth')}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Fault Distance Estimation Model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--model', type=str, default='cnn1d', 
                       choices=['cnn1d', 'dilated_cnn1d', 'resnet1d'],
                       help='Model type')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, 
                       choices=['cuda', 'cpu'], help='Device')
    
    args = parser.parse_args()
    
    # Create config with overrides
    kwargs = {}
    if args.epochs:
        kwargs['NUM_EPOCHS'] = args.epochs
    if args.batch_size:
        kwargs['BATCH_SIZE'] = args.batch_size
    if args.model:
        kwargs['MODEL_TYPE'] = args.model
    if args.lr:
        kwargs['LEARNING_RATE'] = args.lr
    if args.device:
        kwargs['DEVICE'] = args.device
    
    cfg = get_config(**kwargs) if kwargs else Config()
    
    # Train
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
