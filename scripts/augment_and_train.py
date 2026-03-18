#!/usr/bin/env python3
"""
Full Pipeline: Augment Dataset + Train Model
============================================

Script that:
1. Augments original dataset (time shifts + Gaussian noise)
2. Loads augmented dataset for training
3. Trains CNN with larger, more robust dataset

Usage:
    python scripts/augment_and_train.py --input data/data_training \
                                         --output data/data_augmented \
                                         --epochs 150 \
                                         --batch-size 32
"""

import os
import sys
import argparse
import logging

# Add data module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.augmentation import AugmentationPipeline
from data.dataset import DataLoaderFactory
from train import Trainer
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Augment dataset and train fault distance estimation model'
    )
    parser.add_argument(
        '--input',
        default='data/data_training',
        help='Input directory with original CSV files'
    )
    parser.add_argument(
        '--output',
        default='data/data_augmented',
        help='Output directory for augmented CSV files'
    )
    parser.add_argument(
        '--skip-augmentation',
        action='store_true',
        help='Skip augmentation, use existing augmented data'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs (default: 150)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--model',
        default='cnn1d',
        choices=['cnn1d', 'dilated_cnn1d', 'resnet1d'],
        help='Model architecture (default: cnn1d)'
    )
    
    args = parser.parse_args()
    
    # ==================== STEP 1: AUGMENTATION ====================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: DATA AUGMENTATION")
    logger.info("="*70)
    
    if not args.skip_augmentation:
        pipeline = AugmentationPipeline(seed=42)
        stats = pipeline.augment_dataset(args.input, args.output)
        
        logger.info(f"\nAugmentation Statistics:")
        logger.info(f"  Original samples    : {stats['original_count']}")
        logger.info(f"  Total created       : {stats['total_created']}")
        logger.info(f"  Expected total      : {stats['expected_total']}")
        logger.info(f"  Time shifts         : {stats['time_shifts']} (left & right)")
        logger.info(f"  SNR levels (dB)     : {stats['snr_levels']}")
    else:
        logger.info("Skipping augmentation, using existing augmented data...")
    
    # ==================== STEP 2: TRAINING ====================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("="*70)
    
    # Create config
    cfg = get_config()
    cfg.DATA_DIR = args.output  # Use augmented data
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.MODEL_TYPE = args.model
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Data directory : {cfg.DATA_DIR}")
    logger.info(f"  Model type     : {cfg.MODEL_TYPE}")
    logger.info(f"  Epochs         : {cfg.NUM_EPOCHS}")
    logger.info(f"  Batch size     : {cfg.BATCH_SIZE}")
    logger.info(f"  Learning rate  : {cfg.LEARNING_RATE}")
    logger.info(f"  Train split    : {cfg.TRAIN_SPLIT} (80% train, 20% test)")
    
    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(cfg)
    
    # Train
    logger.info("\nStarting training...\n")
    trainer.train()
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: {cfg.LOG_DIR}")
    logger.info(f"Best model saved to: {cfg.SAVE_DIR}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
