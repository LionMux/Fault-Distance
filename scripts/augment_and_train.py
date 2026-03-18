#!/usr/bin/env python3
"""
Full Pipeline: Augment Dataset + Train Model
============================================

FIXED: Val set is now split from ORIGINAL files BEFORE augmentation.
       Only the train portion is augmented -> no data leakage into val.

Pipeline:
    1. Split original files into train_files (80%) and val_files (20%)
    2. Copy val_files -> data/data_val/          (originals, untouched)
    3. Augment train_files -> data/data_augmented_train/
    4. Train using two separate DataLoaders (augmented train + clean val)

Usage:
    python scripts/augment_and_train.py --input data/data_training \
                                         --epochs 150 \
                                         --batch-size 32
"""

import os
import sys
import glob
import shutil
import random
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.augmentation import AugmentationPipeline
from data.dataset import FaultDataset, DataLoaderFactory
from torch.utils.data import DataLoader
from train import Trainer
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_files(input_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Split original CSV files into train/val lists BY FILENAME.
    No augmentation happens here — pure file-level split.
    """
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    random.seed(seed)
    shuffled = csv_files.copy()
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:]

    logger.info(f"Original files total : {len(csv_files)}")
    logger.info(f"Train files          : {len(train_files)} ({train_ratio*100:.0f}%)")
    logger.info(f"Val files            : {len(val_files)}   ({(1-train_ratio)*100:.0f}%)")
    return train_files, val_files


def copy_files_to_dir(file_list: list, target_dir: str):
    """Copy a list of files into target_dir (creates dir if needed)."""
    os.makedirs(target_dir, exist_ok=True)
    for fpath in file_list:
        shutil.copy2(fpath, target_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Augment train split only, keep val clean'
    )
    parser.add_argument('--input',      default='data/data_training')
    parser.add_argument('--train-aug',  default='data/data_augmented_train',
                        help='Output dir for augmented TRAIN files')
    parser.add_argument('--val-dir',    default='data/data_val',
                        help='Output dir for clean VAL files (originals only)')
    parser.add_argument('--skip-augmentation', action='store_true')
    parser.add_argument('--epochs',     type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model',      default='cnn1d',
                        choices=['cnn1d', 'dilated_cnn1d', 'resnet1d'])
    args = parser.parse_args()

    # ==================== STEP 1: FILE-LEVEL SPLIT ====================
    logger.info("=" * 70)
    logger.info("STEP 1: SPLIT ORIGINAL FILES -> train / val")
    logger.info("=" * 70)

    cfg = get_config()
    train_files, val_files = split_files(
        args.input,
        train_ratio=cfg.TRAIN_SPLIT,
        seed=cfg.SEED
    )

    # Copy val originals to data/data_val (clean, no augmentation)
    logger.info(f"Copying val originals -> {args.val_dir}")
    copy_files_to_dir(val_files, args.val_dir)

    # Copy train originals to a temp staging dir for augmentation
    train_staging = args.train_aug + '_staging'
    logger.info(f"Copying train originals -> {train_staging} (staging)")
    copy_files_to_dir(train_files, train_staging)

    # ==================== STEP 2: AUGMENT TRAIN ONLY ====================
    logger.info("=" * 70)
    logger.info("STEP 2: AUGMENT TRAIN FILES ONLY")
    logger.info("=" * 70)

    if not args.skip_augmentation:
        pipeline = AugmentationPipeline(seed=cfg.SEED)
        stats = pipeline.augment_dataset(train_staging, args.train_aug)

        # Also copy the original train files into augmented dir
        # so train set = originals + augmented versions
        copy_files_to_dir(train_files, args.train_aug)

        logger.info(f"Augmentation stats:")
        logger.info(f"  Original train samples : {stats['original_count']}")
        logger.info(f"  Augmented created      : {stats['total_created']}")
        logger.info(f"  Total train samples    : {stats['total_created'] + len(train_files)}")
    else:
        logger.info("Skipping augmentation, using existing dirs...")

    # ==================== STEP 3: TRAINING ====================
    logger.info("=" * 70)
    logger.info("STEP 3: TRAINING (augmented train + clean val)")
    logger.info("=" * 70)

    cfg.NUM_EPOCHS  = args.epochs
    cfg.BATCH_SIZE  = args.batch_size
    cfg.MODEL_TYPE  = args.model

    # Build two SEPARATE datasets: augmented train + clean val
    train_dataset = FaultDataset(
        data_dir=args.train_aug,
        seq_length=cfg.SEQ_LENGTH,
        num_channels=cfg.NUM_CHANNELS,
        normalize=cfg.NORMALIZE_DATA,
    )
    # Val uses scalers fitted on TRAIN to avoid leakage
    val_dataset = FaultDataset(
        data_dir=args.val_dir,
        seq_length=cfg.SEQ_LENGTH,
        num_channels=cfg.NUM_CHANNELS,
        normalize=cfg.NORMALIZE_DATA,
        signal_scalers=train_dataset.signal_scalers,
        distance_scaler=train_dataset.distance_scaler,
    )

    pin = cfg.DEVICE == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=pin
    )

    logger.info(f"Train samples : {len(train_dataset)}")
    logger.info(f"Val samples   : {len(val_dataset)}")
    logger.info(f"Model         : {cfg.MODEL_TYPE}")
    logger.info(f"Epochs        : {cfg.NUM_EPOCHS}")
    logger.info(f"Batch size    : {cfg.BATCH_SIZE}")

    trainer = Trainer(cfg)
    trainer.train_loader = train_loader
    trainer.val_loader   = val_loader
    trainer.train()

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best model -> {cfg.SAVE_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
