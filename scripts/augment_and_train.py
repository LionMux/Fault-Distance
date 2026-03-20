#!/usr/bin/env python3
"""
Full Pipeline: Augment Dataset + Train Model

Supports three launch modes:

  # 1. Defaults (no args needed):
  python scripts/augment_and_train.py

  # 2. YAML config:
  python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml

  # 3. YAML + CLI override:
  python scripts/augment_and_train.py \
      --config configs/augment_train_resnet1d.yaml \
      --set training.num_epochs=200 model.dropout=0.1

  # 4. Legacy CLI (backward compat):
  python scripts/augment_and_train.py --epochs 150 --batch-size 32 --model cnn1d

FIXED: trainer.test_loader overridden with val_loader
       (validate_epoch reads self.test_loader, not self.val_loader)
FIXED: trainer.scalers overridden with train_dataset scalers
       so MAE inverse_transform is consistent
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
from data.dataset import FaultDataset
from torch.utils.data import DataLoader
from train import Trainer
from config import get_config, load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_files(input_dir, train_ratio=0.8, seed=42):
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {input_dir}")
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


def copy_files_to_dir(file_list, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for fpath in file_list:
        shutil.copy2(fpath, target_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Augment + Train pipeline with YAML config support',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # YAML mode
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config. Example: configs/augment_train_cnn1d.yaml')
    parser.add_argument('--set', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Point overrides on top of YAML.\nExample: --set training.num_epochs=200 model.type=resnet1d')
    # Legacy CLI (backward compat)
    parser.add_argument('--input',             default=None)
    parser.add_argument('--train-aug',         default=None)
    parser.add_argument('--val-dir',           default=None)
    parser.add_argument('--skip-augmentation', action='store_true')
    parser.add_argument('--epochs',            type=int, default=None)
    parser.add_argument('--batch-size',        type=int, default=None)
    parser.add_argument('--model',             default=None,
                        choices=['cnn1d', 'dilated_cnn1d', 'resnet1d'])
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Build config: YAML > --set > legacy flags > defaults
    # ----------------------------------------------------------------
    if args.config:
        overrides = {}
        for item in (args.set or []):
            if '=' not in item:
                logger.warning(f"--set '{item}' skipped (no '=')")
                continue
            k, v = item.split('=', 1)
            overrides[k] = v
        cfg = load_config(args.config, overrides if overrides else None)
        logger.info(f"Config loaded from: {args.config}")
    else:
        kwargs = {}
        if args.epochs:     kwargs['NUM_EPOCHS'] = args.epochs
        if args.batch_size: kwargs['BATCH_SIZE'] = args.batch_size
        if args.model:      kwargs['MODEL_TYPE'] = args.model
        cfg = get_config(**kwargs) if kwargs else get_config()
        cfg._augmentation_cfg = {}

    # Resolve augmentation paths: CLI flags > YAML > defaults
    aug = getattr(cfg, '_augmentation_cfg', {})
    input_dir = args.input     or aug.get('input_dir',     'data/data_training')
    train_aug = args.train_aug or aug.get('train_aug_dir', 'data/data_augmented_train')
    val_dir   = args.val_dir   or aug.get('val_dir',       'data/data_val')
    skip_aug  = args.skip_augmentation or aug.get('skip_augmentation', False)

    # Legacy flags override YAML if explicitly provided
    if args.epochs:     cfg.NUM_EPOCHS = args.epochs
    if args.batch_size: cfg.BATCH_SIZE = args.batch_size
    if args.model:      cfg.MODEL_TYPE = args.model

    # ----------------------------------------------------------------
    # STEP 1: SPLIT
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("STEP 1: SPLIT ORIGINAL FILES -> train / val")
    logger.info("=" * 70)
    train_files, val_files = split_files(input_dir, cfg.TRAIN_SPLIT, cfg.SEED)
    logger.info(f"Copying val originals  -> {val_dir}")
    copy_files_to_dir(val_files, val_dir)
    train_staging = train_aug + '_staging'
    logger.info(f"Copying train originals -> {train_staging} (staging)")
    copy_files_to_dir(train_files, train_staging)

    # ----------------------------------------------------------------
    # STEP 2: AUGMENTATION
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("STEP 2: AUGMENT TRAIN FILES ONLY")
    logger.info("=" * 70)
    if not skip_aug:
        pipeline = AugmentationPipeline(seed=cfg.SEED)
        stats = pipeline.augment_dataset(train_staging, train_aug)
        copy_files_to_dir(train_files, train_aug)
        logger.info(f"Augmentation stats:")
        logger.info(f"  Original train samples : {stats['original_count']}")
        logger.info(f"  Augmented created      : {stats['total_created']}")
        logger.info(f"  Total train samples    : {stats['total_created'] + len(train_files)}")
    else:
        logger.info("Skipping augmentation, using existing dirs...")

    # ----------------------------------------------------------------
    # STEP 3: TRAINING
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("STEP 3: TRAINING (augmented train + clean val)")
    logger.info("=" * 70)

    # Train scaler fitted on train only -> no leakage
    train_dataset = FaultDataset(
        data_dir=train_aug,
        seq_length=cfg.SEQ_LENGTH,
        num_channels=cfg.NUM_CHANNELS,
        normalize=cfg.NORMALIZE_DATA,
    )
    # Val reuses train scalers -> no leakage
    val_dataset = FaultDataset(
        data_dir=val_dir,
        seq_length=cfg.SEQ_LENGTH,
        num_channels=cfg.NUM_CHANNELS,
        normalize=cfg.NORMALIZE_DATA,
        signal_scalers=train_dataset.signal_scalers,
        distance_scaler=train_dataset.distance_scaler,
    )

    pin = cfg.DEVICE == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=pin)

    logger.info(f"Train samples  : {len(train_dataset)}")
    logger.info(f"Val   samples  : {len(val_dataset)}")
    logger.info(f"Model          : {cfg.MODEL_TYPE}")
    logger.info(f"Epochs         : {cfg.NUM_EPOCHS}")
    logger.info(f"Batch size     : {cfg.BATCH_SIZE}")
    logger.info(f"Experiment     : {cfg.EXPERIMENT_NAME}")
    dmin = train_dataset.distance_scaler.data_min_[0]
    dmax = train_dataset.distance_scaler.data_max_[0]
    logger.info(f"Distance scaler range (train): [{dmin:.2f}, {dmax:.2f}] km")

    trainer = Trainer(cfg)
    # FIX: validate_epoch() reads self.test_loader -> assign val_loader here
    trainer.test_loader  = val_loader
    trainer.train_loader = train_loader
    # FIX: scalers must match train normalization for correct MAE km
    trainer.scalers = {
        'signal':   train_dataset.signal_scalers,
        'distance': train_dataset.distance_scaler,
    }
    trainer.train()

    # Save used config for reproducibility
    if args.config:
        dest = os.path.join(trainer.run_dir, 'config_used.yaml')
        shutil.copy2(args.config, dest)
        logger.info(f"Config saved   : {dest}")

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best model -> {cfg.SAVE_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
