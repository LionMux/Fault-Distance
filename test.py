"""
Quick test script — runs inference on all CSV files in data/data_test/
and prints a summary table.

Usage:
    python test.py

Place your oscillogram CSV files in data/data_test/ before running.
The script expects files with the same format as training data:
    Columns: distance_km, CT1IA, CT1IB, CT1IC, S1) BUS1UA, S1) BUS1UB, S1) BUS1UC
    Rows   : 400 time steps
"""

import os
import glob
import numpy as np
import torch
import pandas as pd
from config import Config
from models.cnn1d import CNN1D, DilatedCNN1D
from models.resnet1d import ResNet1D

# ============================================================
# TEST CONFIGURATION — edit these if needed
# ============================================================
CHECKPOINT = 'checkpoints/best_model.pth'
TEST_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data_test')
DEVICE     = 'cpu'
HAS_LABELS = True   # set False if your test CSVs have no distance_km column
# ============================================================

SIGNAL_COLS  = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
DISTANCE_COL = 'distance_km'


def build_model(cfg):
    if cfg.MODEL_TYPE == 'cnn1d':
        return CNN1D(
            seq_length=cfg.SEQ_LENGTH,
            num_channels=cfg.NUM_CHANNELS,
            num_filters=cfg.NUM_FILTERS,
            kernel_size=cfg.KERNEL_SIZE,
            dropout=cfg.DROPOUT,
        )
    elif cfg.MODEL_TYPE == 'dilated_cnn1d':
        return DilatedCNN1D(
            seq_length=cfg.SEQ_LENGTH,
            num_channels=cfg.NUM_CHANNELS,
            num_filters=cfg.NUM_FILTERS,
            kernel_size=cfg.KERNEL_SIZE,
            dropout=cfg.DROPOUT,
        )
    elif cfg.MODEL_TYPE == 'resnet1d':
        return ResNet1D(cfg)
    else:
        raise ValueError(f'Unknown model type: {cfg.MODEL_TYPE}')


def load_predictor(checkpoint_path, device):
    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg      = ckpt.get('config', Config())
    scalers  = ckpt.get('scalers', {})
    model    = build_model(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev)
    model.eval()
    print(f'\u2713 Model loaded  ({cfg.MODEL_TYPE})  |  checkpoint epoch {ckpt.get("epoch", "?")}')
    return model, cfg, scalers, dev


def predict_one(csv_path, model, cfg, scalers, device, has_labels):
    df = pd.read_csv(csv_path)

    if has_labels:
        distance_true = float(df[DISTANCE_COL].iloc[0])
        signals = df[SIGNAL_COLS].values.astype('float32')   # (T, 6)
    else:
        distance_true = None
        signals = df.values.astype('float32')

    signals = signals.T   # (6, T)

    # Per-channel normalization
    signal_scalers = scalers.get('signal')
    if signal_scalers:
        for ch, scaler in enumerate(signal_scalers):
            signals[ch] = scaler.transform(signals[ch].reshape(-1, 1)).flatten()

    tensor = torch.FloatTensor(signals).unsqueeze(0).to(device)  # (1, 6, T)

    with torch.no_grad():
        pred_norm = model(tensor).cpu().numpy().flatten()[0]

    dist_scaler = scalers.get('distance')
    prediction  = float(dist_scaler.inverse_transform([[pred_norm]])[0][0]) if dist_scaler else float(pred_norm)

    return prediction, distance_true


def main():
    os.makedirs(TEST_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.csv')))
    if not csv_files:
        print(f'\n[!] No CSV files found in {TEST_DIR}')
        print('    Place your oscillogram CSV files there and re-run test.py')
        return

    model, cfg, scalers, device = load_predictor(CHECKPOINT, DEVICE)

    rows = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            pred, true = predict_one(fpath, model, cfg, scalers, device, HAS_LABELS)
            error = abs(true - pred) if true is not None else None
            rows.append({'File': fname, 'True (km)': true, 'Predicted (km)': round(pred, 4),
                         'Error (km)': round(error, 4) if error is not None else '-'})
        except Exception as e:
            rows.append({'File': fname, 'True (km)': '-', 'Predicted (km)': 'ERROR', 'Error (km)': str(e)})

    df_out = pd.DataFrame(rows)

    print('\n' + '=' * 65)
    print('TEST RESULTS')
    print('=' * 65)
    print(df_out.to_string(index=False))

    if HAS_LABELS:
        errors = [r['Error (km)'] for r in rows if isinstance(r['Error (km)'], float)]
        if errors:
            print('=' * 65)
            print(f'  Files tested : {len(errors)}')
            print(f'  MAE          : {np.mean(errors):.4f} km')
            print(f'  Max error    : {np.max(errors):.4f} km')
            print(f'  Min error    : {np.min(errors):.4f} km')
    print('=' * 65 + '\n')


if __name__ == '__main__':
    main()
