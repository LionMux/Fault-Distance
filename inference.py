"""
Inference script for predicting fault distance from oscillograms.

Usage:
    python inference.py --model checkpoints/best_model.pth --csv data/data_training/1A_0.5km.csv --has-labels --device cpu
    python inference.py --model checkpoints/best_model.pth --csv data/data_training/1A_0.5km.csv --device cpu
"""

import argparse
import numpy as np
import torch
import pandas as pd

from config import Config
from models.cnn1d import CNN1D, DilatedCNN1D
from models.resnet1d import ResNet1D

SIGNAL_COLS = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
DISTANCE_COL = 'distance_km'


class FaultDistancePredictor:

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.cfg = checkpoint.get('config', Config())
        self.scalers = checkpoint.get('scalers', {})

        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded. Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    def _build_model(self):
        if self.cfg.MODEL_TYPE == 'cnn1d':
            return CNN1D(
                seq_length=self.cfg.SEQ_LENGTH,
                num_channels=self.cfg.NUM_CHANNELS,
                num_filters=self.cfg.NUM_FILTERS,
                kernel_size=self.cfg.KERNEL_SIZE,
                dropout=self.cfg.DROPOUT,
            )
        elif self.cfg.MODEL_TYPE == 'dilated_cnn1d':
            return DilatedCNN1D(
                seq_length=self.cfg.SEQ_LENGTH,
                num_channels=self.cfg.NUM_CHANNELS,
                num_filters=self.cfg.NUM_FILTERS,
                kernel_size=self.cfg.KERNEL_SIZE,
                dropout=self.cfg.DROPOUT,
            )
        elif self.cfg.MODEL_TYPE == 'resnet1d':
            return ResNet1D(self.cfg)
        else:
            raise ValueError(f"Unknown model type: {self.cfg.MODEL_TYPE}")

    def predict_from_csv(self, csv_path: str, has_labels: bool = False):
        """
        Predict from a single CSV file.

        Expected format:
            Columns: distance_km, CT1IA, CT1IB, CT1IC, S1) BUS1UA, S1) BUS1UB, S1) BUS1UC
            Rows   : SEQ_LENGTH time steps (e.g. 400)
        """
        print(f"\nLoading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        if has_labels:
            distance_true = float(df[DISTANCE_COL].iloc[0])
            signals = df[SIGNAL_COLS].values.astype(np.float32)   # (SEQ_LENGTH, 6)
        else:
            distance_true = None
            signals = df.values.astype(np.float32)

        # (SEQ_LENGTH, NUM_CHANNELS) -> (NUM_CHANNELS, SEQ_LENGTH)
        signals = signals.T
        print(f"Signal shape: {signals.shape}  (channels x time steps)")

        # Per-channel normalization using saved scalers
        # scalers['signal'] is a list of NUM_CHANNELS StandardScaler objects
        signal_scalers = self.scalers.get('signal')
        if signal_scalers:
            for ch_idx, scaler in enumerate(signal_scalers):
                signals[ch_idx] = scaler.transform(
                    signals[ch_idx].reshape(-1, 1)
                ).flatten()

        # (NUM_CHANNELS, SEQ_LENGTH) -> (1, NUM_CHANNELS, SEQ_LENGTH)
        tensor = torch.FloatTensor(signals).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(tensor).cpu().numpy().flatten()[0]

        # Denormalize prediction using saved distance MinMaxScaler
        dist_scaler = self.scalers.get('distance')
        if dist_scaler:
            prediction = float(dist_scaler.inverse_transform([[pred_norm]])[0][0])
        else:
            prediction = float(pred_norm)

        print(f"\n{'='*50}")
        print("PREDICTION RESULT")
        print(f"{'='*50}")
        print(f"  Predicted distance : {prediction:.4f} km")
        if distance_true is not None:
            error = abs(distance_true - prediction)
            print(f"  True distance      : {distance_true:.4f} km")
            print(f"  Absolute error     : {error:.4f} km")
        print(f"{'='*50}\n")

        return {'prediction': prediction, 'true_distance': distance_true}

    def predict_from_npy(self, npy_path: str):
        print(f"Loading signals from {npy_path}...")
        signals = np.load(npy_path).astype(np.float32)
        print(f"Loaded signals with shape: {signals.shape}")

        if signals.ndim == 2:
            signals = signals.T  # (SEQ_LENGTH, NUM_CHANNELS) -> (NUM_CHANNELS, SEQ_LENGTH)

        signal_scalers = self.scalers.get('signal')
        if signal_scalers:
            for ch_idx, scaler in enumerate(signal_scalers):
                signals[ch_idx] = scaler.transform(
                    signals[ch_idx].reshape(-1, 1)
                ).flatten()

        tensor = torch.FloatTensor(signals).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.model(tensor).cpu().numpy().flatten()[0]

        dist_scaler = self.scalers.get('distance')
        prediction = float(dist_scaler.inverse_transform([[pred_norm]])[0][0]) if dist_scaler else float(pred_norm)

        print(f"Predicted distance: {prediction:.4f} km")
        return prediction

    def save_predictions(self, results: dict, output_path: str):
        pd.DataFrame([results]).to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Predict Fault Distance')
    parser.add_argument('--model', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--csv', type=str, default=None, help='Path to CSV oscillogram file')
    parser.add_argument('--signal', type=str, default=None, help='Path to .npy signal file')
    parser.add_argument('--has-labels', action='store_true',
                        help='CSV has distance_km column (enables error reporting)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    if not args.csv and not args.signal:
        print("Error: provide either --csv or --signal")
        return

    predictor = FaultDistancePredictor(args.model, device=args.device)

    if args.csv:
        results = predictor.predict_from_csv(args.csv, has_labels=args.has_labels)
        if args.output:
            predictor.save_predictions(results, args.output)
    elif args.signal:
        predictor.predict_from_npy(args.signal)


if __name__ == '__main__':
    main()
