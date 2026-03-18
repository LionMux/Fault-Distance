"""
Inference script for predicting fault distance from oscillograms.

Usage:
    python inference.py --model checkpoints/best_model.pth --csv data/test_signals.csv
    python inference.py --model checkpoints/best_model.pth --signal path/to/signal.npy
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from config import Config
from models.cnn1d import CNN1D, DilatedCNN1D
from models.resnet1d import ResNet1D
from utils.metrics import MetricsCalculator


class FaultDistancePredictor:
    """
    Predict fault distance from oscillograms.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path (str): Path to saved checkpoint
            device (str): 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.cfg = checkpoint.get('config', Config())
        self.scalers = checkpoint.get('scalers', {})
        
        # Load model
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded. Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def _build_model(self):
        """Build model based on saved config."""
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
    
    def predict(self, signals: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Predict fault distance from signals.
        
        Args:
            signals (np.ndarray): Signal data
                - Shape (N, 300) if N signals
                - Shape (300,) if single signal
            denormalize (bool): Denormalize output to km
        
        Returns:
            np.ndarray: Predicted distances in km
        """
        # Handle single signal
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        # Check shape
        if signals.shape[1] != self.cfg.SEQ_LENGTH:
            raise ValueError(
                f"Expected signal length {self.cfg.SEQ_LENGTH}, "
                f"got {signals.shape[1]}"
            )
        
        # Normalize if scaler available
        if self.scalers.get('signal'):
            signals = self.scalers['signal'].transform(signals)
        
        # Convert to tensor: (N, seq_length) -> (N, 1, seq_length)
        signals_tensor = torch.FloatTensor(signals).unsqueeze(1).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(signals_tensor)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy().flatten()
        
        # Denormalize if scaler available
        if denormalize and self.scalers.get('distance'):
            predictions = self.scalers['distance'].inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
        
        return predictions
    
    def predict_from_csv(self, csv_path: str, has_labels: bool = False):
        """
        Predict from CSV file.
        
        Args:
            csv_path (str): Path to CSV file
                - Column 0: Fault distance (if has_labels=True)
                - Columns 1-300: Signal values (if has_labels=True)
                - Columns 0-299: Signal values (if has_labels=False)
            has_labels (bool): Whether first column is fault distance
        
        Returns:
            dict: Results including distances, predictions, errors (if has_labels)
        """
        print(f"\nLoading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if has_labels:
            distances_true = df.iloc[:, 0].values.astype(np.float32)
            signals = df.iloc[:, 1:].values.astype(np.float32)
        else:
            distances_true = None
            signals = df.values.astype(np.float32)
        
        print(f"Loaded {signals.shape[0]} signals with shape {signals.shape[1]}")
        
        # Predict
        predictions = self.predict(signals, denormalize=True)
        
        results = {
            'predictions': predictions,
            'signals': signals
        }
        
        # Calculate metrics if labels available
        if distances_true is not None:
            results['true_distances'] = distances_true
            
            # Denormalize ground truth if needed
            if self.scalers.get('distance'):
                distances_true = self.scalers['distance'].inverse_transform(
                    distances_true.reshape(-1, 1)
                ).flatten()
                results['true_distances'] = distances_true
            
            metrics = MetricsCalculator.regression_metrics(distances_true, predictions)
            results['metrics'] = metrics
            
            print(f"\n{'='*60}")
            print("PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"MAE:  {metrics['mae']:.4f} km")
            print(f"RMSE: {metrics['rmse']:.4f} km")
            print(f"R\u00b2:   {metrics['r2']:.6f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
            print(f"{'='*60}\n")
            
            # Show sample predictions
            print("Sample Predictions:")
            print(f"{'Index':<10} {'True (km)':<15} {'Pred (km)':<15} {'Error (km)':<15}")
            print("-" * 55)
            for i in range(min(10, len(distances_true))):
                error = abs(distances_true[i] - predictions[i])
                print(f"{i:<10} {distances_true[i]:<15.4f} {predictions[i]:<15.4f} {error:<15.4f}")
        else:
            print(f"\nPredicted {len(predictions)} distances:")
            for i, dist in enumerate(predictions[:10]):
                print(f"  Signal {i}: {dist:.4f} km")
        
        return results
    
    def predict_from_npy(self, npy_path: str):
        """
        Predict from .npy file containing signal(s).
        
        Args:
            npy_path (str): Path to .npy file
        
        Returns:
            np.ndarray: Predicted distances
        """
        print(f"Loading signals from {npy_path}...")
        signals = np.load(npy_path)
        
        print(f"Loaded signals with shape: {signals.shape}")
        
        predictions = self.predict(signals, denormalize=True)
        
        print(f"\nPredicted {len(predictions)} distances:")
        for i, dist in enumerate(predictions):
            print(f"  Signal {i}: {dist:.4f} km")
        
        return predictions
    
    def save_predictions(self, results: dict, output_path: str):
        """
        Save predictions to CSV.
        
        Args:
            results (dict): Results from predict_from_csv or similar
            output_path (str): Output CSV path
        """
        data = {
            'Predicted_Distance_km': results['predictions']
        }
        
        if 'true_distances' in results:
            data['True_Distance_km'] = results['true_distances']
            data['Error_km'] = np.abs(results['true_distances'] - results['predictions'])
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to {output_path}")


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description='Predict Fault Distance')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV with signals')
    parser.add_argument('--signal', type=str, default=None,
                       help='Path to .npy signal file')
    parser.add_argument('--has-labels', action='store_true',
                       help='CSV first column is fault distance')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.csv and not args.signal:
        print("Error: Provide either --csv or --signal")
        return
    
    # Load predictor
    predictor = FaultDistancePredictor(args.model, device=args.device)
    
    # Predict
    if args.csv:
        results = predictor.predict_from_csv(args.csv, has_labels=args.has_labels)
        
        # Save predictions
        if args.output:
            predictor.save_predictions(results, args.output)
    
    elif args.signal:
        predictor.predict_from_npy(args.signal)


if __name__ == '__main__':
    main()
