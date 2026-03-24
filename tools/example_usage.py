"""
Example usage of Fault Distance estimation system.

Demonstrates:
1. Loading data
2. Training a model
3. Making predictions
4. Evaluating results
"""

import os
import numpy as np
import torch
from config import get_config
from data.dataset import FaultDataset, DataLoaderFactory
from models.cnn1d import CNN1D
from train import Trainer
from inference import FaultDistancePredictor
from utils.metrics import MetricsCalculator


def example_1_basic_training():
    """
    Example 1: Basic training with default configuration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Training")
    print("="*70)
    
    # Create config with custom parameters
    cfg = get_config(
        NUM_EPOCHS=50,         # Short training for demo
        BATCH_SIZE=32,
        MODEL_TYPE='cnn1d',
        LEARNING_RATE=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(cfg)
    
    # Train
    trainer.train()
    
    print("\n✅ Training completed!")
    return trainer


def example_2_load_and_predict():
    """
    Example 2: Load trained model and make predictions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Load and Predict")
    print("="*70)
    
    # Load predictor
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Run example_1_basic_training() first")
        return
    
    predictor = FaultDistancePredictor(checkpoint_path, device='cuda')
    
    # Example: Predict on CSV file with labels
    results = predictor.predict_from_csv(
        'data/oscillograms.csv',
        has_labels=True
    )
    
    print("\n✅ Predictions completed!")
    return results


def example_3_batch_prediction():
    """
    Example 3: Batch prediction on multiple signals.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Prediction")
    print("="*70)
    
    # Load predictor
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    predictor = FaultDistancePredictor(checkpoint_path, device='cuda')
    
    # Create dummy signals for demo
    num_signals = 10
    seq_length = 300
    signals = np.random.randn(num_signals, seq_length).astype(np.float32)
    
    print(f"\nPredicting on {num_signals} signals...")
    
    # Predict
    predictions = predictor.predict(signals, denormalize=True)
    
    print(f"\nPredictions (in km):")
    for i, dist in enumerate(predictions):
        print(f"  Signal {i:2d}: {dist:8.2f} km")
    
    print("\n✅ Batch prediction completed!")
    return predictions


def example_4_model_info():
    """
    Example 4: Print model information.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Information")
    print("="*70)
    
    cfg = get_config(NUM_EPOCHS=10)  # Not used, just for config
    
    # Build model
    model = CNN1D(
        seq_length=300,
        num_filters=64,
        kernel_size=5,
        dropout=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: CNN1D")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    print(f"\n--- Testing Forward Pass ---")
    dummy_input = torch.randn(1, 1, 300)  # (batch=1, channels=1, seq_length=300)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.6f}")
    
    return model


def example_5_data_analysis():
    """
    Example 5: Analyze dataset statistics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Data Analysis")
    print("="*70)
    
    csv_path = 'data/oscillograms.csv'
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        print("Please add your oscillogram data")
        return
    
    # Load dataset
    dataset = FaultDataset(
        csv_path,
        seq_length=300,
        normalize=False  # Get raw values
    )
    
    print(f"\nDataset Statistics (before normalization):")
    print(f"  Total samples: {len(dataset)}")
    
    # Get raw data
    distances_raw = dataset.distances if dataset.scaler_distance is None else \
        dataset.scaler_distance.inverse_transform(
            dataset.distances.reshape(-1, 1)
        ).flatten()
    
    signals_raw = dataset.signals if dataset.scaler_signal is None else \
        dataset.scaler_signal.inverse_transform(dataset.signals)
    
    # Compute statistics
    print(f"\n  Distance (km):")
    print(f"    Min:    {distances_raw.min():.2f}")
    print(f"    Max:    {distances_raw.max():.2f}")
    print(f"    Mean:   {distances_raw.mean():.2f}")
    print(f"    Std:    {distances_raw.std():.2f}")
    print(f"    Median: {np.median(distances_raw):.2f}")
    
    print(f"\n  Signal amplitude:")
    print(f"    Min:    {signals_raw.min():.6f}")
    print(f"    Max:    {signals_raw.max():.6f}")
    print(f"    Mean:   {signals_raw.mean():.6f}")
    print(f"    Std:    {signals_raw.std():.6f}")
    
    # Distance distribution
    print(f"\n  Distance distribution:")
    bins = 10
    counts, edges = np.histogram(distances_raw, bins=bins)
    for i in range(bins):
        bar_width = int(40 * counts[i] / counts.max())
        print(f"    [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {'█' * bar_width} ({counts[i]})")    
    
    return dataset


if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# Fault Distance Estimation - Example Usage")
    print("#"*70)
    
    # Uncomment the example you want to run:
    
    # Example 1: Basic training
    # example_1_basic_training()
    
    # Example 2: Load and predict
    # example_2_load_and_predict()
    
    # Example 3: Batch prediction
    # example_3_batch_prediction()
    
    # Example 4: Model information
    example_4_model_info()
    
    # Example 5: Data analysis
    # example_5_data_analysis()
    
    print("\n" + "#"*70 + "\n")
