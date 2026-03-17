"""
Visualization utilities for model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os


def plot_training_history(history, output_path='logs/training_history.png'):
    """
    Plot training and validation loss.
    
    Args:
        history (dict): Training history with 'train_loss' and 'val_loss'
        output_path (str): Path to save plot
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale loss plot
    ax2.semilogy(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax2.semilogy(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE, log scale)')
    ax2.set_title('Training History (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Training history saved to {output_path}")
    plt.close()


def plot_predictions(y_true, y_pred, output_path='logs/predictions.png'):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: Ground truth distances
        y_pred: Predicted distances
        output_path (str): Path to save plot
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted scatter
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax.set_xlabel('Ground Truth (km)')
    ax.set_ylabel('Predicted (km)')
    ax.set_title('Actual vs Predicted')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[0, 1]
    errors = y_true - y_pred
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    ax.set_xlabel('Error (km)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Absolute error
    ax = axes[1, 0]
    abs_errors = np.abs(errors)
    ax.scatter(y_true, abs_errors, alpha=0.5, s=10, color='green')
    ax.set_xlabel('Ground Truth (km)')
    ax.set_ylabel('Absolute Error (km)')
    ax.set_title('Absolute Error vs Distance')
    ax.grid(True, alpha=0.3)
    
    # Residuals
    ax = axes[1, 1]
    ax.scatter(y_pred, errors, alpha=0.5, s=10, color='orange')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted (km)')
    ax.set_ylabel('Residual (km)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Predictions plot saved to {output_path}")
    plt.close()


def plot_signal_samples(signals, distances, num_samples=5, output_path='logs/signal_samples.png'):
    """
    Plot sample signals from dataset.
    
    Args:
        signals: Array of signals (N, seq_length)
        distances: Array of distances
        num_samples: Number of samples to plot
        output_path (str): Path to save plot
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(signals), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        signal = signals[idx]
        distance = distances[idx]
        
        ax.plot(signal, linewidth=1)
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Signal Sample - Distance: {distance:.2f} km')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Signal samples saved to {output_path}")
    plt.close()
