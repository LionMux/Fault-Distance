# Fault Distance Estimation from Power System Oscillograms

**Deep Learning for Short-Circuit Fault Localization in Power Transmission Lines**

This project implements a 1D Convolutional Neural Network (CNN) for estimating the distance to short-circuit faults in power transmission systems using oscillogram data.

## 🎨 Project Overview

### Problem Statement
When a short-circuit fault occurs in a power transmission line, accurately determining the **fault distance** (distance from the measurement point to the fault location) is critical for rapid fault localization and system recovery.

Traditional methods:
- ❌ Impedance-based calculations (require system parameters)
- ❌ Wave arrival detection (sensitive to noise)
- ❌ Traveling wave analysis (complex algorithms)

**Our solution**: End-to-end deep learning approach using raw oscillogram signals.

### Why 1D CNN?
- **Temporal dependencies**: Short-circuit signals contain characteristic patterns in time domain
- **Translation invariance**: Fault features are consistent across different distances
- **Computational efficiency**: Suitable for real-time fault location systems
- **Minimal preprocessing**: Learns features automatically from raw signals

---

## 📖 Dataset Format

### CSV Structure
Each row contains one fault oscillogram:

```csv
distance_km,sample_0,sample_1,...,sample_399
12.5,0.001,0.002,...,-0.001
25.3,0.002,-0.001,...,0.003
...
```

**Columns:**
- **Column 0**: Fault distance (km) - **Label**
- **Columns 1-400**: Signal samples - **Features** (400 time points)

### Signal Specifications
- **Duration**: 400 ms (100 ms pre-fault + 300 ms post-fault)
- **Sampling rate**: 1 kHz (typical for SCADA systems)
- **Channel**: Single-phase current or voltage
- **Range**: Typically [-1.0, 1.0] A or V (normalized)

### Data Placement
Place your CSV file at:
```
Fault-Distance/
├─ data/
│  └─ oscillograms.csv  ← Your CSV here!
└─ ...
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/LionMux/Fault-Distance.git
cd Fault-Distance
pip install -r requirements.txt
```

### Training
```bash
# Default configuration
python train.py

# Custom hyperparameters
python train.py --epochs 200 --batch-size 64 --model cnn1d --lr 0.0005

# Available models: cnn1d, dilated_cnn1d, resnet1d
python train.py --model dilated_cnn1d --epochs 150
```

### Inference (Prediction)
```bash
# Predict on CSV file with labels (for evaluation)
python inference.py --model checkpoints/best_model.pth \
                    --csv data/test_signals.csv \
                    --has-labels \
                    --output predictions.csv

# Predict on new signals without labels
python inference.py --model checkpoints/best_model.pth \
                    --csv data/new_signals.csv

# Predict on single signal (.npy file)
python inference.py --model checkpoints/best_model.pth \
                    --signal path/to/signal.npy
```

---

## 🏗️ Architecture

### Model Options

#### 1. CNN1D (Default)
Simple 1D CNN for fast training:
```
Input (1, 400)
  ⬇
 Conv1d(1 → 64, k=5) + BN + ReLU + MaxPool
  ⬇
 Conv1d(64 → 128, k=5) + BN + ReLU + MaxPool
  ⬇
 Conv1d(128 → 256, k=5) + BN + ReLU + MaxPool
  ⬇
 Flatten → FC(256, 128, 1)
  ⬇
Output: distance (1,)
```

**Parameters**: ~500k
**Inference time**: <1ms on GPU

#### 2. DilatedCNN1D
With dilated convolutions for larger receptive field:
```
Cascaded dilated convolutions (dilations: 1, 2, 4, 8)
⬇
Larger receptive field with fewer layers
⬇
Better for capturing long-range signal dependencies
```

#### 3. ResNet1D
Residual connections for deeper networks:
```
Residual blocks with skip connections
⬇
Easier to train deeper models
⬇
Better gradient flow during backprop
```

---

## 🛠️ Configuration

Edit `config.py` to customize:

```python
CFG = Config(
    # Model
    MODEL_TYPE='cnn1d',        # 'cnn1d', 'dilated_cnn1d', 'resnet1d'
    NUM_FILTERS=64,            # Initial filters (32-256)
    KERNEL_SIZE=5,             # Conv kernel size (3-7)
    DROPOUT=0.3,               # Dropout rate (0.1-0.5)
    
    # Training
    BATCH_SIZE=32,             # Batch size (16-128)
    NUM_EPOCHS=100,            # Max epochs
    LEARNING_RATE=0.001,       # Initial LR (1e-4 to 1e-2)
    OPTIMIZER='adam',          # 'adam', 'adamw', 'sgd'
    
    # Scheduler
    SCHEDULER_TYPE='cosine',   # 'cosine', 'linear', 'exponential'
    WARMUP_EPOCHS=5,           # Warmup period
    
    # Loss & Regularization
    LOSS_FUNCTION='mse',       # 'mse', 'mae', 'smooth_l1'
    WEIGHT_DECAY=1e-5,         # L2 regularization
    GRADIENT_CLIP=1.0,         # Gradient clipping
    
    # Early Stopping
    EARLY_STOPPING=True,
    PATIENCE=15,               # Stop if no improvement for N epochs
    
    # Data
    TRAIN_SPLIT=0.8,           # 80% train, 20% test
    NORMALIZE_DATA=True,       # StandardScaler + MinMaxScaler for distance
)
```

---

## 📁 Project Structure

```
Fault-Distance/
├─ data/
│  ├─ dataset.py              # PyTorch Dataset class
│  ├─ preprocessing.py        # Data preprocessing & augmentation
│  └─ oscillograms.csv        # 📋 Your CSV data
├─ models/
│  ├─ cnn1d.py               # 1D CNN architectures
│  ├─ resnet1d.py            # ResNet variants
│  └─ blocks.py              # Residual & Dense blocks
├─ utils/
│  ├─ metrics.py             # MAE, RMSE, R² calculations
│  ├─ logger.py              # Training logging
│  └─ plots.py               # Visualization utilities
├─ config.py                # Configuration settings
├─ train.py                # Training script
├─ inference.py            # Prediction script
├─ requirements.txt        # Dependencies
└─ README.md               # This file
```

---

## 📊 Training Metrics

### Monitoring
During training, the system logs:
- **MAE** (Mean Absolute Error): Average prediction error in km
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (Coefficient of Determination): Model fit quality (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Percentage error

### Typical Performance
On realistic power system data:
- **MAE**: 0.5-2.0 km (depending on transmission line length)
- **RMSE**: 0.8-3.0 km
- **R²**: 0.92-0.98

---

## 🤖 Outputs

### Checkpoints
Saved in `checkpoints/`:
```
checkpoints/
├─ best_model.pth         # Best validation performance
└─ checkpoint_epoch_*.pth  # Periodic snapshots
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Scalers (for denormalization)
- Training configuration

### Results
Generated in `logs/`:
```
logs/
├─ training_history.png    # Loss curves (linear + log scale)
├─ predictions.png         # Actual vs predicted scatter plot
├─ signal_samples.png      # Example oscillograms
└─ training_*.log         # Detailed training logs
```

---

## 🔧 Data Preparation Tips

### Normalization (Automatic)
The system automatically:
1. **Signal normalization**: StandardScaler (zero mean, unit variance)
2. **Distance normalization**: MinMaxScaler (0-1 range)
3. **Train/test split**: 80/20 random split with fixed seed

### Data Requirements
- **Minimum samples**: 100+ (ideally 1000+)
- **Distance range**: Cover full range (0 to max line length)
- **Signal quality**: Remove obvious corrupted samples
- **Balanced data**: Similar distribution across distance ranges

### Data Augmentation (Optional)
Enable in preprocessing:
```python
from data.preprocessing import DataAugmentation

# Add noise
signal_noisy = DataAugmentation.add_gaussian_noise(signal, noise_std=0.01)

# Time shift
signal_shifted = DataAugmentation.time_shift(signal, max_shift=10)

# Amplitude scaling
signal_scaled = DataAugmentation.amplitude_scaling(signal, scale_range=(0.8, 1.2))
```

---

## 📚 Advanced Usage

### Custom Training Loop
```python
from train import Trainer
from config import get_config

cfg = get_config(
    NUM_EPOCHS=200,
    BATCH_SIZE=64,
    MODEL_TYPE='dilated_cnn1d',
    LEARNING_RATE=0.0005
)

trainer = Trainer(cfg)
trainer.train()
```

### Batch Prediction
```python
from inference import FaultDistancePredictor
import numpy as np

predictor = FaultDistancePredictor(
    'checkpoints/best_model.pth',
    device='cuda'
)

# Load signals (N, 400)
signals = np.load('signals.npy')

# Predict
distances = predictor.predict(signals, denormalize=True)
print(distances)  # (N,) array of distances in km
```

---

## ⚠️ Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` (32 → 16)
- Use simpler model: `cnn1d` instead of `resnet1d`
- Reduce `NUM_FILTERS`

### Poor Convergence
- Check data normalization
- Ensure distance range is meaningful
- Try different learning rate: `0.0001` to `0.01`
- Increase patience for early stopping

### Low Accuracy on Test Set
- Collect more training data
- Check test data distribution (should match training)
- Try longer training (`NUM_EPOCHS=300`)
- Experiment with different models

---

## 📝 License

MIT License - see LICENSE file for details

---

**Made with ❤️ for power systems engineers and ML practitioners** ⚡
