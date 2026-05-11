---
name: fault-distance
description: Power system fault distance estimation using 1D-CNN/ResNet on PyTorch. Use when working on the Fault-Distance diploma project — processing COMTRADE oscillograms, training regression models, and evaluating distance estimation accuracy.
---

# Fault-Distance Project

Determine fault location in power transmission lines using deep learning on COMTRADE oscillogram data.

## Data

- 6 channels per sample: 3 currents (`Ia`, `Ib`, `Ic`) + 3 voltages (`Ua`, `Ub`, `Uc`)
- 400 samples per signal
- Label: `distance_km`

## Workflow

1. **Data Analysis**
   - Use `jupyter` to load CSV/COMTRADE files and visualize waveforms.
   - Inspect channel integrity and label distribution.

2. **Preprocessing**
   - Apply per-channel standardization or normalization.
   - Use stratified train/validation/test splits when possible.
   - Window extraction around the fault inception point for time-series data.

3. **Modeling**
   - Input shape: `(batch_size, 6, 400)`
   - Preferred architectures: 1D-CNN, ResNet1D with SE blocks
   - Output: regression head predicting distance in km

4. **Metrics**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)
   - Visualize predictions vs ground truth

5. **Experiments**
   - Log loss and MAE per epoch.
   - Compare model variants systematically.

## Project Structure

```
Fault-Distance/
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_training/
├── models/
├── notebooks/
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
├── README.md
└── requirements.txt
```
