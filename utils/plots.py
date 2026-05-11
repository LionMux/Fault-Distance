"""
Publication-quality visualization utilities for model analysis.
Diploma-ready plots with Russian labels, DPI=300, professional styling.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# ---------------------------------------------------------------------------
# Font setup — try Times New Roman, fall back to DejaVu Serif
# ---------------------------------------------------------------------------
_font_ok = False
for _family in ['Times New Roman', 'DejaVu Serif', 'serif']:
    try:
        plt.rcParams['font.family'] = _family
        plt.rcParams['axes.unicode_minus'] = False
        _font_ok = True
        break
    except Exception:
        continue

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

# Professional colour palette
_COLOR_TRAIN = '#1f77b4'      # muted blue
_COLOR_VAL   = '#ff7f0e'      # muted orange
_COLOR_BEST  = '#2ca02c'      # green
_COLOR_ERROR = '#d62728'      # red
_COLOR_RESID = '#9467bd'      # purple


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def plot_training_history(history, output_path='logs/training_history.png'):
    """
    Plot training and validation loss + MAE.
    Two rows: linear scale and log scale.
    """
    _ensure_dir(output_path)
    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # --- Top-left: Loss (linear) ---
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=_COLOR_TRAIN, linewidth=2,
            marker='o', markersize=3, label='Обучение')
    ax.plot(epochs, history['val_loss'], color=_COLOR_VAL, linewidth=2,
            marker='s', markersize=3, label='Валидация')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Кривые потерь (линейный масштаб)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if len(history['val_loss']) > 0:
        best_ep = int(np.argmin(history['val_loss'])) + 1
        best_val = min(history['val_loss'])
        ax.axvline(best_ep, color=_COLOR_BEST, linestyle='--', alpha=0.6,
                   label=f'Лучшая эпоха: {best_ep}')
        ax.scatter([best_ep], [best_val], color=_COLOR_BEST, s=80, zorder=5)

    # --- Top-right: Loss (log) ---
    ax = axes[0, 1]
    ax.semilogy(epochs, history['train_loss'], color=_COLOR_TRAIN, linewidth=2,
                marker='o', markersize=3, label='Обучение')
    ax.semilogy(epochs, history['val_loss'], color=_COLOR_VAL, linewidth=2,
                marker='s', markersize=3, label='Валидация')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss (MSE, log scale)')
    ax.set_title('Кривые потерь (логарифмический масштаб)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # --- Bottom-left: MAE ---
    ax = axes[1, 0]
    if 'val_mae' in history and len(history['val_mae']) > 0:
        ax.plot(epochs, history['val_mae'], color=_COLOR_VAL, linewidth=2,
                marker='s', markersize=3, label='MAE валидации')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('MAE (км)')
        ax.set_title('Средняя абсолютная ошибка')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        best_mae_ep = int(np.argmin(history['val_mae'])) + 1
        best_mae = min(history['val_mae'])
        ax.axvline(best_mae_ep, color=_COLOR_BEST, linestyle='--', alpha=0.6)
        ax.scatter([best_mae_ep], [best_mae], color=_COLOR_BEST, s=80, zorder=5)
    else:
        ax.text(0.5, 0.5, 'MAE недоступно', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Средняя абсолютная ошибка')

    # --- Bottom-right: Learning rate (if available) ---
    ax = axes[1, 1]
    if 'lr' in history and len(history['lr']) > 0:
        ax.plot(epochs, history['lr'], color='#17becf', linewidth=2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Learning rate')
        ax.set_title('Изменение скорости обучения')
        ax.grid(True, alpha=0.3)
    else:
        # If no LR history, show convergence zoom (last 30% of epochs)
        zoom_start = max(0, int(len(epochs) * 0.7))
        ax.plot(epochs[zoom_start:], history['train_loss'][zoom_start:],
                color=_COLOR_TRAIN, linewidth=2, marker='o', markersize=3, label='Обучение')
        ax.plot(epochs[zoom_start:], history['val_loss'][zoom_start:],
                color=_COLOR_VAL, linewidth=2, marker='s', markersize=3, label='Валидация')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Сходимость (последние 30% эпох)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Training history saved to {output_path} (DPI=300)")
    plt.close()


def plot_predictions(y_true, y_pred, output_path='logs/predictions.png'):
    """
    Comprehensive prediction analysis.
    4 panels: scatter, histogram, abs error vs distance, residuals.
    """
    _ensure_dir(output_path)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor('white')

    # --- Panel 1: Actual vs Predicted ---
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, s=25, edgecolors='none', color=_COLOR_TRAIN)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.75, zorder=3, label='Идеальное совпадение')
    ax.set_xlabel('Истинное расстояние, км')
    ax.set_ylabel('Предсказанное расстояние, км')
    ax.set_title('Соответствие: истинное vs предсказанное')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    # Annotate R²
    r2 = 1 - np.sum(errors**2) / np.sum((y_true - np.mean(y_true))**2)
    ax.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Panel 2: Error distribution with normal fit ---
    ax = axes[0, 1]
    n_bins = max(15, int(np.sqrt(len(errors))))
    counts, bins, patches = ax.hist(errors, bins=n_bins, density=True,
                                     edgecolor='black', alpha=0.7, color=_COLOR_VAL)
    # Fit normal
    mu, sigma = stats.norm.fit(errors)
    x_fit = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma), 'r-', linewidth=2,
            label=f'Норм. распр.  $\mu={mu:.3f}$, $\sigma={sigma:.3f}$')
    ax.axvline(mu, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Ошибка (истинное − предсказанное), км')
    ax.set_ylabel('Плотность вероятности')
    ax.set_title('Распределение ошибок')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Absolute error vs distance ---
    ax = axes[1, 0]
    ax.scatter(y_true, abs_errors, alpha=0.6, s=25, edgecolors='none', color=_COLOR_ERROR)
    # Running median
    sort_idx = np.argsort(y_true)
    y_sorted = y_true[sort_idx]
    e_sorted = abs_errors[sort_idx]
    window = max(5, len(y_sorted) // 10)
    if len(y_sorted) >= window:
        medians = []
        centers = []
        for i in range(0, len(y_sorted) - window + 1, window // 2):
            medians.append(np.median(e_sorted[i:i+window]))
            centers.append(np.mean(y_sorted[i:i+window]))
        ax.plot(centers, medians, color='darkgreen', linewidth=2.5, label='Скользящая медиана')
    ax.set_xlabel('Истинное расстояние, км')
    ax.set_ylabel('Абсолютная ошибка, км')
    ax.set_title('Абсолютная ошибка vs расстояние')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Residuals vs Predicted ---
    ax = axes[1, 1]
    ax.scatter(y_pred, errors, alpha=0.6, s=25, edgecolors='none', color=_COLOR_RESID)
    ax.axhline(0, color='r', linestyle='--', linewidth=1.5)
    # 95% confidence band (approx)
    if len(errors) > 10:
        mu_res = np.mean(errors)
        sigma_res = np.std(errors)
        ax.axhline(mu_res + 1.96*sigma_res, color='orange', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.axhline(mu_res - 1.96*sigma_res, color='orange', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.set_xlabel('Предсказанное расстояние, км')
    ax.set_ylabel('Остаток (истинное − предсказанное), км')
    ax.set_title('График остатков')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Predictions plot saved to {output_path} (DPI=300)")
    plt.close()


def plot_metrics_summary(y_true, y_pred, output_path='logs/metrics_summary.png'):
    """
    Additional metrics visualisation:
    - CDF of absolute errors
    - Error percentiles bar chart
    - Histogram of absolute errors
    """
    _ensure_dir(output_path)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    abs_err = np.abs(y_true - y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    # CDF
    ax = axes[0]
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err) * 100
    ax.plot(sorted_err, cdf, color=_COLOR_TRAIN, linewidth=2.5)
    ax.axvline(np.median(abs_err), color='r', linestyle='--', linewidth=1.5,
               label=f'Медиана: {np.median(abs_err):.3f} км')
    ax.axhline(95, color='orange', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.set_xlabel('Абсолютная ошибка, км')
    ax.set_ylabel('Накопленная частота, %')
    ax.set_title('CDF абсолютной ошибки')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Percentiles
    ax = axes[1]
    percentiles = [50, 75, 90, 95, 99]
    vals = [np.percentile(abs_err, p) for p in percentiles]
    bars = ax.bar([f'{p}%' for p in percentiles], vals, color=_COLOR_VAL,
                  edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Абсолютная ошибка, км')
    ax.set_title('Процентили абсолютной ошибки')
    ax.grid(True, alpha=0.3, axis='y')

    # Error bands
    ax = axes[2]
    bins = [0, 0.5, 1.0, 2.0, 5.0, np.inf]
    labels = ['<0.5', '0.5–1', '1–2', '2–5', '>5']
    counts = []
    for i in range(len(bins)-1):
        counts.append(np.sum((abs_err >= bins[i]) & (abs_err < bins[i+1])))
    counts = np.array(counts)
    pct = counts / len(abs_err) * 100
    bars = ax.bar(labels, pct, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd'],
                  edgecolor='black', alpha=0.8)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{p:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Доля выборки, %')
    ax.set_xlabel('Диапазон ошибки, км')
    ax.set_title('Распределение ошибок по диапазонам')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Metrics summary saved to {output_path} (DPI=300)")
    plt.close()


def plot_signal_samples(signals, distances, num_samples=5, output_path='logs/signal_samples.png'):
    """
    Plot sample signals from dataset.
    """
    _ensure_dir(output_path)
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))
    if num_samples == 1:
        axes = [axes]

    indices = np.random.choice(len(signals), num_samples, replace=False)
    for i, idx in enumerate(indices):
        ax = axes[i]
        signal = signals[idx]
        distance = distances[idx]
        ax.plot(signal, linewidth=1, color=_COLOR_TRAIN)
        ax.set_ylabel('Амплитуда')
        ax.set_title(f'Осциллограмма — расстояние: {distance:.2f} км')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Отсчёт')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Signal samples saved to {output_path} (DPI=300)")
    plt.close()
