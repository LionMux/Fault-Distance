"""
Publication-quality comparison plots for model evaluation.
Russian labels, DPI=300, professional styling.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Shared style with utils/plots.py ---
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

# Palette for up to 6 models
_MODEL_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
]


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def _model_label(model_type: str) -> str:
    """Return display label for model type."""
    mapping = {
        'cnn1d': 'CNN1D',
        'dilated_cnn1d': 'DilatedCNN1D',
        'resnet1d': 'ResNet1D',
    }
    return mapping.get(model_type, model_type.upper())


def plot_comparison_training_loss(histories_dict, output_path):
    """
    Plot training and validation loss curves for multiple models on a single figure.

    Args:
        histories_dict: dict {model_type: history dict with 'train_loss' and 'val_loss' lists}
        output_path: path to save PNG
    """
    _ensure_dir(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # Left: linear scale
    ax = axes[0]
    for idx, (model_type, history) in enumerate(histories_dict.items()):
        epochs = np.arange(1, len(history['train_loss']) + 1)
        color = _MODEL_COLORS[idx % len(_MODEL_COLORS)]
        ax.plot(epochs, history['train_loss'], color=color, linewidth=2,
                marker='o', markersize=3, label=_model_label(model_type))
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Кривые потерь на обучении (линейный масштаб)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Right: log scale
    ax = axes[1]
    for idx, (model_type, history) in enumerate(histories_dict.items()):
        epochs = np.arange(1, len(history['train_loss']) + 1)
        color = _MODEL_COLORS[idx % len(_MODEL_COLORS)]
        ax.semilogy(epochs, history['train_loss'], color=color, linewidth=2,
                    marker='o', markersize=3, label=_model_label(model_type))
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss (MSE, log scale)')
    ax.set_title('Кривые потерь на обучении (логарифмический масштаб)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Comparison training loss saved to {output_path} (DPI=300)")
    plt.close()


def plot_comparison_mae(histories_dict, output_path):
    """
    Plot validation MAE curves for multiple models on a single figure.

    Args:
        histories_dict: dict {model_type: history dict with 'val_mae' list}
        output_path: path to save PNG
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    for idx, (model_type, history) in enumerate(histories_dict.items()):
        if 'val_mae' not in history or len(history['val_mae']) == 0:
            continue
        epochs = np.arange(1, len(history['val_mae']) + 1)
        color = _MODEL_COLORS[idx % len(_MODEL_COLORS)]
        ax.plot(epochs, history['val_mae'], color=color, linewidth=2,
                marker='s', markersize=3, label=_model_label(model_type))

    ax.set_xlabel('Эпоха')
    ax.set_ylabel('MAE (км)')
    ax.set_title('Средняя абсолютная ошибка на валидации')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Comparison MAE saved to {output_path} (DPI=300)")
    plt.close()


def plot_comparison_metrics_bar(results_dict, output_path):
    """
    Bar chart comparing MAE, RMSE, R^2, MAPE across models.

    Args:
        results_dict: dict {model_type: {'mae': float, 'rmse': float, 'r2': float, 'mape': float}}
        output_path: path to save PNG
    """
    _ensure_dir(output_path)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    models = list(results_dict.keys())
    labels = [_model_label(m) for m in models]
    colors = [_MODEL_COLORS[i % len(_MODEL_COLORS)] for i in range(len(models))]

    metrics = [
        ('mae', 'MAE (км)', '{:.4f}'),
        ('rmse', 'RMSE (км)', '{:.4f}'),
        ('r2', 'R^2', '{:.4f}'),
        ('mape', 'MAPE (%)', '{:.2f}'),
    ]

    for ax, (key, title, fmt) in zip(axes, metrics):
        values = [results_dict[m].get(key, 0) for m in models]
        bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
        offset = max(values) * 0.01 if max(values) > 0 else 0.01
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    fmt.format(val), ha='center', va='bottom', fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        if len(labels) > 3:
            ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Comparison metrics bar chart saved to {output_path} (DPI=300)")
    plt.close()


def plot_comparison_error_boxplot(results_dict, output_path):
    """
    Boxplot of absolute errors for each model.

    Args:
        results_dict: dict {model_type: {'errors': np.ndarray or list}}
        output_path: path to save PNG
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    models = []
    error_lists = []
    for model_type in results_dict:
        errors = results_dict[model_type].get('errors')
        if errors is not None and len(errors) > 0:
            models.append(_model_label(model_type))
            error_lists.append(np.array(errors).flatten())

    if len(error_lists) == 0:
        ax.text(0.5, 0.5, 'Нет данных об ошибках', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Распределение абсолютных ошибок')
    else:
        bp = ax.boxplot(error_lists, labels=models, patch_artist=True)
        for patch, color in zip(bp['boxes'], _MODEL_COLORS[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Абсолютная ошибка (км)')
        ax.set_title('Распределение абсолютных ошибок по моделям')
        ax.grid(True, alpha=0.3, axis='y')
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Comparison error boxplot saved to {output_path} (DPI=300)")
    plt.close()


def plot_comparison_table(results_dict, output_path):
    """
    PNG table with model comparison results.

    Args:
        results_dict: dict {model_type: {'mae': float, 'rmse': float, 'r2': float, 'mape': float}}
        output_path: path to save PNG
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(12, 0.8 + 0.5 * len(results_dict)))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    models = list(results_dict.keys())

    rows = []
    for m in models:
        r = results_dict[m]
        rows.append([
            _model_label(m),
            f"{r.get('mae', 0):.4f}",
            f"{r.get('rmse', 0):.4f}",
            f"{r.get('r2', 0):.4f}",
            f"{r.get('mape', 0):.2f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=['Модель', 'MAE (км)', 'RMSE (км)', 'R^2', 'MAPE (%)'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Header styling
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')

    ax.set_title('Сравнение моделей', fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Comparison table saved to {output_path} (DPI=300)")
    plt.close()
