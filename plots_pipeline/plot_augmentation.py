"""
Визуализация аугментации данных.
"""

import numpy as np
from .plot_engine import setup_figure, save_fig, PHASE_COLORS_LIST, annotate_info


def plot_time_shift(time_ms, original, shifted_left, shifted_right, shift_amount,
                    output_path):
    """
    Визуализация временного сдвига (time-shift augmentation).

    Parameters
    ----------
    time_ms : np.ndarray
        Время в мс.
    original : np.ndarray, shape (6, T)
        Исходный сигнал.
    shifted_left : np.ndarray, shape (6, T)
        Сдвиг влево.
    shifted_right : np.ndarray, shape (6, T)
        Сдвиг вправо.
    shift_amount : int
        Величина сдвига (в отсчётах).
    output_path : str
        Путь для сохранения.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)
    labels = ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C']
    units = ['А', 'А', 'А', 'кВ', 'кВ', 'кВ']

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        color = PHASE_COLORS_LIST[idx % 3]
        ax.plot(time_ms, original[idx], color=color, linewidth=1.5,
                label='Исходный', alpha=0.9)
        ax.plot(time_ms, shifted_left[idx], color='#ff7f0e', linewidth=1.0,
                label=f'Сдвиг влево ({shift_amount})', alpha=0.7, linestyle='--')
        ax.plot(time_ms, shifted_right[idx], color='#2ca02c', linewidth=1.0,
                label=f'Сдвиг вправо ({shift_amount})', alpha=0.7, linestyle=':')
        ax.set_xlabel('Время, мс')
        ax.set_ylabel(f'{labels[idx]}, {units[idx]}')
        ax.set_title(f'{labels[idx]} — временной сдвиг')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Аугментация: временной сдвиг (±{shift_amount} отсчётов)',
                 fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_noise_levels(time_ms, original, noisy_dict, output_path):
    """
    Визуализация добавления гауссова шума с разными уровнями SNR.

    Parameters
    ----------
    time_ms : np.ndarray
        Время в мс.
    original : np.ndarray, shape (6, T)
        Исходный сигнал.
    noisy_dict : dict
        {snr_db: signal_array} — зашумлённые версии.
    output_path : str
        Путь для сохранения.
    """
    n_snr = len(noisy_dict)
    # Показываем только IA и UA для наглядности
    fig, axes = setup_figure(figsize=(16, 3 * (n_snr + 1)), nrows=n_snr + 1, ncols=2)

    labels = ['I_A', 'U_A']
    units = ['А', 'кВ']
    colors_snr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Исходный сигнал (верхняя строка)
    for col, (idx, lab, unit) in enumerate(zip([0, 3], labels, units)):
        ax = axes[0, col]
        ax.plot(time_ms, original[idx], color='black', linewidth=1.5, label='Исходный')
        ax.set_ylabel(f'{lab}, {unit}')
        ax.set_title(f'{lab} — исходный сигнал')
        ax.legend(loc='upper right')

    # Зашумлённые версии
    for row, (snr_db, noisy_sig) in enumerate(sorted(noisy_dict.items()), start=1):
        color = colors_snr[(row - 1) % len(colors_snr)]
        for col, (idx, lab, unit) in enumerate(zip([0, 3], labels, units)):
            ax = axes[row, col]
            ax.plot(time_ms, original[idx], color='black', linewidth=1.0,
                    label='Исходный', alpha=0.4)
            ax.plot(time_ms, noisy_sig[idx], color=color, linewidth=0.8,
                    label=f'SNR = {snr_db} дБ', alpha=0.9)
            ax.set_ylabel(f'{lab}, {unit}')
            ax.set_title(f'{lab} — SNR = {snr_db} дБ')
            ax.legend(loc='upper right')

    for row in range(n_snr + 1):
        axes[row, 0].set_xlabel('Время, мс')
        axes[row, 1].set_xlabel('Время, мс')

    fig.suptitle('Аугментация: гауссов шум с разными уровнями SNR',
                 fontsize=16, fontweight='bold', y=1.01)
    save_fig(fig, output_path)
