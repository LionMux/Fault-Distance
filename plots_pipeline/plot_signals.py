"""
Графики осциллограмм и спектров.
"""

import numpy as np
from .plot_engine import setup_figure, save_fig, PHASE_COLORS_LIST, annotate_info


def plot_oscillogram_6ch(time_ms, signals, channel_labels, units, title,
                         output_path, t0_ms=None):
    """
    Построить 6 каналов осциллограммы в сетке 2×3.

    Parameters
    ----------
    time_ms : np.ndarray, shape (T,)
        Время в миллисекундах.
    signals : np.ndarray, shape (6, T)
        Сигналы: [IA, IB, IC, UA, UB, UC].
    channel_labels : list[str]
        Подписи каналов, например ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C'].
    units : list[str]
        Единицы измерения, например ['А', 'А', 'А', 'кВ', 'кВ', 'кВ'].
    title : str
        Заголовок фигуры.
    output_path : str
        Путь для сохранения PNG.
    t0_ms : float, optional
        Момент КЗ в мс — отметить вертикальной линией.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        color = PHASE_COLORS_LIST[idx % 3]
        ax.plot(time_ms, signals[idx], color=color, linewidth=1.2, label=channel_labels[idx])
        ax.set_xlabel('Время, мс')
        ax.set_ylabel(f'{channel_labels[idx]}, {units[idx]}')
        ax.set_title(f'{channel_labels[idx]}')
        ax.legend(loc='upper right')
        if t0_ms is not None:
            ax.axvline(t0_ms, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f't₀ = {t0_ms:.1f} мс')
            ax.legend(loc='upper right')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_fft_spectrum(time, signal, fs, title, output_path, color=None):
    """
    Построить односторонний спектр амплитуд сигнала.

    Parameters
    ----------
    time : np.ndarray
        Временная ось (для информации).
    signal : np.ndarray, shape (T,)
        Одномерный сигнал.
    fs : float
        Частота дискретизации [Гц].
    title : str
        Заголовок.
    output_path : str
        Путь для сохранения.
    color : str, optional
        Цвет линии.
    """
    fig, axes = setup_figure(figsize=(12, 5), nrows=1, ncols=2)
    ax_time = axes[0]
    ax_freq = axes[1]

    # Временной график
    time_ms = np.arange(len(signal)) / fs * 1000.0
    ax_time.plot(time_ms, signal, color=color or '#1f77b4', linewidth=1.0)
    ax_time.set_xlabel('Время, мс')
    ax_time.set_ylabel('Амплитуда')
    ax_time.set_title('Сигнал во времени')

    # FFT
    N = len(signal)
    yf = np.fft.rfft(signal)
    xf = np.fft.rfftfreq(N, 1.0 / fs)
    amplitude = np.abs(yf) / N
    amplitude[1:-1] *= 2
    # В дБ относительно максимума
    amp_db = 20 * np.log10(amplitude + 1e-12)
    amp_db -= amp_db.max()

    ax_freq.plot(xf, amp_db, color=color or '#1f77b4', linewidth=1.2)
    ax_freq.set_xlabel('Частота, Гц')
    ax_freq.set_ylabel('Амплитуда, дБ')
    ax_freq.set_title('Амплитудный спектр')
    ax_freq.set_xlim(0, min(fs / 2, 500))  # показываем до 500 Гц
    ax_freq.axvline(50, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label='50 Гц')
    ax_freq.axvline(100, color='orange', linestyle=':', linewidth=1.0, alpha=0.5, label='100 Гц')
    ax_freq.legend(loc='upper right')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    save_fig(fig, output_path)


def plot_channel_comparison(time_ms, original, processed, title, ylabel,
                            output_path, label_orig='Исходный', label_proc='Обработанный',
                            color_orig='#1f77b4', color_proc='#ff7f0e'):
    """
    Сравнение одного канала: исходный vs обработанный.
    Верхняя панель — оба сигнала, нижняя — разность.
    """
    fig, axes = setup_figure(figsize=(14, 8), nrows=2, ncols=1)
    ax_top = axes[0]
    ax_bot = axes[1]

    ax_top.plot(time_ms, original, color=color_orig, linewidth=1.2, label=label_orig, alpha=0.9)
    ax_top.plot(time_ms, processed, color=color_proc, linewidth=1.2, label=label_proc, alpha=0.9)
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(title)
    ax_top.legend(loc='upper right')

    diff = processed - original
    ax_bot.plot(time_ms, diff, color='#2ca02c', linewidth=1.0, label='Разность (обработанный − исходный)')
    ax_bot.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_bot.set_xlabel('Время, мс')
    ax_bot.set_ylabel(f'Δ {ylabel}')
    ax_bot.set_title('Разность сигналов')
    ax_bot.legend(loc='upper right')

    # RMS разности
    rms_diff = np.sqrt(np.mean(diff ** 2))
    annotate_info(ax_bot, f'RMS разности = {rms_diff:.4f}', loc='upper right')

    save_fig(fig, output_path)


def plot_multi_channel_overlay(time_ms, signals, labels, colors, title,
                               output_path, ylabel='Амплитуда'):
    """
    Наложить несколько каналов на один график.
    """
    fig, axes = setup_figure(figsize=(14, 6), nrows=1, ncols=1)
    ax = axes[0]

    for sig, lab, col in zip(signals, labels, colors):
        ax.plot(time_ms, sig, color=col, linewidth=1.2, label=lab, alpha=0.85)

    ax.set_xlabel('Время, мс')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    save_fig(fig, output_path)
