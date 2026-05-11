"""
Визуализация этапов предобработки осциллограмм.
"""

import numpy as np
from scipy import signal as scipy_signal
from .plot_engine import setup_figure, save_fig, PHASE_COLORS_LIST, annotate_info, add_vertical_line


def plot_dc_removal(time_ms, sig_before, sig_after, fs, output_path):
    """
    Визуализация удаления апериодической (DC) составляющей.

    Parameters
    ----------
    time_ms : np.ndarray
        Время в мс.
    sig_before : np.ndarray, shape (6, T)
        Сигналы до обработки.
    sig_after : np.ndarray, shape (6, T)
        Сигналы после remove_dc_period.
    fs : float
        Частота дискретизации [Гц].
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
        ax.plot(time_ms, sig_before[idx], color=color, linewidth=1.0,
                label='Исходный', alpha=0.6)
        ax.plot(time_ms, sig_after[idx], color=color, linewidth=1.5,
                label='После удаления DC', linestyle='-')
        ax.set_xlabel('Время, мс')
        ax.set_ylabel(f'{labels[idx]}, {units[idx]}')
        ax.set_title(f'{labels[idx]} — удаление DC-составляющей')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Этап 1: Удаление апериодической (DC) составляющей', fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_t0_detection(time_ms, current, d4, delta_i, t0_idx, fs, output_path):
    """
    Визуализация детекции момента КЗ (t0).

    Parameters
    ----------
    time_ms : np.ndarray
        Время в мс.
    current : np.ndarray, shape (T,)
        Ток фазы A (или суммарный ток).
    d4 : np.ndarray, shape (T,)
        Четвёртая разность D4(k).
    delta_i : np.ndarray, shape (T,)
        Цикл-разностный индекс Delta_i(k).
    t0_idx : int
        Индекс момента КЗ.
    fs : float
        Частота дискретизации [Гц].
    output_path : str
        Путь для сохранения.
    """
    fig, axes = setup_figure(figsize=(14, 10), nrows=3, ncols=1)
    ax_current = axes[0]
    ax_d4 = axes[1]
    ax_delta = axes[2]

    t0_ms = time_ms[t0_idx] if t0_idx < len(time_ms) else time_ms[-1]

    # Ток с маркером t0
    ax_current.plot(time_ms, current, color='#1f77b4', linewidth=1.2)
    ax_current.axvline(t0_ms, color='red', linestyle='--', linewidth=2.0, alpha=0.8,
                       label=f't₀ = {t0_ms:.1f} мс')
    ax_current.set_ylabel('I_A, А')
    ax_current.set_title('Ток фазы A с отметкой момента КЗ')
    ax_current.legend(loc='upper left')

    # D4(k)
    ax_d4.plot(time_ms, np.abs(d4), color='#9467bd', linewidth=1.0)
    ax_d4.axvline(t0_ms, color='red', linestyle='--', linewidth=2.0, alpha=0.8)
    ax_d4.set_ylabel('|D₄(k)|')
    ax_d4.set_title('Четвёртая разность D₄(k) — грубое определение')

    # Delta_i(k)
    ax_delta.plot(time_ms, delta_i, color='#2ca02c', linewidth=1.0)
    ax_delta.axvline(t0_ms, color='red', linestyle='--', linewidth=2.0, alpha=0.8)
    ax_delta.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_delta.set_xlabel('Время, мс')
    ax_delta.set_ylabel('Δ_i(k), А')
    ax_delta.set_title('Цикл-разностный индекс Δ_i(k) — точное определение')

    annotate_info(ax_current, f'Отсчёт t₀: {t0_idx}', loc='upper right')

    fig.suptitle('Этап 3: Детекция момента КЗ (t₀)', fontsize=16, fontweight='bold', y=1.01)
    save_fig(fig, output_path)


def plot_symseq(time_ms, symseq, output_path):
    """
    Визуализация симметричных составляющих (6 каналов).

    Parameters
    ----------
    time_ms : np.ndarray
        Время в мс.
    symseq : np.ndarray, shape (6, T)
        [|I1|, |I2|, |I0|, |U1|, |U2|, |U0|].
    output_path : str
        Путь для сохранения.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)
    labels = ['|I₁|', '|I₂|', '|I₀|', '|U₁|', '|U₂|', '|U₀|']
    units = ['А', 'А', 'А', 'кВ', 'кВ', 'кВ']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        ax.plot(time_ms, symseq[idx], color=colors[idx], linewidth=1.2)
        ax.set_xlabel('Время, мс')
        ax.set_ylabel(f'{labels[idx]}, {units[idx]}')
        ax.set_title(f'{labels[idx]} — симметричная составляющая')

    fig.suptitle('Этап 4: Симметричные составляющие (Fortescue)', fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)


def plot_normalization(time_ms, sig_before, sig_after, method, output_path):
    """
    Визуализация нормализации.

    Parameters
    ----------
    time_ms : np.ndarray
        Время в мс.
    sig_before : np.ndarray, shape (6, T)
        До нормализации.
    sig_after : np.ndarray, shape (6, T)
        После нормализации.
    method : str
        'standard' или 'minmax'.
    output_path : str
        Путь для сохранения.
    """
    fig, axes = setup_figure(figsize=(16, 10), nrows=2, ncols=3)
    axes = axes.reshape(2, 3)
    labels = ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C']
    units_before = ['А', 'А', 'А', 'кВ', 'кВ', 'кВ']
    units_after = ['у.е.', 'у.е.', 'у.е.', 'у.е.', 'у.е.', 'у.е.']

    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        color = PHASE_COLORS_LIST[idx % 3]
        ax.plot(time_ms, sig_before[idx], color=color, linewidth=1.0,
                label=f'Исходный ({units_before[idx]})', alpha=0.6)
        ax.plot(time_ms, sig_after[idx], color=color, linewidth=1.5,
                label=f'Нормализованный ({units_after[idx]})', linestyle='-')
        ax.set_xlabel('Время, мс')
        ax.set_ylabel(f'{labels[idx]}')
        ax.set_title(f'{labels[idx]} — нормализация ({method})')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'Этап 5: Нормализация сигналов (метод: {method})', fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, output_path)
