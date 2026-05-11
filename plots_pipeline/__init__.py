"""
Модульная система визуализации для диплома.
"""

from .plot_engine import ensure_dir, save_fig, setup_figure, annotate_info, add_vertical_line
from .plot_signals import plot_oscillogram_6ch, plot_fft_spectrum, plot_channel_comparison, plot_multi_channel_overlay
from .plot_preprocessing import plot_dc_removal, plot_t0_detection, plot_symseq, plot_normalization
from .plot_augmentation import plot_time_shift, plot_noise_levels

__all__ = [
    'ensure_dir', 'save_fig', 'setup_figure', 'annotate_info', 'add_vertical_line',
    'plot_oscillogram_6ch', 'plot_fft_spectrum', 'plot_channel_comparison', 'plot_multi_channel_overlay',
    'plot_dc_removal', 'plot_t0_detection', 'plot_symseq', 'plot_normalization',
    'plot_time_shift', 'plot_noise_levels',
]
