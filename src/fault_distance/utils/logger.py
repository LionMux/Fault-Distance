"""
Logging utilities for training and debugging.
"""

import logging
import os
from datetime import datetime


def get_logger(name):
    """
    Get logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)


def setup_logger(log_file=None, level=logging.INFO):
    """
    Setup global logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Create logs directory if needed
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        print(f"Logs will be saved to: {log_file}")

    return logger


class TrainingLogger:
    """
    Log training progress and metrics.
    """

    def __init__(self, output_dir='logs'):
        """
        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(output_dir, f'training_{timestamp}.log')

        # Initialize logger
        self.logger = setup_logger(self.log_file)

    def log_epoch(self, epoch, train_loss, val_loss, metrics=None):
        """
        Log epoch results.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dict of additional metrics
        """
        msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        if metrics:
            for key, value in metrics.items():
                msg += f" | {key}: {value:.6f}"
        self.logger.info(msg)

    def log_config(self, config):
        """
        Log configuration.

        Args:
            config: Config dict or object
        """
        self.logger.info("="*60)
        self.logger.info("Configuration")
        self.logger.info("="*60)
        if isinstance(config, dict):
            for key, value in config.items():
                self.logger.info(f"  {key}: {value}")
        else:
            for key in dir(config):
                if not key.startswith('_'):
                    value = getattr(config, key)
                    if not callable(value):
                        self.logger.info(f"  {key}: {value}")
        self.logger.info("="*60)
