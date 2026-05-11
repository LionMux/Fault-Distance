"""
Model comparison pipeline for diploma.
Trains multiple architectures on the same data, compares metrics, and selects the best.

Usage:
    python scripts/compare_models.py --models cnn1d resnet1d dilated_cnn1d --epochs 100 --config configs/base.yaml
    python scripts/compare_models.py --models cnn1d resnet1d --epochs 2
"""

import argparse
import os
import sys
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, get_config, load_config
from train import Trainer
from utils.metrics import MetricsCalculator
from utils.plots import plot_predictions
from utils.comparison_plots import (
    plot_comparison_training_loss,
    plot_comparison_mae,
    plot_comparison_metrics_bar,
    plot_comparison_error_boxplot,
    plot_comparison_table,
)

from utils.activation_recorder import ActivationRecorder, LayerRecordMode, resolve_cnn_layer_names
from utils.activation_export import export_layer_snapshots


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple fault-distance models')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['cnn1d', 'resnet1d', 'dilated_cnn1d'],
        choices=['cnn1d', 'dilated_cnn1d', 'resnet1d'],
        help='List of models to compare',
    )
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data-dir', type=str, default=None, help='Training data directory')
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Test data directory (CSV files). If not provided, uses validation split.',
    )
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible split')

    # New: run multiple comparison rounds
    parser.add_argument('--num-runs', type=int, default=1, help='How many comparison runs to perform')
    parser.add_argument(
        '--save-all-runs',
        type=str,
        default='false',
        help='If true, save artifacts for each run (subject to improvement rule for the best folder)',
    )
    parser.add_argument(
        '--seed-offset',
        type=int,
        default=0,
        help='Additional seed offset applied per run: seed + seed-offset + run_index',
    )

    # keep backward-compat with older code that refers to args.seed_offset
    # argparse converts hyphenated names automatically, but we ensure attribute exists.
    parser.set_defaults(seed_offset=0)

    return parser.parse_args()


def run_inference_on_loader(model, loader, cfg, scalers, device):
    """Run inference on a DataLoader and return denormalized y_true, y_pred."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, distances in loader:
            signals = signals.to(device)
            preds = model(signals).cpu().numpy().flatten()
            y_true.extend(distances.numpy().flatten())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Denormalize
    scaler = scalers.get('distance')
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    elif getattr(cfg, 'NORMALIZATION_MODE', 'standard') == 'pu':
        y_true = y_true * cfg.LINE_L_KM
        y_pred = y_pred * cfg.LINE_L_KM

    return y_true, y_pred


def run_inference_on_csv_dir(checkpoint_path, test_dir, device):
    """Run inference on CSV files in directory using FaultDistancePredictor."""
    import glob
    from inference import FaultDistancePredictor

    predictor = FaultDistancePredictor(checkpoint_path, device=device)

    csv_files = sorted(glob.glob(os.path.join(test_dir, '*.csv')))
    if not csv_files:
        return None, None

    rows = []
    for fpath in csv_files:
        try:
            result = predictor.predict_from_csv(fpath, has_labels=True)
            rows.append({
                'true': result['true_distance'],
                'pred': result['prediction'],
            })
        except Exception as e:
            print(f"  [WARN] Failed on {os.path.basename(fpath)}: {e}")

    if len(rows) == 0:
        return None, None

    df = pd.DataFrame(rows)
    y_true = df['true'].dropna().values
    y_pred = df['pred'].values[:len(y_true)]
    return y_true, y_pred


def _parse_bool_str(v: str) -> bool:
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')


def _read_existing_best_maes(runs_dir: str) -> List[float]:
    """
    Expect folder names like:
      2.5917_MAE/
      1.8287_MAE__resnet1d/
    Returns list of MAE floats found.
    """
    if not os.path.isdir(runs_dir):
        return []

    maes: List[float] = []
    for name in os.listdir(runs_dir):
        full = os.path.join(runs_dir, name)
        if not os.path.isdir(full):
            continue

        if '_MAE' not in name:
            continue

        # support both patterns:
        #   "{mae}_MAE" and "{mae}_MAE__{model}"
        prefix = name.split('_MAE')[0]
        try:
            maes.append(float(prefix))
        except ValueError:
            continue

    return maes


def _format_best_folder(mae_km: float, best_model_type: str) -> str:
    # Example: 1.8287_MAE__resnet1d
    return f"{mae_km:.4f}_MAE__{best_model_type}"


def _export_random_tensor_signals_csv(*, cfg: Config, out_csv_path: str, seed: int) -> None:
    """
    Pick a random sample from ALL oscillograms (FaultDataset) and export its signals to CSV.
    CSV format:
      time_idx, ch0, ch1, ..., ch{C-1}
    """
    from data.dataset import FaultDataset

    # IMPORTANT: FaultDataset will apply t0/symseq/remove_dc based on cfg, and normalization too (cfg.NORMALIZE_DATA).
    ds = FaultDataset(
        data_dir=cfg.DATA_DIR,
        seq_length=cfg.SEQ_LENGTH,
        num_channels=cfg.NUM_CHANNELS,
        normalize=cfg.NORMALIZE_DATA,
        cfg=cfg,
    )

    if len(ds) == 0:
        raise ValueError("Dataset empty; cannot export random tensor.")

    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(ds)))
    signals_t, _distance_t = ds[idx]  # signals: (C, T)

    signals = signals_t.detach().cpu().numpy()
    c, t = signals.shape

    df = pd.DataFrame(signals.T, columns=[f"ch{ch}" for ch in range(c)])
    df.insert(0, "time_idx", np.arange(t, dtype=int))

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8")


def _save_best_run_if_improved(
    *,
    best_mae_km: float,
    best_model_type: str,
    run_dir: str,
    thesis_runs_dir: str,
    cfg: Config,
    save_seed: int,
) -> bool:
    """
    Saves run artifacts into:
      thesis_runs_dir/{best_mae}_MAE__{best_model}/
    only if best_mae_km is better than the currently best MAE found in thesis_runs_dir.
    """
    existing = _read_existing_best_maes(thesis_runs_dir)
    current_best = min(existing) if existing else float('inf')

    if best_mae_km >= current_best:
        print(
            f"[INFO] Skip saving best run: best_mae_km={best_mae_km:.4f} >= current_best={current_best:.4f}"
        )
        return False

    target_name = _format_best_folder(best_mae_km, best_model_type)
    target_dir = os.path.join(thesis_runs_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    # Copy artifacts of that comparison run (same as before)
    items = [
        'comparison_config.yaml',
        'comparison_summary.csv',
        'comparison_table.png',
        'comparison_training_loss.png',
        'comparison_mae.png',
        'comparison_metrics_bar.png',
        'comparison_error_boxplot.png',
    ]
    for item in items:
        src = os.path.join(run_dir, item)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target_dir, item))

    for model_type in ['cnn1d', 'resnet1d', 'dilated_cnn1d']:
        src_model_dir = os.path.join(run_dir, f'model_{model_type}')
        if not os.path.isdir(src_model_dir):
            continue

        # Куда переносим (только при улучшении best MAE)
        checkpoints_dir = os.path.join(target_dir, 'checkpoints', model_type)
        results_dir = os.path.join(target_dir, 'results', model_type)
        images_dir = os.path.join(target_dir, 'images')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # training history
        src_history = os.path.join(src_model_dir, 'training_history.png')
        if os.path.exists(src_history):
            dst_history = os.path.join(images_dir, f"training_history__{model_type}.png")
            if os.path.exists(dst_history):
                os.remove(dst_history)
            shutil.move(src_history, dst_history)

        # test predictions
        src_preds = os.path.join(src_model_dir, 'test_predictions.png')
        if os.path.exists(src_preds):
            dst_preds = os.path.join(images_dir, f"test_predictions__{model_type}.png")
            if os.path.exists(dst_preds):
                os.remove(dst_preds)
            shutil.move(src_preds, dst_preds)

        # test results json
        src_json = os.path.join(src_model_dir, 'test_results.json')
        if os.path.exists(src_json):
            dst_json = os.path.join(results_dir, 'test_results.json')
            if os.path.exists(dst_json):
                os.remove(dst_json)
            shutil.move(src_json, dst_json)

        # checkpoint: best_model.pth предпочтительнее, иначе последний *.pth
        src_ckpt = os.path.join(src_model_dir, 'best_model.pth')
        if not os.path.exists(src_ckpt):
            pths = sorted(
                [os.path.join(src_model_dir, f) for f in os.listdir(src_model_dir) if f.endswith('.pth')]
            )
            src_ckpt = pths[-1] if pths else ""

        if src_ckpt and os.path.exists(src_ckpt):
            dst_ckpt = os.path.join(checkpoints_dir, os.path.basename(src_ckpt))
            if os.path.exists(dst_ckpt):
                os.remove(dst_ckpt)
            shutil.move(src_ckpt, dst_ckpt)

    # NEW: export random tensor signals to CSV
    random_tensor_csv = os.path.join(target_dir, 'random_sample_signals.csv')
    _export_random_tensor_signals_csv(cfg=cfg, out_csv_path=random_tensor_csv, seed=save_seed)

    print(f"[OK] Saved improved best run to: {target_dir}")
    return True


def export_demo_layer_graphs(*, model, model_type: str, test_loader, out_dir: str, device: torch.device) -> None:
    """
    Save thesis-like demo graphs of activation behavior for the model.
    We:
      - auto-resolve layers by type
      - capture one batch from loader
      - export activation snapshots via export_layer_snapshots()
    """
    # Get one batch
    model.eval()
    batch = None
    for signals, _distances in test_loader:
        batch = signals
        break

    if batch is None:
        print(f"[WARN] No batch available for activation export: {model_type}")
        return

    batch = batch.to(device)

    layers = resolve_cnn_layer_names(model, model_type=model_type, layers_cfg="auto")
    if not layers:
        print(f"[WARN] No layers resolved for activation export: {model_type}")
        return

    recorder = ActivationRecorder(
        model,
        layer_names=layers,
        mode=LayerRecordMode(activation_source='module_output', aggregation='mean'),
    )

    try:
        snapshots = recorder.capture(batch, epoch=0, probe_id="demo", max_channels_to_plot=8)
        export_layer_snapshots(
            snapshots,
            out_dir=os.path.join(out_dir, 'layer_activations_demo', model_type),
            formats=['png', 'csv'],
            dpi=300,
            max_channels_to_plot=8,
        )
    finally:
        recorder.remove()


def run_single_comparison(args, *, cfg: Config, seed: int) -> Tuple[str, Dict[str, dict], str, float]:
    """
    Runs a single comparison round and returns:
      comparison_dir,
      results (metrics per model),
      best_model_type,
      best_mae_km
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Важно: не пишем в папку logs/, чтобы не раздувать диск.
    # Временные артефакты сравнения складываем в output/.
    comparison_dir = os.path.join(
        'output',
        'thesis',
        'runs_tmp',
        f'comparison_{timestamp}_run_seed_{seed}',
    )
    os.makedirs(comparison_dir, exist_ok=True)
    print(f"Comparison directory: {comparison_dir}")

    # Apply per-run config overrides
    cfg = cfg  # local alias
    cfg.SEED = seed
    if args.device is not None:
        cfg.DEVICE = args.device

    # Save comparison config
    config_path = os.path.join(comparison_dir, 'comparison_config.yaml')
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            {
                'models': args.models,
                'epochs': cfg.NUM_EPOCHS,
                'data_dir': cfg.DATA_DIR,
                'test_dir': args.test_dir,
                'device': cfg.DEVICE,
                'seed': cfg.SEED,
                'timestamp': timestamp,
                'num_runs': args.num_runs,
            },
            f,
            allow_unicode=True,
            default_flow_style=False,
        )

    histories: Dict[str, dict] = {}
    results: Dict[str, dict] = {}

    device = torch.device('cuda' if str(cfg.DEVICE).lower() == 'cuda' else 'cpu')

    for model_type in args.models:
        print(f"\n{'='*70}")
        print(f"Training model: {model_type}")
        print(f"{'='*70}")

        model_dir = os.path.join(comparison_dir, f'model_{model_type}')
        os.makedirs(model_dir, exist_ok=True)

        model_cfg = get_config(
            MODEL_TYPE=model_type,
            NUM_EPOCHS=cfg.NUM_EPOCHS,
            DATA_DIR=cfg.DATA_DIR,
            DEVICE=cfg.DEVICE,
            SEED=cfg.SEED,
            SAVE_DIR=model_dir,
            LOG_DIR=model_dir,
            SAVE_BEST_ONLY=True,
            SAVE_THESIS_PLOTS=False,  # важно: не плодим output/thesis на каждом раунде сравнения
        )

        for attr in [
            'NUM_CHANNELS', 'SEQ_LENGTH', 'BATCH_SIZE', 'LEARNING_RATE', 'NORMALIZATION_MODE',
            'SYMSEQ_ENABLED', 'T0_ENABLED', 'REMOVE_DC_ENABLED',
            'NUM_FILTERS', 'KERNEL_SIZE', 'DROPOUT',
            'OPTIMIZER', 'LOSS_FUNCTION', 'WEIGHT_DECAY',
            'GRADIENT_CLIP', 'SCHEDULER_TYPE', 'WARMUP_EPOCHS',
            'EARLY_STOPPING', 'PATIENCE', 'MIN_DELTA',
            'SAVE_EVERY_N_EPOCHS', 'TRAIN_SPLIT', 'NORMALIZE_DATA',
            'LINE_UNOM_KV', 'LINE_L_KM', 'LINE_R1_OHM_KM', 'LINE_X1_OHM_KM',
            'SAMPLING_FREQ_HZ', 'MAINS_FREQ_HZ',
            'T0_PRE_MS', 'T0_POST_MS', 'T0_ETA_I', 'T0_ETA_U',
        ]:
            if hasattr(cfg, attr):
                setattr(model_cfg, attr, getattr(cfg, attr))

        trainer = Trainer(model_cfg)
        trainer.train()

        histories[model_type] = {
            'train_loss': trainer.history['train_loss'].copy(),
            'val_loss': trainer.history['val_loss'].copy(),
            'val_mae': trainer.history['val_mae'].copy(),
        }

        checkpoint_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if checkpoints:
                checkpoint_path = os.path.join(model_dir, sorted(checkpoints)[-1])

        # Load best checkpoint for inference
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
            if "model_state_dict" in ckpt:
                trainer.model.load_state_dict(ckpt["model_state_dict"], strict=True)

        # Inference (validation or csv test dir)
        if args.test_dir and os.path.isdir(args.test_dir) and any(f.endswith('.csv') for f in os.listdir(args.test_dir)):
            print(f"\n[INFO] Running inference on test directory: {args.test_dir}")
            y_true, y_pred = run_inference_on_csv_dir(checkpoint_path, args.test_dir, cfg.DEVICE)
        else:
            print(f"\n[INFO] Running inference on validation set (best checkpoint)")
            y_true, y_pred = run_inference_on_loader(
                trainer.model,
                trainer.test_loader,
                model_cfg,
                trainer.scalers,
                trainer.device,
            )

        if y_true is not None and y_pred is not None and len(y_true) > 0:
            metrics = MetricsCalculator.regression_metrics(y_true, y_pred)
            metrics['errors'] = np.abs(y_true - y_pred)
            results[model_type] = metrics

            test_results = {
                'model_type': model_type,
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else None
                            for k, v in metrics.items() if k != 'errors'},
                'predictions': y_pred.tolist(),
                'true_values': y_true.tolist(),
            }
            with open(os.path.join(model_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)

            plot_predictions(y_true, y_pred, os.path.join(model_dir, 'test_predictions.png'))
            MetricsCalculator.print_regression_metrics(y_true, y_pred, dataset_name=model_type)
        else:
            print(f"[WARN] No inference results for {model_type}")
            results[model_type] = {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'mape': float('inf'),
                'errors': [],
            }

        # Export demo activation graphs
        export_demo_layer_graphs(
            model=trainer.model,
            model_type=model_type,
            test_loader=trainer.test_loader,
            out_dir=comparison_dir,
            device=trainer.device,
        )

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("Building comparison plots...")
    print(f"{'='*70}")

    plot_comparison_training_loss(histories, os.path.join(comparison_dir, 'comparison_training_loss.png'))
    plot_comparison_mae(histories, os.path.join(comparison_dir, 'comparison_mae.png'))
    plot_comparison_metrics_bar(results, os.path.join(comparison_dir, 'comparison_metrics_bar.png'))
    plot_comparison_error_boxplot(results, os.path.join(comparison_dir, 'comparison_error_boxplot.png'))
    plot_comparison_table(results, os.path.join(comparison_dir, 'comparison_table.png'))

    summary_rows = []
    for model_type in args.models:
        r = results[model_type]
        summary_rows.append({
            'Модель': model_type,
            'MAE (км)': f"{r['mae']:.4f}",
            'RMSE (км)': f"{r['rmse']:.4f}",
            'R^2': f"{r['r2']:.4f}",
            'MAPE (%)': f"{r['mape']:.2f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(comparison_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"[OK] Summary CSV saved to {summary_csv}")

    best_model = min(results, key=lambda m: results[m]['mae'])
    best_mae = results[best_model]['mae']

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model} (MAE = {best_mae:.4f} km)")
    print(f"{'='*70}")

    # IMPORTANT:
    # Не копируем comparison_*.png в output/thesis/model_comparison на каждом прогоне.
    # Иначе output/thesis/*.png будет расти даже на неулучшениях.
    # Артефакты диплома сохраняются только при улучшении MAE внутри:
    #   _save_best_run_if_improved() -> output/thesis/runs/<MAE>_MAE__...

    best_model_dir = os.path.join('output', 'thesis', 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    best_src_dir = os.path.join(comparison_dir, f'model_{best_model}')
    best_checkpoint_src = os.path.join(best_src_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_src):
        shutil.copy2(best_checkpoint_src, os.path.join(best_model_dir, 'best_model.pth'))
    with open(os.path.join(best_model_dir, 'best_model_name.txt'), 'w', encoding='utf-8') as f:
        f.write(best_model)

    best_pred_src = os.path.join(best_src_dir, 'test_predictions.png')
    if os.path.exists(best_pred_src):
        shutil.copy2(best_pred_src, os.path.join(best_model_dir, 'best_predictions.png'))

    print(f"[OK] Best model artifacts copied to {best_model_dir}")

    print(summary_df.to_string(index=False))
    print(f"{'='*70}\n")

    return comparison_dir, results, best_model, float(best_mae)


def main():
    args = parse_args()

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")

    # Load base config
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        cfg = Config()

    # Apply base overrides once
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
    if args.data_dir is not None:
        cfg.DATA_DIR = args.data_dir
    if args.device is not None:
        cfg.DEVICE = args.device

    cfg.DEVICE = args.device if args.device is not None else cfg.DEVICE

    thesis_runs_dir = os.path.join('output', 'thesis', 'runs')
    os.makedirs(thesis_runs_dir, exist_ok=True)

    save_all_runs = _parse_bool_str(args.save_all_runs)

    print(f"[INFO] thesis_runs_dir={thesis_runs_dir}")
    print(f"[INFO] num_runs={args.num_runs} save_all_runs={save_all_runs}")

    best_saved_any = False
    for run_idx in range(args.num_runs):
        seed = args.seed + args.seed_offset + run_idx
        print(f"\n{'='*70}")
        print(f"[RUN {run_idx+1}/{args.num_runs}] seed={seed}")
        print(f"{'='*70}")

        comparison_dir, results, best_model_type, best_mae_km = run_single_comparison(args, cfg=cfg, seed=seed)

        saved = _save_best_run_if_improved(
            best_mae_km=best_mae_km,
            best_model_type=best_model_type,
            run_dir=comparison_dir,
            thesis_runs_dir=thesis_runs_dir,
            cfg=cfg,
            save_seed=seed,
        )
        best_saved_any = best_saved_any or saved

    if not best_saved_any:
        print("[INFO] No improved run was found; thesis runs directory unchanged.")


if __name__ == '__main__':
    main()
