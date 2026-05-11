from __future__ import annotations

import csv
import os
import re
from dataclasses import asdict
from typing import Dict, List, Literal, Optional

import numpy as np

from utils.plots import _COLOR_TRAIN, _COLOR_VAL  # reuse thesis palette


ExportFormat = Literal["png", "csv", "npy"]


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _sanitized_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")


def export_layer_snapshots(
    snapshots,
    *,
    out_dir: str,
    formats: List[ExportFormat],
    dpi: int = 300,
    max_channels_to_plot: int = 8,
) -> None:
    """
    Export activation snapshots.

    - PNG: use snapshot.activation_curve (if present). If absent, skip PNG for that snapshot.
    - CSV: always export scalar stats; one row per snapshot.
    - NPY: optional raw activation is not exported here (plan requirement: keep disk small).
      The Snapshot is expected to already have aggregated stats/curve; NPY can be used later
      if you extend snapshot to include raw tensors.

    snapshots: List[LayerActivationSnapshot] from utils/activation_recorder.py
    """
    if not snapshots:
        return

    formats_set = set(formats)
    if not formats_set:
        return

    os.makedirs(out_dir, exist_ok=True)
    _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "snapshots.csv")

    # CSV export (append mode)
    if "csv" in formats_set:
        # Write header if file doesn't exist.
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "epoch",
                        "probe_id",
                        "layer_name",
                        "activation_shape",
                        "mean",
                        "std",
                        "min",
                        "max",
                        "rms",
                        "energy",
                    ]
                )

            for snap in snapshots:
                shape = tuple(getattr(snap, "activation_shape", ()))
                stats: Dict[str, float] = getattr(snap, "activation_stats", {}) or {}
                writer.writerow(
                    [
                        snap.epoch,
                        snap.probe_id,
                        snap.layer_name,
                        str(shape),
                        stats.get("mean", float("nan")),
                        stats.get("std", float("nan")),
                        stats.get("min", float("nan")),
                        stats.get("max", float("nan")),
                        stats.get("rms", float("nan")),
                        stats.get("energy", float("nan")),
                    ]
                )

    # PNG export
    if "png" in formats_set:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for snap in snapshots:
            curve = getattr(snap, "activation_curve", None)
            if not curve:
                continue

            time_axis = curve.get("time_axis")
            values = curve.get("values")
            if not time_axis or not values:
                continue

            fig = plt.figure(figsize=(14, 4.5))
            ax = fig.add_subplot(1, 1, 1)

            ax.plot(time_axis, values, color=_COLOR_TRAIN, linewidth=2.2, label="Активность (агрегированная)")
            ax.set_xlabel("Индекс отсчёта")
            ax.set_ylabel("Усреднённая |A|")
            ax.set_title(f"Активность слоя: {snap.layer_name}  |  epoch={snap.epoch}  |  {snap.probe_id}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            out_path = os.path.join(
                out_dir,
                f"epoch_{snap.epoch:03d}__layer_{_sanitized_name(snap.layer_name)}__{_sanitized_name(snap.probe_id)}.png",
            )
            _ensure_dir(out_path)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)


def export_epoch_layer_stats_csv(epoch_stats_rows: List[Dict[str, object]], out_path: str) -> None:
    """
    Write epoch-wise layer stats to CSV.
    epoch_stats_rows: list of dicts: {epoch, set, layer_name, metric, value}
    """
    _ensure_dir(out_path)

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = None
        if write_header:
            writer = csv.DictWriter(f, fieldnames=["epoch", "set", "layer_name", "metric", "value"])
            writer.writeheader()
        else:
            writer = csv.DictWriter(f, fieldnames=["epoch", "set", "layer_name", "metric", "value"])

        for row in epoch_stats_rows:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "set": row["set"],
                    "layer_name": row["layer_name"],
                    "metric": row["metric"],
                    "value": row["value"],
                }
            )


def export_epoch_layer_stats_curves(
    *,
    epoch_stats,
    out_dir: str,
    dpi: int = 300,
) -> None:
    """
    Create per-layer curves PNG for each metric aggregated over epochs.

    epoch_stats: list of dicts: {epoch, set, layer_name, metric, value}
    """
    if not epoch_stats:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_dir)

    # Organize by layer then metric
    by_layer: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in epoch_stats:
        layer = row["layer_name"]
        metric = row["metric"]
        by_layer.setdefault(layer, {}).setdefault(metric, []).append(row)

    for layer_name, metrics_map in by_layer.items():
        for metric, rows in metrics_map.items():
            # sort by epoch
            rows = sorted(rows, key=lambda r: int(r["epoch"]))
            train_rows = [r for r in rows if r["set"] == "train"]
            val_rows = [r for r in rows if r["set"] == "val"]

            fig = plt.figure(figsize=(14, 4.5))
            ax = fig.add_subplot(1, 1, 1)

            if train_rows:
                xs = [int(r["epoch"]) for r in train_rows]
                ys = [float(r["value"]) for r in train_rows]
                ax.plot(xs, ys, color=_COLOR_TRAIN, linewidth=2.4, marker="o", markersize=3, label="Train")
            if val_rows:
                xs = [int(r["epoch"]) for r in val_rows]
                ys = [float(r["value"]) for r in val_rows]
                ax.plot(xs, ys, color=_COLOR_VAL, linewidth=2.4, marker="s", markersize=3, label="Val")

            ax.set_xlabel("Эпоха")
            ax.set_ylabel(metric)
            ax.set_title(f"Эпохи vs слой: {layer_name}  |  metric={metric}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            out_path = os.path.join(out_dir, f"layer_{_sanitized_name(layer_name)}__metric_{_sanitized_name(metric)}.png")
            _ensure_dir(out_path)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)
