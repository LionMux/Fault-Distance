from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
from torch.utils.data._utils.collate import default_collate


ProbeMode = Literal["first_val_sample", "first_train_sample", "fixed_index"]


@dataclass(frozen=True)
class ProbeSpec:
    probe_mode: ProbeMode
    fixed_index: Optional[int] = None
    max_samples: int = 1  # default: 1 for “first oscillogram”
    batch_size: int = 1  # only for locating probe in dataloader


def _to_batch_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has a batch dimension as first axis.
    Accepts either (C, T) or (B, C, T) and returns (B, C, T).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.dim() == 0:
        raise ValueError("Probe signal tensor has invalid ndim=0.")
    if x.dim() == 1:
        # Uncommon for CNN1D; still handle: (T,) -> (1, T)
        return x.unsqueeze(0)
    return x.unsqueeze(0) if x.dim() >= 2 and x.size(0) != 1 else x


def _dataset_first_samples(dataset, count: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    samples = []
    for i in range(count):
        item = dataset[i]
        if not isinstance(item, (tuple, list)) or len(item) < 1:
            raise ValueError("Dataset __getitem__ must return (signals, distances, ...).")
        signals = item[0]
        distances = item[1] if len(item) > 1 else None
        samples.append((signals, distances))
    return samples


@torch.no_grad()
def select_probe_batches(
    *,
    set_name: Literal["train", "val"],
    probe_spec: ProbeSpec,
    dataloader: torch.utils.data.DataLoader,
    seed: int,
    device: torch.device,
) -> List[Tuple[str, torch.Tensor]]:
    """
    Deterministically pick probe sample(s) from the provided dataloader.

    Returns:
        List of (probe_id, signals_batch) where signals_batch shape is (B, C, T) (or (B, F) for FC).
    """
    if probe_spec.max_samples <= 0:
        return []

    # Deterministic selection via generator (kept for config completeness).
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    # Interpret probe mode relative to "set_name".
    if probe_spec.probe_mode == "first_train_sample" and set_name != "train":
        raise ValueError("probe_mode=first_train_sample requires set_name='train'.")
    if probe_spec.probe_mode == "first_val_sample" and set_name != "val":
        raise ValueError("probe_mode=first_val_sample requires set_name='val'.")

    dataset = dataloader.dataset
    if dataset is None:
        raise ValueError("Dataloader has no dataset.")

    if probe_spec.probe_mode in ("first_val_sample", "first_train_sample"):
        start_idx = 0
    elif probe_spec.probe_mode == "fixed_index":
        if probe_spec.fixed_index is None:
            raise ValueError("probe_mode=fixed_index requires fixed_index != None.")
        start_idx = int(probe_spec.fixed_index)
    else:
        raise ValueError(f"Unknown probe_mode: {probe_spec.probe_mode}")

    # For max_samples>1, take a contiguous block deterministically.
    idxs = list(range(start_idx, start_idx + probe_spec.max_samples))

    # Extract from underlying dataset directly (plan requirement: dataset[0] wrapped into batch dim).
    signals_list: List[torch.Tensor] = []
    for i in idxs:
        item = dataset[i]
        if not isinstance(item, (tuple, list)) or len(item) < 1:
            raise ValueError("Dataset __getitem__ must return (signals, distances, ...).")

        signals = item[0]
        if not isinstance(signals, torch.Tensor):
            raise TypeError(f"Expected signals tensor at dataset[{i}] index 0, got {type(signals)}")

        # Ensure batch dimension per-sample, then later concatenate.
        if signals.dim() >= 2:
            signals_b = signals.unsqueeze(0)
        else:
            signals_b = signals.unsqueeze(0)  # (T,) -> (1, T)
        signals_list.append(signals_b)

    signals_batch = torch.cat(signals_list, dim=0).to(device=device)

    # Create a stable probe_id. For multi-sample we encode a range.
    if probe_spec.max_samples == 1:
        probe_id = f"{set_name}_0" if start_idx == 0 else f"{set_name}_{start_idx}"
    else:
        probe_id = f"{set_name}_{start_idx}-{start_idx + probe_spec.max_samples - 1}"

    # Return as one batch (B=max_samples). This matches “max_samples” semantics.
    return [(probe_id, signals_batch)]
