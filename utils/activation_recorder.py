from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn


ActivationSource = Literal["post_conv", "post_relu", "post_pool", "module_output"]
AggregationMetric = Literal["mean", "median", "rms", "energy"]


@dataclass(frozen=True)
class LayerRecordMode:
    activation_source: ActivationSource = "module_output"
    aggregation: AggregationMetric = "mean"


@dataclass
class LayerActivationSnapshot:
    epoch: int
    probe_id: str
    layer_name: str
    activation_shape: Tuple[int, ...]
    activation_stats: Dict[str, float]
    activation_curve: Optional[Dict[str, List[float]]] = None
    # activation_curve: {"time_axis": [...], "values": [...]}


@dataclass
class _HookRecord:
    # Stores latest forward output from that hook call.
    output: Optional[Any] = None


class ActivationRecorder:
    """
    Record forward activations for selected layers using forward hooks.

    Usage:
      recorder = ActivationRecorder(model, layer_names=[...], mode=LayerRecordMode(...))
      snapshots = recorder.capture(inputs, epoch=..., probe_id=...)
      recorder.remove()
    """

    def __init__(self, model: nn.Module, layer_names: List[str], mode: LayerRecordMode):
        self.model = model
        self.layer_names = list(layer_names)
        self.mode = mode

        # Resolve module names to actual modules.
        name_to_module = dict(model.named_modules())
        missing = [n for n in self.layer_names if n not in name_to_module]
        if missing:
            raise ValueError(f"ActivationRecorder: layer(s) not found: {missing}")

        self._name_to_module = {n: name_to_module[n] for n in self.layer_names}
        self._records: Dict[str, _HookRecord] = {n: _HookRecord() for n in self.layer_names}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        for layer_name, module in self._name_to_module.items():
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self._handles.append(handle)

    def _make_hook(self, layer_name: str) -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            self._records[layer_name].output = output

        return _hook

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    @staticmethod
    def _extract_activation_tensor(output: Any) -> Optional[torch.Tensor]:
        """
        Robust extraction:
        - tensor -> tensor
        - tuple/list -> first tensor
        - dict -> first tensor value
        - None -> None
        """
        if output is None:
            return None
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
            return None
        if isinstance(output, dict):
            for _, v in output.items():
                if isinstance(v, torch.Tensor):
                    return v
            return None
        return None

    def capture(
        self,
        inputs: torch.Tensor,
        epoch: int,
        probe_id: str,
        *,
        max_channels_to_plot: int = 8,
    ) -> List[LayerActivationSnapshot]:
        self.model.eval()  # stable hooks; no output change expected in eval mode
        # Clear previous records.
        for n in self.layer_names:
            self._records[n].output = None

        with torch.no_grad():
            _ = self.model(inputs)

        snapshots: List[LayerActivationSnapshot] = []
        for layer_name in self.layer_names:
            out = self._records[layer_name].output
            act = self._extract_activation_tensor(out)
            if act is None:
                warnings.warn(f"ActivationRecorder: layer '{layer_name}' output is not a tensor; skipping.")
                continue

            act_cpu = act.detach().cpu()
            stats = activation_tensor_stats(act_cpu)
            curve = build_activation_curve(act_cpu, max_channels=max_channels_to_plot)

            snapshots.append(
                LayerActivationSnapshot(
                    epoch=epoch,
                    probe_id=probe_id,
                    layer_name=layer_name,
                    activation_shape=tuple(act_cpu.shape),
                    activation_stats=stats,
                    activation_curve=curve,
                )
            )
        return snapshots


def activation_tensor_stats(activation: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    Recommended keys: mean, std, min, max, rms, energy
    Additionally (needed for epoch-wise curves):
      - mean_abs
      - sparsity (fraction(|x| < eps))
    Computed over all elements.
    """
    x = activation.float()
    mean = x.mean().item()
    std = x.std(unbiased=False).item()
    min_v = x.min().item()
    max_v = x.max().item()
    rms = torch.sqrt(torch.mean(x * x) + eps).item()
    energy = torch.mean(x * x).item()  # energy per element; thesis-friendly & scale-invariant
    mean_abs = torch.mean(torch.abs(x)).item()
    sparsity = torch.lt(torch.abs(x), eps).float().mean().item()

    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min_v),
        "max": float(max_v),
        "mean_abs": float(mean_abs),
        "sparsity": float(sparsity),
        "rms": float(rms),
        "energy": float(energy),
    }


def epoch_layer_stats_aggregate(activation: torch.Tensor, metric: str) -> float:
    """
    Scalar metric from tensor (plan requirement).
    Supported:
      - mean_abs
      - rms
      - energy
      - sparsity
    """
    x = activation.float()
    if metric == "mean_abs":
        return torch.mean(torch.abs(x)).item()
    if metric == "rms":
        return torch.sqrt(torch.mean(x * x) + 1e-12).item()
    if metric == "energy":
        return torch.mean(x * x).item()
    if metric == "sparsity":
        eps = 1e-12
        return (torch.lt(torch.abs(x), eps).float().mean()).item()
    raise ValueError(f"Unknown metric: {metric}")


def _sanitized_name(name: str) -> str:
    # used by export, but also helps for debug
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")


def build_activation_curve(
    activation: torch.Tensor,
    *,
    max_channels: int = 8,
) -> Optional[Dict[str, List[float]]]:
    """
    Build a 1D curve over time indices for conv-like activations.
    - (B, C, T) -> mean over channels of abs(x)
    - (C, T)    -> mean over channels of abs(x)
    - (B, F)    -> None (not time-series)
    - scalar/1D  -> None
    """
    x = activation
    if x.dim() == 3:
        # (B, C, T)
        x = x.mean(dim=0)  # -> (C, T)
    elif x.dim() == 2:
        # (C, T)
        pass
    else:
        return None

    if x.dim() != 2:
        return None

    # Select subset of channels if too many.
    c, t = x.shape
    if c > max_channels:
        # pick channels with highest variance
        var = x.var(dim=1, unbiased=False)
        topk = torch.topk(var, k=max_channels, largest=True).indices
        x = x[topk]
    curve = torch.mean(torch.abs(x), dim=0)  # -> (T,)
    values = curve.detach().cpu().tolist()
    time_axis = list(range(len(values)))
    return {"time_axis": time_axis, "values": values}


def _auto_pick_first_n_convs(model: nn.Module, n: int) -> List[str]:
    """
    Best-effort: pick first N modules whose name contains 'conv' and are nn.Conv1d.
    """
    conv_names: List[str] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d) and "conv" in name.lower():
            conv_names.append(name)
    return conv_names[:n]


def _auto_pick_following_bn_pool(model: nn.Module, conv_names: List[str]) -> List[str]:
    """
    Best-effort: for each conv name, include adjacent/belonging bn/relu/pool modules
    if their names look like they belong.
    This is heuristic; export is optional so thesis visuals can still work.
    """
    all_names = [n for n, _ in model.named_modules()]
    out: List[str] = []

    def _contains_any(s: str, needles: Iterable[str]) -> bool:
        s_low = s.lower()
        return any(nd in s_low for nd in needles)

    for cn in conv_names:
        out.append(cn)
        # include modules with same prefix segment (common in simple blocks)
        prefix = cn.split(".")[0] if "." in cn else cn
        for name in all_names:
            if name == cn:
                continue
            if prefix and (prefix in name):
                if _contains_any(name, ["bn", "batchnorm", "pool", "maxpool", "avgpool", "relu"]):
                    if name not in out:
                        out.append(name)
        # Fallback include just pool/bn somewhere after (if heuristics above find nothing)
    # Deduplicate preserving order
    dedup: List[str] = []
    seen = set()
    for n in out:
        if n not in seen:
            dedup.append(n)
            seen.add(n)
    return dedup


def resolve_cnn_layer_names(model: nn.Module, model_type: str, layers_cfg: Any = "auto") -> List[str]:
    """
    Best-effort "auto" layer selection without hard-coding exact internal names.
    Plan requirement: for CNN1D/DilatedCNN1D capture conv1/conv2/conv3 (+optional bn/pool).
    For ResNet1D capture stem/stage1..stage4/gap/head (if present).
    """
    model_type = str(model_type).lower().strip()

    if layers_cfg not in ("auto", None, "auto_all_conv_blocks", "auto_conv", "auto_all"):
        # If user passed explicit list of names, validate elsewhere in ActivationRecorder.
        if isinstance(layers_cfg, list):
            return layers_cfg
        # Unknown config: fall back to auto.
        warnings.warn(f"resolve_cnn_layer_names: unrecognized layers_cfg={layers_cfg}; falling back to auto.")
    # model_type handling
    if model_type in ("cnn1d", "dilated_cnn1d", "dilated_cnn1d"):
        convs = _auto_pick_first_n_convs(model, 3)
        if not convs:
            # fallback: any conv1d modules, first 3
            convs = [name for name, m in model.named_modules() if isinstance(m, nn.Conv1d)][:3]
        if not convs:
            raise ValueError("Could not auto-resolve convolutional layers for CNN1D model.")
        layers = _auto_pick_following_bn_pool(model, convs)
        return layers

    if model_type == "resnet1d":
        # Heuristic patterns
        candidates: List[str] = []
        for name, _m in model.named_modules():
            nl = name.lower()
            if any(k in nl for k in ["stem", "stage", "layer", "gap", "head"]):
                candidates.append(name)

        # Prefer ordered "stem" then stage1..stage4 then gap/head
        def _score(n: str) -> int:
            nl = n.lower()
            if "stem" in nl:
                return 0
            if "stage1" in nl:
                return 1
            if "stage2" in nl:
                return 2
            if "stage3" in nl:
                return 3
            if "stage4" in nl:
                return 4
            if "gap" in nl:
                return 90
            if "head" in nl:
                return 91
            # stageX general fallback
            if "stage" in nl:
                # extract stage number if present
                m = re.search(r"stage(\d+)", nl)
                if m:
                    return 10 + int(m.group(1))
                return 10
            return 50

        candidates = sorted(set(candidates), key=_score)
        if not candidates:
            # fallback to all conv1d
            candidates = [name for name, m in model.named_modules() if isinstance(m, nn.Conv1d)]
        if not candidates:
            raise ValueError("Could not auto-resolve layers for ResNet1D model.")

        return candidates[:20]  # keep hook count bounded for thesis visuals

    # unknown
    raise ValueError(f"resolve_cnn_layer_names: unknown model_type={model_type}")
