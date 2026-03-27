"""Manual verification script for fault inception (t0) detection.

Usage (from repo root):
    python tests/fault_inception/check_t0.py

Place your COMTRADE files (.cfg + .dat) in:
    tests/fault_inception/oscillograms/

Output example:
    File                        | sample |   t0, ms  |    fs, Hz
    ----------------------------|--------|-----------|----------
    cutAB_5km_t10ms.cfg         |    500 |    10.000 |     50000
    cutAC0_10km_t25ms.cfg       |   1250 |    25.000 |     50000

Adjust PARAMS_OVERRIDE below to tune the detector.
"""

import sys
import os
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root so imports work regardless of CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

# ---------------------------------------------------------------------------
# Tunable parameters — edit here to adjust the detector
# ---------------------------------------------------------------------------
PARAMS_OVERRIDE = dict(
    mains_hz         = 50.0,   # fundamental frequency of the power system
    coarse_top_k     = 5,      # number of D4 peaks considered for coarse t0
    coarse_window_ms = 200.0,  # half-window around coarse t0, ms
    pre_fault_ms     = 20.0,   # pre-fault history kept when cropping
    post_fault_ms    = 60.0,   # post-fault window kept when cropping
    threshold_mult   = 1.0,    # adaptive threshold multiplier (lower = earlier)
)

# ---------------------------------------------------------------------------
# Folder with test oscillograms
# ---------------------------------------------------------------------------
OSC_DIR = Path(__file__).parent / "oscillograms"

# ---------------------------------------------------------------------------
# Import detection function
# ---------------------------------------------------------------------------
try:
    from fault_inception import FaultInceptionParams, detect_t0_multi_phase
except ImportError:
    # fallback: try importing from data package inside src
    from src.fault_distance.data.fault_inception import (
        FaultInceptionParams,
        detect_t0_multi_phase,
    )


# ---------------------------------------------------------------------------
# COMTRADE parser (uses comtrade library if available, else minimal built-in)
# ---------------------------------------------------------------------------

def _parse_comtrade(cfg_path: Path):
    """Return (fs_hz, ia, ib, ic) from a COMTRADE .cfg/.dat pair.

    Tries `comtrade` library first; falls back to a minimal ASCII parser.

    Returns
    -------
    fs_hz : float
    ia, ib, ic : np.ndarray  (phase currents, physical values)
    """
    import numpy as np

    try:
        import comtrade
        rec = comtrade.load(str(cfg_path))
        fs_hz = float(rec.cfg.sample_rates[0][0])

        # find channels by name (case-insensitive, look for Ia/Ib/Ic)
        names = [ch.ch_id.upper() for ch in rec.cfg.analog_channels]

        def _find(patterns):
            for pat in patterns:
                for idx, n in enumerate(names):
                    if pat in n:
                        return rec.analog[idx]
            return None

        ia = _find(["IA", "I_A", "CT1IA", "PHASE_A_I"])
        ib = _find(["IB", "I_B", "CT1IB", "PHASE_B_I"])
        ic = _find(["IC", "I_C", "CT1IC", "PHASE_C_I"])

        if ia is None or ib is None or ic is None:
            # fallback: take first three analog channels
            ia = rec.analog[0]
            ib = rec.analog[1]
            ic = rec.analog[2]

        return fs_hz, np.asarray(ia), np.asarray(ib), np.asarray(ic)

    except Exception:
        pass  # fall through to minimal parser

    # -----------------------------------------------------------------------
    # Minimal ASCII COMTRADE parser (no external deps)
    # -----------------------------------------------------------------------
    import re
    import numpy as np

    cfg_lines = cfg_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # 1) station line (skip)
    # 2) counts line: TT,nA,nD
    counts_line = cfg_lines[1] if len(cfg_lines) > 1 else ""
    m = re.match(r"(\d+),(\d+)A,(\d+)D", counts_line.strip())
    num_analog = int(m.group(2)) if m else 3

    # 3) analog channel definitions (num_analog lines)
    analog_ch_names = []
    for k in range(2, 2 + num_analog):
        if k < len(cfg_lines):
            parts = cfg_lines[k].split(",")
            ch_name = parts[1].strip().upper() if len(parts) > 1 else f"CH{k-2}"
            analog_ch_names.append(ch_name)
        else:
            analog_ch_names.append(f"CH{k-2}")

    # skip digital channels
    # find sample rate line (after analogs + digitals + 1 line)
    # heuristic: scan for line matching "<float>,<int>" pattern
    fs_hz = 10000.0  # default fallback
    for line in cfg_lines:
        m = re.match(r"^([0-9.]+),(\d+)$", line.strip())
        if m:
            candidate = float(m.group(1))
            if candidate > 100:  # likely a sampling rate, not a small number
                fs_hz = candidate
                break

    # parse .dat file (ASCII format: sample_no, timestamp, ch1, ch2, ...)
    dat_path = cfg_path.with_suffix(".dat")
    if not dat_path.exists():
        dat_path = cfg_path.with_suffix(".DAT")

    data_rows = []
    with dat_path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip().replace(",", " ")
            if not line:
                continue
            try:
                vals = list(map(float, line.split()))
                data_rows.append(vals)
            except ValueError:
                continue

    if not data_rows:
        raise ValueError(f"No numeric data found in {dat_path}")

    arr = np.array(data_rows, dtype=np.float64)
    # columns: [sample_no, timestamp_us, ch0, ch1, ch2, ...]
    ch_start = 2
    n_avail = arr.shape[1] - ch_start

    def _name_to_idx(patterns):
        for pat in patterns:
            for idx, n in enumerate(analog_ch_names):
                if pat in n:
                    return idx
        return None

    ia_idx = _name_to_idx(["IA", "I_A", "CT1IA"]) or 0
    ib_idx = _name_to_idx(["IB", "I_B", "CT1IB"]) or min(1, n_avail - 1)
    ic_idx = _name_to_idx(["IC", "I_C", "CT1IC"]) or min(2, n_avail - 1)

    ia = arr[:, ch_start + ia_idx]
    ib = arr[:, ch_start + ib_idx]
    ic = arr[:, ch_start + ic_idx]

    return fs_hz, ia, ib, ic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import numpy as np

    if not OSC_DIR.exists():
        print(f"[ERROR] Oscillograms directory not found: {OSC_DIR}")
        print("        Create it and place your .cfg + .dat files there.")
        sys.exit(1)

    cfg_files = sorted(
        list(OSC_DIR.glob("*.cfg")) + list(OSC_DIR.glob("*.CFG"))
    )

    if not cfg_files:
        print(f"[INFO ] No .cfg files found in {OSC_DIR}")
        print("        Place COMTRADE pairs (.cfg + .dat) there and re-run.")
        sys.exit(0)

    # ---- header ----
    col_file = 36
    col_sample = 8
    col_ms = 10
    col_fs = 10

    sep = "-" * (col_file + col_sample + col_ms + col_fs + 10)
    header = (
        f"{'File':<{col_file}} | "
        f"{'sample':>{col_sample}} | "
        f"{'t0, ms':>{col_ms}} | "
        f"{'fs, Hz':>{col_fs}}"
    )
    print()
    print(header)
    print(sep)

    ok_count = 0
    err_count = 0
    results = []

    for cfg_path in cfg_files:
        fname = cfg_path.name
        try:
            fs_hz, ia, ib, ic = _parse_comtrade(cfg_path)

            params = FaultInceptionParams(
                fs_hz=fs_hz,
                **PARAMS_OVERRIDE,
            )

            currents = np.stack([ia, ib, ic], axis=0)  # (3, T)
            t0_sample = detect_t0_multi_phase(currents, params)

            if t0_sample is None:
                row = (
                    f"{fname:<{col_file}} | "
                    f"{'NOT FOUND':>{col_sample}} | "
                    f"{'---':>{col_ms}} | "
                    f"{fs_hz:>{col_fs}.0f}"
                )
                err_count += 1
            else:
                t0_ms = t0_sample / fs_hz * 1000.0
                row = (
                    f"{fname:<{col_file}} | "
                    f"{t0_sample:>{col_sample}d} | "
                    f"{t0_ms:>{col_ms}.3f} | "
                    f"{fs_hz:>{col_fs}.0f}"
                )
                ok_count += 1

            results.append((fname, row, None))
            print(row)

        except Exception as exc:
            err_row = (
                f"{fname:<{col_file}} | "
                f"{'ERROR':>{col_sample}} | "
                f"{'---':>{col_ms}} | "
                f"{'---':>{col_fs}}"
            )
            results.append((fname, err_row, str(exc)))
            print(err_row)
            print(f"  ↳ {exc}")
            err_count += 1

    print(sep)
    print(f"\nTotal: {len(cfg_files)} files   OK: {ok_count}   Failed/Not found: {err_count}")
    print()

    # verbose errors block
    verbose_errors = [(f, e) for f, _, e in results if e is not None]
    if verbose_errors:
        print("=" * 60)
        print("ERRORS:")
        for fname, err in verbose_errors:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
