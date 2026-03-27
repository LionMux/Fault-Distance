"""Manual verification script for fault inception (t0) detection.

Usage (from repo root):
    python tests/fault_inception/check_t0.py

Place your COMTRADE files (.cfg + .dat) in:
    tests/fault_inception/oscillograms/

Output example:
    File                                 |   sample |     t0, ms |     fs, Hz
    -------------------------------------|----------|------------|----------
    cutAB_5km_t10ms.cfg                  |      500 |     10.000 |      50000
    cutAC0_10km_t25ms.cfg                |     1250 |     25.000 |      50000

Adjust the three config blocks below:
  1. PARAMS_OVERRIDE   — detector tuning
  2. CHANNEL_NAMES     — exact channel names as they appear in your .cfg
  3. CHANNEL_SCALE     — multiply raw value by this to get physical units (A / kV)
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root so imports work regardless of CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data"))

# ===========================================================================
# 1) DETECTOR PARAMETERS  — tune if t0 is wrong
# ===========================================================================
PARAMS_OVERRIDE = dict(
    mains_hz         = 50.0,   # fundamental frequency of the power system, Hz
    coarse_top_k     = 5,      # how many D4 peaks to consider for coarse t0
    coarse_window_ms = 2000.0,  # half-window around coarse t0, ms
    pre_fault_ms     = 20.0,   # pre-fault history kept when cropping (no effect on detection)
    post_fault_ms    = 60.0,   # post-fault window kept when cropping  (no effect on detection)
    threshold_mult   = 1.0,    # adaptive threshold multiplier — lower = fires earlier
)

# ===========================================================================
# 2) CHANNEL NAMES
#    Write the EXACT name as it appears in the .cfg file (field 2 of each
#    analog channel line).  Matching is case-insensitive substring, so "IA"
#    matches "CT1Ia", "Phase_IA", etc.
#
#    Run the script once WITHOUT filling anything — it will print all channel
#    names found in each .cfg so you can copy-paste them here.
#
#    Set a key to None to keep the automatic pattern-based guess.
# ===========================================================================
CHANNEL_NAMES = dict(
    ia = "CT1IAprim",   # e.g. "CT1IA"   or  "Ia"   or  "I_A"   — phase A current
    ib = "CT1IBprim",   # e.g. "CT1IB"
    ic = "CT1ICprim",   # e.g. "CT1IC"
    ua = "S1) VT1UAprim",   # e.g. "BUS1UA" or  "Ua"   or  "U_A"   — phase A voltage (display only)
    ub = "S1) VT1UBprim",   # e.g. "BUS1UB"
    uc = "S1) VT1UCprim",   # e.g. "BUS1UC"
)

# ===========================================================================
# 3) CHANNEL SCALING
#    Multiply the raw .dat value by this factor to get PHYSICAL units (A / kV).
#
#    If your channels are already in primary units — leave everything as 1.0.
#
#    If they are in SECONDARY units, set the ratio:
#      CT 600/5   → ia/ib/ic = 120.0
#      VT 110/0.1 → ua/ub/uc = 1100.0
#
#    When the `comtrade` library is installed, the 'a' coefficient from .cfg
#    is applied automatically by the library — set scale to 1.0 in that case.
#    For the built-in ASCII parser: if you leave scale = 1.0, the 'a'
#    coefficient from the .cfg line is used automatically.
# ===========================================================================
CHANNEL_SCALE = dict(
    ia = 1.0,   # set to CT ratio if .dat is in secondary amperes
    ib = 1.0,
    ic = 1.0,
    ua = 1.0,   # set to VT ratio if .dat is in secondary volts
    ub = 1.0,
    uc = 1.0,
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
    from src.fault_distance.data.fault_inception import (
        FaultInceptionParams,
        detect_t0_multi_phase,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ch_patterns(key: str) -> list:
    """Return search patterns for a channel key, honouring CHANNEL_NAMES."""
    explicit = CHANNEL_NAMES.get(key)
    if explicit is not None:
        return [explicit.upper()]
    fallbacks = {
        "ia": ["CT1IA", "IA", "I_A", "PHASE_A_I", "IL1", "IA1"],
        "ib": ["CT1IB", "IB", "I_B", "PHASE_B_I", "IL2", "IB1"],
        "ic": ["CT1IC", "IC", "I_C", "PHASE_C_I", "IL3", "IC1"],
        "ua": ["BUS1UA", "UA", "U_A", "VA", "V_A", "UL1", "VL1"],
        "ub": ["BUS1UB", "UB", "U_B", "VB", "V_B", "UL2", "VL2"],
        "uc": ["BUS1UC", "UC", "U_C", "VC", "V_C", "UL3", "VL3"],
    }
    return fallbacks.get(key, [])


def _find_idx(names_upper: list, key: str):
    """Return index of first matching channel name, or None."""
    for pat in _ch_patterns(key):
        for idx, n in enumerate(names_upper):
            if pat in n:
                return idx
    return None


# ---------------------------------------------------------------------------
# COMTRADE parser
# ---------------------------------------------------------------------------

def _parse_comtrade(cfg_path: Path):
    """Return (fs_hz, channels_dict) where channels_dict has keys:
       'ia', 'ib', 'ic', 'ua', 'ub', 'uc'  (np.ndarray or None)
    All returned arrays are in PHYSICAL units after scaling.
    """
    import numpy as np

    # ---- try `comtrade` library (handles binary, applies a/b/skew) ---------
    try:
        import comtrade
        rec = comtrade.load(str(cfg_path))
        fs_hz = float(rec.cfg.sample_rates[0][0])
        names_up = [ch.ch_id.upper() for ch in rec.cfg.analog_channels]

        def _get(key):
            idx = _find_idx(names_up, key)
            if idx is None:
                return None
            arr = np.asarray(rec.analog[idx], dtype=np.float64)
            # library already applied cfg 'a' coeff; only apply user override
            scale = CHANNEL_SCALE[key]
            return arr * scale

        ch = {k: _get(k) for k in ("ia", "ib", "ic", "ua", "ub", "uc")}

        # positional fallback for missing currents
        for i, k in enumerate(("ia", "ib", "ic")):
            if ch[k] is None and i < len(rec.analog):
                ch[k] = np.asarray(rec.analog[i], dtype=np.float64) * CHANNEL_SCALE[k]

        return fs_hz, ch

    except Exception:
        pass  # fall through

    # ---- minimal built-in ASCII parser (no external deps) ------------------
    import re
    import numpy as np

    cfg_text  = cfg_path.read_text(encoding="utf-8", errors="replace")
    cfg_lines = cfg_text.splitlines()

    # line index 1: "TT,nA,nD"
    m = re.match(r"(\d+),(\d+)A,(\d+)D",
                 (cfg_lines[1] if len(cfg_lines) > 1 else "").strip())
    num_analog  = int(m.group(2)) if m else 3

    # analog channel lines: 2 .. 2+num_analog-1
    # COMTRADE99/2013: n,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS
    analog_names  = []
    analog_a_coef = []
    for k in range(2, 2 + num_analog):
        if k < len(cfg_lines):
            parts = cfg_lines[k].split(",")
            analog_names.append(parts[1].strip() if len(parts) > 1 else f"CH{k-2}")
            try:
                analog_a_coef.append(float(parts[5]) if len(parts) > 5 else 1.0)
            except ValueError:
                analog_a_coef.append(1.0)
        else:
            analog_names.append(f"CH{k-2}")
            analog_a_coef.append(1.0)

    names_up = [n.upper() for n in analog_names]

    # sampling rate: first line matching "<float>,<int>" with value > 100
    fs_hz = 10000.0
    for line in cfg_lines:
        m = re.match(r"^([0-9.]+),(\d+)$", line.strip())
        if m:
            cand = float(m.group(1))
            if cand > 100:
                fs_hz = cand
                break

    # read .dat
    dat_path = cfg_path.with_suffix(".dat")
    if not dat_path.exists():
        dat_path = cfg_path.with_suffix(".DAT")
    if not dat_path.exists():
        raise FileNotFoundError(f".dat not found next to {cfg_path.name}")

    rows = []
    with dat_path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip().replace(",", " ")
            if not line:
                continue
            try:
                rows.append(list(map(float, line.split())))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric data in {dat_path.name}")

    arr     = np.array(rows, dtype=np.float64)
    ch_col0 = 2          # columns: [sample_no, timestamp_us, ch0, ch1, ...]
    n_avail = arr.shape[1] - ch_col0

    def _get_raw(key, fallback_pos):
        idx = _find_idx(names_up, key)
        if idx is None:
            idx = min(fallback_pos, n_avail - 1)
        if idx >= n_avail:
            return None
        # scaling: user override takes priority; else use 'a' from .cfg
        user_scale = CHANNEL_SCALE[key]
        a = analog_a_coef[idx] if idx < len(analog_a_coef) else 1.0
        effective  = user_scale if user_scale != 1.0 else a
        return arr[:, ch_col0 + idx] * effective

    ch = {
        "ia": _get_raw("ia", 0),
        "ib": _get_raw("ib", 1),
        "ic": _get_raw("ic", 2),
        "ua": _get_raw("ua", 3) if n_avail > 3 else None,
        "ub": _get_raw("ub", 4) if n_avail > 4 else None,
        "uc": _get_raw("uc", 5) if n_avail > 5 else None,
    }
    return fs_hz, ch


def _list_channels_from_cfg(cfg_path: Path) -> list:
    """Return list of (index, name) for all analog channels in .cfg."""
    import re
    lines = cfg_path.read_text(encoding="utf-8", errors="replace").splitlines()
    m = re.match(r"(\d+),(\d+)A,(\d+)D",
                 (lines[1] if len(lines) > 1 else "").strip())
    num_analog = int(m.group(2)) if m else 0
    result = []
    for k in range(2, 2 + num_analog):
        if k < len(lines):
            parts = lines[k].split(",")
            name = parts[1].strip() if len(parts) > 1 else f"CH{k-2}"
            result.append((k - 2, name))
    return result


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

    # ---- channel listing (so user can fill CHANNEL_NAMES) ------------------
    print()
    print("=" * 62)
    print(" CHANNEL LISTING — copy exact names into CHANNEL_NAMES if needed")
    print("=" * 62)
    for cfg_path in cfg_files:
        chs = _list_channels_from_cfg(cfg_path)
        print(f"  {cfg_path.name}:")
        for idx, name in chs:
            print(f"    [{idx:>2}]  {name}")
    print()

    # ---- results table -----------------------------------------------------
    col_f  = 38
    col_s  =  8
    col_ms = 10
    col_hz = 10
    sep    = "-" * (col_f + col_s + col_ms + col_hz + 10)

    print(
        f"{'File':<{col_f}} | "
        f"{'sample':>{col_s}} | "
        f"{'t0, ms':>{col_ms}} | "
        f"{'fs, Hz':>{col_hz}}"
    )
    print(sep)

    ok_count  = 0
    err_count = 0
    results   = []

    for cfg_path in cfg_files:
        fname = cfg_path.name
        try:
            fs_hz, ch = _parse_comtrade(cfg_path)

            ia, ib, ic = ch["ia"], ch["ib"], ch["ic"]
            if ia is None or ib is None or ic is None:
                raise ValueError(
                    "Could not resolve one or more current channels.\n"
                    "        Run the script once to see channel names, then fill "
                    "CHANNEL_NAMES['ia'/'ib'/'ic'] at the top of this file."
                )

            params    = FaultInceptionParams(fs_hz=fs_hz, **PARAMS_OVERRIDE)
            currents  = np.stack([ia, ib, ic], axis=0)   # (3, T)
            t0_sample = detect_t0_multi_phase(currents, params)

            if t0_sample is None:
                row = (
                    f"{fname:<{col_f}} | "
                    f"{'NOT FOUND':>{col_s}} | "
                    f"{'---':>{col_ms}} | "
                    f"{fs_hz:>{col_hz}.0f}"
                )
                err_count += 1
            else:
                t0_ms = t0_sample / fs_hz * 1000.0
                row = (
                    f"{fname:<{col_f}} | "
                    f"{t0_sample:>{col_s}d} | "
                    f"{t0_ms:>{col_ms}.3f} | "
                    f"{fs_hz:>{col_hz}.0f}"
                )
                ok_count += 1

            results.append((fname, row, None))
            print(row)

        except Exception as exc:
            err_row = (
                f"{fname:<{col_f}} | "
                f"{'ERROR':>{col_s}} | "
                f"{'---':>{col_ms}} | "
                f"{'---':>{col_hz}}"
            )
            results.append((fname, err_row, str(exc)))
            print(err_row)
            print(f"  ↳ {exc}")
            err_count += 1

    print(sep)
    print(f"\nTotal: {len(cfg_files)} files   OK: {ok_count}   Failed / Not found: {err_count}")
    print()

    verbose_errors = [(f, e) for f, _, e in results if e is not None]
    if verbose_errors:
        print("=" * 62)
        print("ERRORS:")
        for fname, err in verbose_errors:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
