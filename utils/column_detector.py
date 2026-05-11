"""
Robust column name detector for oscillogram CSV files.

Key disambiguation rule for numeric phase notation (I1/I2/I3, U1/U2/U3):
  - If ALL THREE numeric variants found (I1+I2+I3) → they are PHASES (Ia,Ib,Ic)
  - If only TWO found (I1+I2, no I3) → they are SEQUENCE COMPONENTS, not phases
    In that case, fall back to letter-based detection (Ia/Ib/Ic).
"""

import re
from typing import Optional


# ─── Синонимы фаз ─────────────────────────────────────────────────────────────
_PHASE_SYNONYMS = {
    'A': ['A', 'a', 'А', 'а'],           # только буквы, цифру 1 — отдельно
    'B': ['B', 'b', 'В', 'в', 'Б', 'б'],
    'C': ['C', 'c', 'С', 'с'],
}

# Цифровые обозначения фаз (используются ТОЛЬКО если все три присутствуют)
_PHASE_DIGITS = {'A': '1', 'B': '2', 'C': '3'}

# Синонимы типа сигнала
_TYPE_SYNONYMS = {
    'I': ['I', 'i', 'И', 'и'],
    'U': ['U', 'u', 'V', 'v', 'У', 'у'],
}


def _classify_numeric_columns(candidates: list) -> dict:
    """
    Check if numeric-phase columns (I1/I2/I3, U1/U2/U3) represent
    PHASES or SEQUENCE COMPONENTS.

    Returns
    -------
    dict with keys 'I_numeric_ok' and 'U_numeric_ok' (bool each).
    True  = all three numeric variants found → treat as phases.
    False = incomplete set → numeric digits are sequence labels, ignore them.
    """
    result = {}
    for sig_type, type_chars in _TYPE_SYNONYMS.items():
        found_digits = set()
        for c in candidates:
            s = c.strip()
            if len(s) < 2:
                continue
            # Последние 2 символа: type_char + digit
            if s[-2] in type_chars and s[-1] in ('1', '2', '3'):
                found_digits.add(s[-1])
            # Или digit + type_char в конце
            if s[-2] in ('1', '2', '3') and s[-1] in type_chars:
                found_digits.add(s[-2])

        # Все три цифры присутствуют → это фазы
        result[f'{sig_type}_numeric_ok'] = found_digits >= {'1', '2', '3'}

    return result


def _ends_with(col: str, type_chars: list, phase_chars: list,
               allow_digits: bool = True) -> bool:
    """
    Check if column name ends with [type][phase] or [phase][type].

    Parameters
    ----------
    allow_digits : bool
        If False, digit-based phase matching is skipped
        (used when digits are sequence labels, not phases).
    """
    s = col.strip()
    if len(s) < 1:
        return False

    # Строим полный список символов фазы
    search_chars = list(phase_chars)

    last2 = s[-2:] if len(s) >= 2 else ''

    # Прямая проверка последних 2 символов
    if len(last2) == 2:
        t, p = last2[0], last2[1]
        if t in type_chars and p in search_chars:
            return True
        if t in search_chars and p in type_chars:
            return True

    # Regex: type_char + 0-3 разделителя + phase_char в конце строки
    for tc in type_chars:
        for pc in search_chars:
            if re.search(re.escape(tc) + r'[\W_]{0,3}' + re.escape(pc) + r'$', s):
                return True
            if re.search(re.escape(pc) + r'[\W_]{0,3}' + re.escape(tc) + r'$', s):
                return True

    return False


def _score_match(col: str, type_chars: list, phase_chars: list) -> int:
    """Higher score = better match. Prefers clean short names."""
    score = 0
    s = col.strip()
    if len(s) >= 2:
        if s[-2] in type_chars and s[-1] in phase_chars:
            score += 10
        elif s[-2] in phase_chars and s[-1] in type_chars:
            score += 10
    score -= len(s)
    return score


def detect_signal_columns(
    columns: list,
    distance_col: Optional[str] = None,
) -> dict:
    """
    Detect which CSV column corresponds to each of the 6 signal channels.

    Disambiguation rule for numeric naming:
      - I1 + I2 + I3 all present → phases (Ia=I1, Ib=I2, Ic=I3)
      - Only I1 + I2 (no I3)     → sequence components, NOT phases;
        fall back to letter detection (Ia/Ib/Ic)

    Parameters
    ----------
    columns : list[str]
        All column names from the CSV.
    distance_col : str, optional
        Distance column to exclude. Auto-detected if None.

    Returns
    -------
    dict with keys ['Ia', 'Ib', 'Ic', 'Ua', 'Ub', 'Uc']

    Raises
    ------
    ValueError if any channel cannot be identified.

    Examples
    --------
    >>> detect_signal_columns(['distance_km', 'CT1IA', 'CT1IB', 'CT1IC',
    ...                        'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC'])
    {'Ia': 'CT1IA', 'Ib': 'CT1IB', 'Ic': 'CT1IC',
     'Ua': 'S1) BUS1UA', 'Ub': 'S1) BUS1UB', 'Uc': 'S1) BUS1UC'}

    >>> # I1+I2 without I3 → sequence labels, use letter-based detection
    >>> detect_signal_columns(['dist_km', 'Ia', 'Ib', 'Ic',
    ...                        'Ua', 'Ub', 'Uc', 'I1', 'I2', 'U1', 'U2'])
    {'Ia': 'Ia', 'Ib': 'Ib', 'Ic': 'Ic',
     'Ua': 'Ua', 'Ub': 'Ub', 'Uc': 'Uc'}
    """
    # Исключаем колонку расстояния
    if distance_col is not None:
        candidates = [c for c in columns if c != distance_col]
    else:
        dist_pat = re.compile(r'distance|dist|_km\b|расст', re.IGNORECASE)
        candidates = [c for c in columns if not dist_pat.search(c)]

    # Классифицируем числовые колонки
    numeric_status = _classify_numeric_columns(candidates)

    channels = {
        'Ia': ('I', 'A'), 'Ib': ('I', 'B'), 'Ic': ('I', 'C'),
        'Ua': ('U', 'A'), 'Ub': ('U', 'B'), 'Uc': ('U', 'C'),
    }

    result: dict = {}
    errors: list = []
    used: set = set()

    for ch_name, (sig_type, phase) in channels.items():
        type_chars  = _TYPE_SYNONYMS[sig_type]

        # Решаем: включать ли цифровое обозначение этой фазы
        numeric_ok  = numeric_status[f'{sig_type}_numeric_ok']
        phase_chars = list(_PHASE_SYNONYMS[phase])
        if numeric_ok:
            phase_chars.append(_PHASE_DIGITS[phase])

        matched = [
            c for c in candidates
            if c not in used and _ends_with(c, type_chars, phase_chars)
        ]

        if len(matched) == 0:
            errors.append(f"  '{ch_name}': no match among {candidates}")
        elif len(matched) == 1:
            result[ch_name] = matched[0]
            used.add(matched[0])
        else:
            best = max(matched, key=lambda c: _score_match(c, type_chars, phase_chars))
            result[ch_name] = best
            used.add(best)
            import warnings
            warnings.warn(
                f"Column '{ch_name}': multiple matches {matched}, "
                f"selected: '{best}'",
                stacklevel=2,
            )

    if errors:
        raise ValueError(
            "Could not detect all signal columns:\n" + "\n".join(errors) +
            f"\n\nAvailable columns: {list(columns)}"
        )

    return result


def detect_distance_column(columns: list) -> str:
    """
    Auto-detect the distance/target column.

    Raises
    ------
    ValueError if not found.
    """
    pat = re.compile(r'distance|dist|_km\b|расст', re.IGNORECASE)
    matched = [c for c in columns if pat.search(c)]

    if len(matched) == 1:
        return matched[0]
    elif len(matched) == 0:
        raise ValueError(
            f"Cannot detect distance column. Available: {list(columns)}"
        )
    else:
        for preferred in ('distance_km', 'dist_km', 'distance'):
            if preferred in matched:
                return preferred
        return matched[0]
