"""
Tests for column_detector — must handle all real-world naming variants.
Run with: pytest src/fault_distance/utils/tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from fault_distance.utils.column_detector import detect_signal_columns, detect_distance_column

EXPECTED_KEYS = {'Ia', 'Ib', 'Ic', 'Ua', 'Ub', 'Uc'}


class TestRealWorldFormats:

    def test_your_csv_format(self):
        """Твой реальный формат из CSV."""
        cols = ['distance_km', 'CT1IA', 'CT1IB', 'CT1IC',
                'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert r == {
            'Ia': 'CT1IA', 'Ib': 'CT1IB', 'Ic': 'CT1IC',
            'Ua': 'S1) BUS1UA', 'Ub': 'S1) BUS1UB', 'Uc': 'S1) BUS1UC',
        }

    def test_old_format_no_space(self):
        """Старый формат без пробела."""
        cols = ['distance_km', 'CT1IA', 'CT1IB', 'CT1IC',
                'S1)BUS1UA', 'S1)BUS1UB', 'S1)BUS1UC']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert set(r.keys()) == EXPECTED_KEYS

    def test_simple_lowercase(self):
        cols = ['distance_km', 'ia', 'ib', 'ic', 'ua', 'ub', 'uc']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert r['Ia'] == 'ia'
        assert r['Ua'] == 'ua'

    def test_simple_uppercase(self):
        cols = ['distance_km', 'IA', 'IB', 'IC', 'UA', 'UB', 'UC']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert set(r.keys()) == EXPECTED_KEYS

    def test_numeric_phases(self):
        """I1/I2/I3 и U1/U2/U3."""
        cols = ['dist_km', 'I1', 'I2', 'I3', 'U1', 'U2', 'U3']
        r = detect_signal_columns(cols)
        assert set(r.keys()) == EXPECTED_KEYS

    def test_mixed_case(self):
        cols = ['distance_km', 'Ia', 'Ib', 'Ic', 'Ua', 'Ub', 'Uc']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert r['Ia'] == 'Ia'

    def test_cyrillic_phase_letters(self):
        """Кириллические А/В/С в названиях каналов."""
        # С — кириллическая буква (U+0421), А — кириллическая (U+0410)
        cols = ['distance_km', 'IА', 'IВ', 'IС', 'UА', 'UВ', 'UС']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert set(r.keys()) == EXPECTED_KEYS

    def test_voltage_as_V(self):
        """Напряжения обозначены через V."""
        cols = ['distance_km', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert r['Ua'] == 'Va'


class TestDistanceDetection:

    def test_distance_km(self):
        assert detect_distance_column(['distance_km', 'Ia', 'Ib']) == 'distance_km'

    def test_dist_km(self):
        assert detect_distance_column(['dist_km', 'Ia', 'Ib']) == 'dist_km'

    def test_not_found(self):
        with pytest.raises(ValueError):
            detect_distance_column(['Ia', 'Ib', 'Ic'])


class TestEdgeCases:

    def test_auto_distance_exclusion(self):
        """Без явного distance_col — автодетект исключения."""
        cols = ['distance_km', 'CT1IA', 'CT1IB', 'CT1IC',
                'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
        r = detect_signal_columns(cols)  # без distance_col
        assert set(r.keys()) == EXPECTED_KEYS

    def test_raises_on_missing(self):
        """Только 5 каналов — должен поднять ValueError."""
        cols = ['distance_km', 'CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB']
        with pytest.raises(ValueError, match="Uc"):
            detect_signal_columns(cols, distance_col='distance_km')

class TestNumericDisambiguation:
    """I1/I2/I3 — все три → фазы. I1/I2 без I3 → последовательности, не фазы."""

    def test_all_three_numeric_are_phases(self):
        """I1+I2+I3 и U1+U2+U3 → это фазы A/B/C."""
        cols = ['distance_km', 'I1', 'I2', 'I3', 'U1', 'U2', 'U3']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert r == {
            'Ia': 'I1', 'Ib': 'I2', 'Ic': 'I3',
            'Ua': 'U1', 'Ub': 'U2', 'Uc': 'U3',
        }

    def test_i1_i2_without_i3_are_sequences(self):
        """I1+I2 без I3 → это последовательности, фазы ищем по буквам."""
        cols = ['distance_km', 'Ia', 'Ib', 'Ic', 'Ua', 'Ub', 'Uc',
                'I1', 'I2', 'U1', 'U2']
        r = detect_signal_columns(cols, distance_col='distance_km')
        # Фазы должны найтись по буквам, не по цифрам
        assert r == {
            'Ia': 'Ia', 'Ib': 'Ib', 'Ic': 'Ic',
            'Ua': 'Ua', 'Ub': 'Ub', 'Uc': 'Uc',
        }

    def test_mixed_numeric_and_letter(self):
        """Только токи числовые (I1/I2/I3), напряжения буквенные."""
        cols = ['distance_km', 'I1', 'I2', 'I3', 'Ua', 'Ub', 'Uc']
        r = detect_signal_columns(cols, distance_col='distance_km')
        assert r['Ia'] == 'I1'
        assert r['Ib'] == 'I2'
        assert r['Ic'] == 'I3'
        assert r['Ua'] == 'Ua'
