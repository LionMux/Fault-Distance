# Обзор: математика pipeline — от COMTRADE до тензора в модель

## Входные данные
- Формат: COMTRADE (`.cfg` + `.dat`)
- Содержимое: осциллограммы токов и напряжений фаз A, B, C
- Частота дискретизации: `fs` [Гц] (обычно 2000–5000 Гц)
- Длительность: ~200–400 мс

## Pipeline (последовательность этапов)

```
COMTRADE (.cfg + .dat)
    ↓  tools/data_comtrade_to_csv.py
CSV файл (столбцы: time, CT1IA, CT1IB, CT1IC, S1) BUS1UA, S1) BUS1UB, S1) BUS1UC, distance_km, fs_hz)
    ↓  data/dataset.py  (FaultDataset.__init__)
1. Центрирование по предыстории (center_by_prehistory)
2. Удаление DC-составляющей (remove_dc_period)
3. Детекция момента КЗ и кадрирование (detect_t0_and_crop)  [опционально]
4. Pad / trim до SEQ_LENGTH
5. Симметричные составляющие (sliding_window_symseq)  [опционально]
6. Нормализация (standard или p.u.)
    ↓
Тензор (NUM_CHANNELS, SEQ_LENGTH) → модель PyTorch
```

## Файлы документации

| Файл | Содержимое |
|---|---|
| `00_overview.md` | Этот файл — обзор pipeline |
| `01_comtrade_to_csv.md` | Конвертация COMTRADE → CSV |
| `02_centering.md` | Центрирование по предыстории |
| `03_dc_removal.md` | Удаление DC-составляющей (скользящее среднее) |
| `04_t0_detection.md` | Детекция момента КЗ (RMS-based) |
| `05_windowing.md` | Кадрирование окна [pre_fault, post_fault] |
| `06_symmetrical_components.md` | Симметричные составляющие (sliding-window DFT + Fortescue) |
| `07_normalization.md` | Нормализация: standard (z-score) vs p.u. |
| `08_tensor_format.md` | Финальный формат тензора |

---

## Параметры по умолчанию (из config.py)

| Параметр | Значение | Описание |
|---|---|---|
| `SEQ_LENGTH` | 400 | Длина окна в отсчётах |
| `SAMPLING_FREQ_HZ` | 5000 (fallback) | Частота дискретизации |
| `MAINS_FREQ_HZ` | 50 | Сетевая частота |
| `T0_PRE_MS` | 50 | Предыстория перед КЗ [мс] |
| `T0_POST_MS` | 150 | Постистория после КЗ [мс] |
| `T0_ETA_I` | 0.5 | Порог роста тока |
| `T0_ETA_U` | 0.85 | Порог падения напряжения |
| `NORMALIZATION_MODE` | 'standard' | 'standard' или 'pu' |

---

**Примечание:** Все формулы приведены в файлах `01_*.md` – `08_*.md`.
