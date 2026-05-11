# Tools — Утилиты проекта

Каждый скрипт здесь — standalone CLI-утилита. Запускайте напрямую:
`python tools/<script>.py [args]`

## По категориям

### Работа с данными (`data_*`)

| Скрипт | Описание | Зависимости |
|---|---|---|
| `data_comtrade_to_csv.py` | Конвертер COMTRADE (.cfg + .dat) → CSV проекта. Читает настройки из `data_comtrade_config.ini`. | `comtrade` (опционально) |

**Конфигурация:** `data_comtrade_config.ini` — параметры линии (`line_length_km`, `Unom`, `R1_Ом/км`, `X1_Ом/км`).

### Отладка / визуализация (`debug_*`)

| Скрипт | Описание | Зависимости |
|---|---|---|
| `debug_fortescue.py` | Минимальная проверка матрицы Фортескью на синтетике (без CSV, без FFT). | numpy |
| `debug_inspect_symseq.py` | Визуальная проверка симметричных составляющих на реальном CSV. Рисует скользящее окно → FFT → Fortescue. | numpy, pandas, matplotlib |

### Экспорт результатов (`export_*`)

| Скрипт | Описание | Зависимости |
|---|---|---|
| `export_symseq_to_comtrade.py` | Экспорт симметричных составляющих (скользящее окно) обратно в COMTRADE (.cfg + .dat ASCII). | numpy, pandas |
