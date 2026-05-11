# Fault-Distance — справка для AI-агентов

> Проект предсказывает расстояние до места короткого замыкания (КЗ) по осциллограммам токов и напряжений с помощью 1D-CNN / ResNet1D (PyTorch).
> Язык проекта: Python, документация и визуализации — на русском.

---

## Обзор проекта

Каждый CSV-файл — это один осциллограф (одно событие КЗ). Модель получает на вход тензор `(B, NUM_CHANNELS, SEQ_LENGTH)` и предсказывает скаляр — расстояние до КЗ в километрах (регрессия).

### Формат входных CSV

Обязательные столбцы:
- `distance_km` — метка (одинакова во всех строках файла)
- `CT1IA`, `CT1IB`, `CT1IC` — мгновенные токи фаз A/B/C [А]
- `S1) BUS1UA`, `S1) BUS1UB`, `S1) BUS1UC` — мгновенные напряжения фаз A/B/C [кВ]
- `fs_hz` — частота дискретизации (записывается `comtrade_to_csv.py`; для старых файлов используется fallback `SAMPLING_FREQ_HZ`)

Количество строк на файл рекомендуется `SEQ_LENGTH = 400`.

### Технологический стек

- **PyTorch** >= 2.0.0 (основной фреймворк)
- **NumPy, Pandas, SciPy, scikit-learn** — данные и предобработка
- **Matplotlib, Seaborn** — визуализация (DPI=300, русские подписи)
- **PyYAML** — конфигурация экспериментов
- **pytest** — тестирование
- **tqdm** — прогресс-бары
- **comtrade** (опционально) — чтение COMTRADE-файлов

Проект не использует `pyproject.toml`, `setup.py` или `package.json`; зависимости задаются в `requirements.txt`, а запуск производится напрямую через `python <script>.py`.

---

## Структура каталогов

```
Fault-Distance/
├── .agents/
│   └── skills/               # AI-навыки (auto-orchestrator, ml-engineer, web-search, ...)
├── .kilo/                    # Метаданные агента Kilo
├── .sixth/                   # Метаданные агента Sixth
├── .vscode/
│   └── settings.json         # Настройки VS Code
├── checkpoints/              # Сохранённые веса (.pth)
├── configs/                  # YAML-конфиги экспериментов
│   ├── base.yaml             # Базовый конфиг (не запускать напрямую)
│   ├── augment_train_cnn1d.yaml
│   ├── augment_train_resnet1d.yaml
│   └── activation_smoke.yaml
├── data/
│   ├── __init__.py
│   ├── augmentation.py       # TimeShift + GaussianNoise + AugmentationPipeline
│   ├── dataset.py            # FaultDataset + DataLoaderFactory
│   ├── fault_classifier.py   # (удалён) классификация КЗ больше не используется
│   ├── fault_inception.py    # Детекция момента КЗ (t0) и кадрирование
│   ├── preprocessing.py      # Фильтры, sliding-window symseq, DataPreprocessor
│   ├── csv_all/              # Полный набор CSV (~100 файлов, тренировка + тест)
│   ├── data_test/            # COMTRADE-файлы для тестирования (.cfg + .dat)
│   ├── data_test_csv/        # CSV для тестирования
│   └── data_training/        # CSV для обучения
├── logs/                     # Логи обучения (run_YYYYMMDD_HHMMSS/)
│   └── run_.../              # training.log, .png, activations/ (опционально)
├── logs_smoke/               # Логи smoke-тестов
│   └── run_.../
│       └── activations/      # epoch_curves/, snapshots/, epoch_stats.csv
├── output/                   # Сгенерированные артефакты (не коммитятся)
│   ├── thesis/               # Графики для диплома
│   ├── pipeline/             # Визуализации этапов предобработки
│   └── symseq/               # Экспортированные COMTRADE симм. составляющих
├── models/
│   ├── __init__.py
│   ├── blocks.py             # SEBlock1D, ResBlock1D, InvertedResBlock1D
│   ├── cnn1d.py              # CNN1D, DilatedCNN1D, CNN1DRegressor
│   └── resnet1d.py           # FaultResNet1D (ResNet + SE-блоки)
├── plots_pipeline/           # Визуализация шагов предобработки
│   ├── __init__.py
│   ├── demo.py
│   ├── plot_augmentation.py
│   ├── plot_engine.py
│   ├── plot_preprocessing.py
│   └── plot_signals.py
├── scripts/
│   ├── augment_and_train.py  # Полный pipeline: split → augment → train
│   ├── compare_models.py     # Сравнение нескольких архитектур на одних данных
│   ├── health_check_training.py  # Preflight-проверка (forward, backward, checkpoint, inference)
│   └── visualize_augmentation.py
├── symseq/                   # Симметричные составляющие (Fortescue)
│   ├── __init__.py
│   ├── adapter.py            # Batch adapter: (B,6,N) → symseq features
│   ├── core.py               # abc_to_seq / seq_to_abc / batch transforms
│   ├── fourier.py            # Оценка фазоров через FFT
│   ├── power_systems.py      # symseq_from_waveforms
│   └── tests/                # test_core.py, test_fourier.py, test_adapter.py
├── tests/                    # pytest-тесты
│   ├── fault_inception/      # Ручные проверки t0 (check_t0.py, oscillograms/, README.md)
│   ├── test_activation_recorder.py
│   ├── test_augmentation.py
│   └── test_preprocessing_pipeline.py
├── tmp/                      # Временные файлы
│   └── temp_smoke_train_run.py
├── tools/
│   ├── __init__.py
│   ├── README
│   ├── comtrade_to_csv.py    # Конвертер COMTRADE → CSV
│   ├── debug_fortescue.py
│   ├── example_usage.py
│   ├── inspect_symseq.py
│   └── symseq_to_comtrade.py
├── utils/
│   ├── __init__.py
│   ├── activation_export.py  # Экспорт активаций (PNG/CSV)
│   ├── activation_recorder.py# Запись активаций по слоям (ActivationRecorder)
│   ├── column_detector.py    # Автоопределение имён столбцов CSV
│   ├── logger.py             # TrainingLogger
│   ├── comparison_plots.py   # Сравнительные графики для нескольких моделей
│   ├── metrics.py            # MAE, MSE, RMSE, R², MAPE
│   ├── plots.py              # Графики для диплома (русские подписи, DPI=300)
│   └── probe_selection.py    # Выбор probe-батчей
├── AGENTS.md                 # Этот файл
├── IMPROVEMENTS.md           # Список запланированных улучшений
├── README.md                 # Основная документация проекта

├── config.py                 # Config dataclass + YAML-loader
├── implementation_plan.md    # План реализации диплома
├── inference.py              # Инференс на один CSV
├── requirements.txt          # Зависимости Python
├── test.py                   # Батчевое тестирование
└── train.py                  # Основной скрипт обучения
```

---

## Система конфигурации

Конфигурация задаётся в `config.py` (класс `Config`) и может быть переопределена через:

1. **Python API:** `get_config(NUM_EPOCHS=200, BATCH_SIZE=64)`
2. **CLI train.py:** `python train.py --model resnet1d --epochs 100 --batch-size 32`
3. **YAML:** `python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml`
4. **YAML + CLI override:** `--set training.num_epochs=200 model.dropout=0.1`

Приоритет (от низшего к высшему): defaults → `configs/base.yaml` → experiment YAML → CLI `--set` overrides.

Ключевые поля `Config`:
- `MODEL_TYPE`: `'cnn1d' | 'dilated_cnn1d' | 'resnet1d'`
- `NUM_CHANNELS`: 6 (фазные) или 12 (фазные + симметричные составляющие)
- `SEQ_LENGTH`: 400
- `NORMALIZATION_MODE`: `'standard'` (StandardScaler + MinMaxScaler) или `'pu'` (физическая нормировка)
- `SYMSEQ_ENABLED`: True → датасет автоматически дополняет 6 каналов симметричными составляющими
- `T0_ENABLED`: True → автообрезка по моменту КЗ
- `REMOVE_DC_ENABLED`: True → удаление апериодической составляющей (скользящее среднее за период)

---

## Предобработка и аугментация

### Предобработка сигналов
- **Remove DC period** (`REMOVE_DC_ENABLED`) — **основной метод** удаления апериодической составляющей: вычитание скользящего среднего за период сети (20 мс для 50 Гц). Не вносит фазовых искажений. Перед ним выполняется `center_by_prehistory` (центрирование по первым 20 мс).
- **Fault inception (t0)** (`T0_ENABLED`) — двухступенчатый алгоритм (D4 + cycle-difference) для обнаружения момента КЗ и кадрирования окна `[pre_fault_ms, post_fault_ms]`.
- **Symmetrical components** (`SYMSEQ_ENABLED`) — sliding-window DFT даёт time-varying магнитуды симметричных составляющих `|I1|,|I2|,|I0|,|U1|,|U2|,|U0|`, которые конкатенируются к 6 фазным каналам (итого 12).

### Аугментация (`data/augmentation.py`)
- **TimeShift**: сдвиг осциллограммы влево/вправо с паддингом первой/последней строки; сохраняется `SEQ_LENGTH` строк.
- **GaussianNoise**: добавление белого шума с уровнями SNR `[1, 5, 10, 20, 40]` дБ.
- **AugmentationPipeline**: для каждого исходного файла создаёт `2 × 5 shifts × 5 SNR = 50` аугментированных копий.

---

## Модели

| Модель | Вход | Описание |
|---|---|---|
| `cnn1d` | `(B, C, 400)` | 3 блока Conv1d→BN→ReLU→MaxPool→Dropout, затем FC 256→128→1 |
| `dilated_cnn1d` | `(B, C, 400)` | Расширенные свёртки (dilations=[1,2,4,8]), больший receptive field |
| `resnet1d` | `(B, C, 400)` | ResNet с SE-блоками, depth=1..4, GAP + head |

`C` = `NUM_CHANNELS` (6 или 12 в зависимости от `SYMSEQ_ENABLED`).

---

## Команды сборки и запуска

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Обучение
```bash
# Базовый запуск
python train.py

# С параметрами
python train.py --model cnn1d --epochs 100 --batch-size 32 --lr 0.001

# YAML-конфиг
python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml

# YAML + override
python scripts/augment_and_train.py --config configs/augment_train_cnn1d.yaml \
    --set training.num_epochs=200
```

### Инференс
```bash
python inference.py --model checkpoints/best_model.pth --csv data/data_training/1A_0.5km.csv --has-labels --device cpu
```

### Тестирование (батч)
```bash
python test.py
```
Перед запуском положите тестовые CSV в `data/data_test_csv/` и убедитесь, что `CHECKPOINT` в `test.py` указывает на нужный .pth.

### Сравнение моделей
```bash
# Сравнить CNN1D и ResNet1D на 2 эпохах (smoke-test)
python scripts/compare_models.py --models cnn1d resnet1d --epochs 2

# Полное сравнение всех трёх архитектур
python scripts/compare_models.py --models cnn1d resnet1d dilated_cnn1d --epochs 100 --config configs/base.yaml

# С внешним тестовым набором
python scripts/compare_models.py --models cnn1d resnet1d --epochs 50 --test-dir data/data_test_csv/
```
Скрипт последовательно обучает каждую модель на одних и тех же данных, запускает inference, строит сравнительные графики и определяет лучшую модель по MAE. Результаты сохраняются в:
- `logs/comparison_YYYYMMDD_HHMMSS/` — логи, графики, чекпоинты
- `output/thesis/model_comparison/` — копии сравнительных графиков
- `output/thesis/best_model/` — чекпоинт и графики лучшей модели

### Preflight health-check
```bash
python scripts/health_check_training.py --config configs/base.yaml
```
Проверяет: один forward/backward шаг, сохранение/загрузку чекпоинта, инференс на одном CSV.

### Конвертация COMTRADE → CSV
```bash
# Настроить tools/data_comtrade_config.ini, затем:
python tools/data_comtrade_to_csv.py
```

### Аугментация данных
```bash
python data/augmentation.py --input data/data_training --output data/data_augmented
```

### Тесты
```bash
pytest tests/ -v
```

---

## Экспорт активаций (дипломная функция)

Включить в YAML:
```yaml
activation_export:
  enabled: true
  layers: auto
  capture_epochs: ["first", "best", "every_n:5"]
  export_formats: ["png", "csv"]
```

Результаты сохраняются в `logs/run_.../activations/`:
- `snapshots/` — PNG-кривые активаций по слоям для probe-выборки.
- `snapshots.csv` — scalar stats (mean, std, rms, energy, ...).
- `epoch_stats.csv` — эпоховая агрегация метрик по слоям.
- `epoch_curves/` — графики эпоховых метрик для каждого слоя.

Probe выбирается детерминированно (`first_val_sample` по умолчанию) из валидационного набора.

---

## Организация кода и соглашения

### Импорты
- В `scripts/` и `tests/` часто используется `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))` для импорта корневых модулей.
- Модели импортируются напрямую: `from models.cnn1d import CNN1D, DilatedCNN1D`.

### Стиль кода
- **Docstrings**: Google-style или NumPy-style; приветствуется русский язык в документации и визуализациях.
- **Type hints**: активно используются в новых модулях (`utils/activation_recorder.py`, `utils/probe_selection.py` и др.).
- **Именование**: `CamelCase` для классов, `snake_case` для функций/переменных, `UPPER_CASE` для констант конфигурации.
- **Визуализация**: `matplotlib.use('Agg')`; DPI=300; русские подписи осей и легенд; палитра согласована через `_COLOR_*` константы в `utils/plots.py`.

### Нормализация
- **Стандартная** (`NORMALIZATION_MODE='standard'`): per-channel `StandardScaler` для сигналов, `MinMaxScaler` для расстояния → `[0, 1]`.
- **p.u.** (`NORMALIZATION_MODE='pu'`): физическая нормировка по базисным значениям. Токи делятся на `Ibase = S_base_MVA·10⁶ / (√3·Unom·10³)`, напряжения делятся на `Ubase = Unom·10³ / √3`, расстояние делится на `LINE_L_KM`. Не требует fitted scalers — подходит для inference на новых линиях.

### Чекпоинты
Сохраняются в `checkpoints/` и содержат:
```python
{
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'config': Config,
    'scalers': {'signal': [...], 'distance': MinMaxScaler}
}
```
При загрузке в `inference.py` и `test.py` скалеры используются для обратной нормализации предсказаний.

### Логи и артефакты
Каждый запуск `train.py` создаёт уникальную папку:
```
logs/run_YYYYMMDD_HHMMSS/
├── training_YYYYMMDD_HHMMSS.log
├── training_history.png
├── predictions.png
└── metrics_summary.png
```

При включённом `activation_export` добавляется:
```
logs/run_.../activations/
├── snapshots/                # PNG-кривые активаций по слоям
├── snapshots.csv             # scalar stats (mean, std, rms, energy, ...)
├── epoch_stats.csv           # эпоховая агрегация метрик по слоям
└── epoch_curves/             # графики эпоховых метрик для каждого слоя
```

Smoke-тесты (`logs_smoke/`) имеют аналогичную структуру, но обычно содержат меньше эпох.

---

## Тестирование

Тесты написаны на **pytest**.

- `tests/test_preprocessing_pipeline.py` — проверяет `remove_dc_period`, классификацию типа КЗ, формирование 12-канального тензора, вариативность sliding-window symseq.
- `tests/test_augmentation.py` — проверяет time-shift, Gaussian noise, полный augmentation pipeline (создание файлов, сохранение distance_km).
- `tests/test_activation_recorder.py` — проверяет запись активаций, экспорт PNG/CSV, корректность статистик.
- `symseq/tests/test_core.py`, `test_fourier.py`, `test_adapter.py` — тесты симметричных составляющих (Fortescue).
- `tests/test_column_detector.py` — проверка автоопределения имён столбцов CSV.
- `tests/fault_inception/` — ручные проверки детекции t0 (скрипт `check_t0.py` + осциллограммы в `oscillograms/`).

Запуск:
```bash
pytest tests/ -v
```

Некоторые тесты (`TestTwelveChannelDataset`) требуют реальных CSV в `data/csv_all/` и пропускаются (`pytest.skip`), если директория отсутствует.

---

## Безопасность и ограничения

- Проект не содержит сетевых сервисов, секретов или чувствительных данных.
- Все пути к данным относительны рабочей директории.
- `.gitignore` исключает: `checkpoints/`, `logs/`, `*.pth`, `*.csv`, `data/*.csv`.
- При работе с COMTRADE-файлами убедитесь, что `tools/data_comtrade_config.ini` содержит корректную `line_length_km`.

---

## Частые ловушки

1. **Каналы при inference:** если модель обучена с `SYMSEQ_ENABLED=True`, а входной CSV имеет только 6 каналов, `inference.py` и `test.py` автоматически дополняют их sliding-window symseq. Если каналы всё равно не совпадают — скрипт падает с `ValueError`.
2. **Несовпадение `SIGNAL_COLS`:** в старых версиях использовались имена без пробела (`S1)BUS1UA`). Текущий код ожидает `S1) BUS1UA` (с пробелом) — см. `data/dataset.py`.
3. **fs_hz:** новые CSV, созданные `comtrade_to_csv.py`, содержат столбец `fs_hz`. Если его нет, используется `cfg.SAMPLING_FREQ_HZ` (fallback 2000 Гц). Не хардкодьте частоту дискретизации в алгоритмах — читайте её из CSV или cfg.
4. **YAML и `NUM_CHANNELS`:** многие YAML-конфиги задают `data.num_channels: 6`. Если `SYMSEQ_ENABLED=True`, `load_config()` автоматически корректирует `NUM_CHANNELS` до 12.
5. **Augmentation pipeline:** `scripts/augment_and_train.py` создаёт промежуточную папку `*_staging`. Убедитесь, что на диске достаточно места: каждый исходный файл порождает 50 аугментированных копий.
