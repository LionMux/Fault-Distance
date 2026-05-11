# Этап 4: Кадрирование окна вокруг t₀ (crop_around_t0)

## Назначение
Выделение релевантного фрагмента осциллограммы вокруг момента КЗ фиксированной длины.

## Математика

### Размеры окна в отсчётах
```
pre_samp  = round(pre_fault_ms  · 10⁻³ · fs)
post_samp = round(post_fault_ms · 10⁻³ · fs)
```

### Границы окна
```
start = max(0, t₀ - pre_samp)
end   = min(N, t₀ + post_samp)
```

### Извлечение окна
```
window = x[start:end]   // форма (T_crop, C)
```

### Приведение к целевой длине (resample)
Если `target_length` задана и `T_crop ≠ target_length`:
```
window_resampled = resample(window, target_length, axis=0)
```

где `resample` — линейная интерполяция scipy.signal.resample.

### Позиция t₀ в новом окне
```
total = pre_samp + post_samp
frac = pre_samp / total
t₀_local = round(frac · (target_length - 1))
```

## Параметры
| Параметр | Значение | Описание |
|---|---|---|
| `pre_fault_ms` | 50.0 | Предыстория перед КЗ [мс] |
| `post_fault_ms` | 150.0 | Постистория после КЗ [мс] |
| `target_length` | 400 | Целевая длина [отсчёты] |
| `fs` | из CSV | Частота дискретизации [Гц] |

## Пример
При `fs = 2000` Гц:
- `pre_samp = round(50 · 0.001 · 2000) = 100` отсчётов
- `post_samp = round(150 · 0.001 · 2000) = 300` отсчётов
- `T_crop = 100 + 300 = 400` отсчётов = 200 мс
- Это совпадает с `SEQ_LENGTH = 400` → resample не нужен

При `fs = 5000` Гц:
- `pre_samp = 250`, `post_samp = 750`
- `T_crop = 1000` отсчётов
- Resample до 400 отсчётов

## Pad / Trim (fallback)
Если t₀ не обнаружен:
```
если T < SEQ_LENGTH:
    pad нулями до SEQ_LENGTH
если T > SEQ_LENGTH:
    обрезать до первых SEQ_LENGTH отсчётов
```

## Свойства
- **Фиксированная длина выхода**: всегда `SEQ_LENGTH` отсчётов
- **Сохранение пропорций**: t₀ находится на фиксированной позиции (~25% от начала)
- **Resampling**: адаптация к разным частотам дискретизации
