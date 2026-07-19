# -*- coding: utf-8 -*-
"""
Единые runtime-утилиты для test bench (Фаза 0):
  - DEVICE      : cuda если доступна, иначе cpu (device-agnostic запуск);
  - set_seed()  : фиксация генераторов случайности для воспроизводимости;
  - make_split(): честный train/val/test сплит по файлам (без утечки val==test).

Никаких тяжёлых зависимостей кроме numpy/torch.
"""
import os
import random

import numpy as np
import torch

# --- Устройство: GPU если есть, иначе CPU. Заменяет жёсткие .cuda(). ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_str():
    """Человекочитаемое описание устройства, например 'cuda (NVIDIA GeForce RTX 4090)'."""
    if DEVICE.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    return "cpu (GPU не найдена!)"

# Сид по умолчанию (можно переопределить через переменную окружения BENCH_SEED)
SEED = int(os.environ.get("BENCH_SEED", "42"))


def set_seed(seed=SEED):
    """Фиксирует random/numpy/torch для воспроизводимости. Возвращает использованный seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # детерминизм cuDNN: чтобы разброс по сидам отражал СИД, а не аппаратный шум
        # (важно для парного мульти-сид сравнения целей mel vs Whisper)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def make_split(n_files, test_start):
    """
    Честный сплит по индексам файлов.

      test  = files[test_start:]   -- ОТЛОЖЕННЫЙ тест, только для финального отчёта;
                                       НЕ участвует в early stopping / выборе модели.
      val   = последний train-файл -- для early stopping (выбора модели).
      train = остальные ранние файлы.

    Это убирает баг исходного кода, где X_val и X_test были одним и тем же срезом,
    из-за чего early stopping подсматривал в тестовую выборку.

    Возвращает (train_idx, val_idx, test_idx) — списки индексов файлов.
    """
    # подстраховка для краевых случаев (например, debug с малым числом файлов)
    test_start = max(1, min(int(test_start), n_files - 1))
    test_idx = list(range(test_start, n_files))
    if test_start >= 2:
        val_idx = [test_start - 1]
        train_idx = list(range(0, test_start - 1))
    else:
        # доступен лишь один обучающий файл (только debug/smoke): val совпадает с train.
        # Для реальных метрик такой режим не использовать.
        val_idx = [0]
        train_idx = [0]
    return train_idx, val_idx, test_idx


def shift_ecog_lead(x, lead_ms, frame_rate):
    """Тест «нейро-lead»: сдвиг ВХОДА (ECoG). В продукции мотор/фонология опережают акустику.
    lead_ms>0 => нейроокно берётся на lead_ms РАНЬШЕ относительно цели/слова (нейро ВЕДЁТ
    акустику); lead_ms<0 => позже (перцепционный знак, контроль). Применяется ОДИНАКОВО на
    этапе регрессии и на --mode hidden, поэтому цель и метки слова (по аудио-времени) НЕ
    двигаются — выравнивание стадий сохраняется само. frame_rate = sampling_rate/downsampling_coef (~1 кГц).

    ЗАЧЕМ сдвигаем ВХОД, а не ЦЕЛЬ: единый механизм в обеих стадиях. Сдвиг ЦЕЛИ
    рассинхронил бы stage-1 (цель строится из звука) и stage-2 (метки слова берутся из
    АУДИО-границ через prepare_frames) — пришлось бы согласованно двигать ещё и words_info.
    Сдвиг входа же оставляет всю аудио-таймлинию на месте, поэтому stage-2 не меняется.

    КРАЙ окна заполняется ПОВТОРОМ крайнего реального отсчёта (edge-replication), НЕ нулём.
    Прежняя нулевая заливка давала СТУПЕНЬКУ сигнал->0, которую полосовой/огибающий
    conv-фронт (RF ~40 мс) превращал в ложный переходной ВСПЛЕСК у кадра k (это и был баг).
    Повтор края убирает разрыв: нет ступеньки и нет искусственного нуля во входе свёртки.
    Длина массива не меняется (T) -> downstream обеих стадий не трогается. Остаточный
    артефакт (k кадров «плато» у начала/конца файла) затрагивает лишь ~k крайних окон и
    после полосового conv даёт ~0 без всплеска."""
    if not lead_ms:
        return x
    k = int(round(lead_ms / 1000.0 * frame_rate))
    if k == 0 or abs(k) >= x.shape[0]:   # сдвиг больше длины файла -> тест бессмыслен, не трогаем
        return x
    out = np.roll(x, k, axis=0)          # k>0: out[i]=x[i-k] -> окно заканчивается на k кадров РАНЬШЕ
    if k > 0:
        out[:k] = out[k]                 # повтор первого реального отсчёта (вместо нуля/обёрнутого хвоста)
    else:
        out[k:] = out[k - 1]             # повтор последнего реального отсчёта
    return out.astype(x.dtype, copy=False)
