"""
extractors.py
~~~~~~~~~~~~~

Функции для извлечения случайных сегментов аудио с аннотациями из трёх каталогов:
  drone_sounds_250kHz, DataSED, Hifitts.

Все три функции имеют единый контракт:

    (signal, annotations) = extract_*(..., duration_s, resolution_ms, sr, rng)

  - signal      : np.ndarray, float32, моно, len(signal) == round(duration_s * sr)
  - annotations : list[Annotation]  — onset/offset в секундах, относительно начала отрезка

Единый формат Annotation позволяет накладывать звуки друг на друга:
  чтобы наложить env на drone достаточно просуммировать сигналы и объединить
  списки аннотаций (drone-аннотации остаются, env-аннотации остаются — все
  уже привязаны к [0, duration_s]).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import json

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as ss
import pandas as pd
from annotation import Annotation


# ==========================================================================
#                        PRIVATE HELPERS
# ==========================================================================


def _quantize_time(t: float, resolution_ms: float) -> float:
    """Округлить время до ближайшего кратного resolution_ms.

    Если resolution_ms <= 0, возвращает t без изменений.
    """
    if resolution_ms <= 0:
        return t
    step = resolution_ms / 1000.0
    return round(t / step) * step


def _read_mono(
    filepath: Union[str, Path],
    target_sr: int,
    lowpass_hz: Optional[float] = None,
) -> np.ndarray:
    """Прочитать WAV → моно → ресэмпл → lowpass.

    Parameters
    ----------
    filepath : str | Path
        Путь к WAV-файлу.
    target_sr : int
        Целевая частота дискретизации.
    lowpass_hz : float | None
        Частота среза ФНЧ Баттерворта 2-го порядка (None → не применять).

    Returns
    -------
    np.ndarray  (1-D, float32)
    """
    y, fs = sf.read(str(filepath), dtype="float32")
    if y.ndim > 1:
        y = y[:, 0]

    if fs != target_sr:
        y = librosa.resample(y=y, orig_sr=fs, target_sr=target_sr)

    if lowpass_hz is not None and lowpass_hz > 0:
        nyq = target_sr / 2.0
        if lowpass_hz < nyq:
            sos = ss.butter(2, lowpass_hz, btype="low", fs=target_sr, output="sos")
            y = ss.sosfiltfilt(sos, y).astype(np.float32)

    return y


def _random_crop(
    y: np.ndarray,
    sr: int,
    duration_s: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Вырезать случайный отрезок длиной duration_s из сигнала y.


    Returns
    -------
    cropped : np.ndarray (float32)
    crop_start_sec : float
        Смещение начала кропа в исходном файле (секунды).
    """
    n_target = int(round(duration_s * sr))
    n_available = len(y)

    # # Если файл короче duration_s — добивает тишиной справа.
    # if n_available <= n_target:
    #     cropped = np.zeros(n_target, dtype=np.float32)
    #     if n_available > 0:
    #         cropped[:n_available] = y[:n_available]
    #     return cropped, 0.0

    if n_available <= n_target:
        return y, 0.0

    max_start = n_available - n_target
    start_sample = int(rng.integers(0, max_start + 1))
    cropped = y[start_sample : start_sample + n_target].copy()
    return cropped, start_sample / sr


def _random_crop_within_event(
    y: np.ndarray,
    sr: int,
    duration_s: float,
    events: List[List[float]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Вырезать отрезок длиной duration_s, гарантируя старт внутри активного события.

    Алгоритм:
      1. Выбрать случайное событие из списка *events*.
      2. Выбрать случайную точку старта внутри этого события.
      3. Если отрезок duration_s не помещается до конца сигнала —
         прижать старт к ``max(0, len - n_target)``.
      4. Если событие целиком короче duration_s и расположено ближе
         к концу файла, чем позволяет кроп — отступаем к началу события
         и обрезаем по фактической длине файла.

    Parameters
    ----------
    y : np.ndarray
        Входной сигнал.
    sr : int
        Частота дискретизации.
    duration_s : float
        Целевая длина отрезка (секунды).
    events : list[[float, float]]
        Интервалы событий ``[[start_sec, end_sec], ...]``
        (в координатах исходного сигнала).
    rng : numpy.random.Generator

    Returns
    -------
    cropped : np.ndarray (float32)
    crop_start_sec : float
    """
    n_target = int(round(duration_s * sr))
    n_available = len(y)

    # # Короткий сигнал — добиваем тишиной (нет событий, которые можно использовать)
    # if n_available <= n_target:
    #     cropped = np.zeros(n_target, dtype=np.float32)
    #     if n_available > 0:
    #         cropped[:n_available] = y[:n_available]
    #     return cropped, 0.0
   
    # Короткий сигнал возвращаем полностью
    if n_available <= n_target:
        return y, 0.0

    if not events:
        # Нет событий — fallback к обычному случайному кропу
        start_sample = int(rng.integers(0, n_available - n_target + 1))
        return y[start_sample : start_sample + n_target].copy(), start_sample / sr

    # Переводим границы событий в сэмплы
    event_ranges_samples: List[Tuple[int, int]] = []
    for ev_start, ev_end in events:
        i0 = max(0, int(round(ev_start * sr)))
        i1 = min(n_available, int(round(ev_end * sr)))
        if i1 > i0:
            event_ranges_samples.append((i0, i1))

    if not event_ranges_samples:
        start_sample = int(rng.integers(0, n_available - n_target + 1))
        return y[start_sample : start_sample + n_target].copy(), start_sample / sr

    # Выбираем случайное событие
    ev_idx = int(rng.integers(0, len(event_ranges_samples)))
    ev_start_sample, ev_end_sample = event_ranges_samples[ev_idx]

    # Максимально допустимый старт кропа, чтобы уместить n_target сэмплов
    max_crop_start = n_available - n_target

    # Идеальный диапазон старта: внутри события [ev_start, ev_end - 1]
    ideal_start_min = ev_start_sample
    ideal_start_max = ev_end_sample - 1

    # Пересекаем с допустимым диапазоном [0, max_crop_start]
    actual_start_min = max(ideal_start_min, 0)
    actual_start_max = min(ideal_start_max, max_crop_start)

    if actual_start_max < actual_start_min:
        # Событие слишком близко к концу файла — прижимаем к началу события
        actual_start_min = max(ideal_start_min, 0)
        actual_start_max = actual_start_min  # фиксированная точка

    start_sample = int(rng.integers(actual_start_min, actual_start_max + 1))
    end_sample = min(start_sample + n_target, n_available)
    cropped = np.zeros(n_target, dtype=np.float32)
    cropped[: end_sample - start_sample] = y[start_sample:end_sample]

    return cropped, start_sample / sr


def _shift_and_clip_annotations(
    events: List[List[float]],
    labels: List[str],
    crop_start_sec: float,
    duration_sec: float,
    resolution_ms: float,
) -> List[Annotation]:
    """Сдвинуть события на локальную шкалу, обрезать по границам, квантизовать.

    Если начало события попало до crop_start_sec → onset = 0.
    Если конец события попал после crop_start_sec + duration_sec → offset = duration_sec.
    """
    annotations: List[Annotation] = []
    for (start, end), label in zip(events, labels):
        # Сдвиг в локальную шкалу
        s = float(start) - float(crop_start_sec)
        e = float(end) - float(crop_start_sec)

        # Обрезка по границам отрезка
        s = max(0.0, s)
        e = min(float(duration_sec), e)

        # Событие целиком вне отрезка — пропускаем
        if e <= s:
            continue

        # Квантизация
        s = _quantize_time(s, resolution_ms)
        e = _quantize_time(e, resolution_ms)

        # После квантизации onset может сравняться с offset — пропускаем
        if s >= e:
            continue

        annotations.append(Annotation(onset=s, offset=e, label=label))

    return annotations


def _has_overlap(intervals: List[List[float]]) -> bool:
    """Проверка наличия пересекающихся интервалов.

    Смежные интервалы [a, b) и [b, c) **не** считаются пересекающимися.
    """
    if len(intervals) < 2:
        return False
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    for i in range(len(sorted_iv) - 1):
        if sorted_iv[i][1] > sorted_iv[i + 1][0]:
            return True
    return False


# ==========================================================================
#                     PUBLIC API — 3 ФУНКЦИИ
# ==========================================================================


def extract_drone_segment(
    files:dict,
    duration_s: float,
    resolution_ms: float,
    sr: int,
    rng: np.random.Generator,
    lowpass_hz: Optional[float] = None,
) -> Tuple[np.ndarray, List[Annotation]]:
    """Извлечь случайный сегмент звука дрона.

    У дрона нет посекундной разметки — генерируется одна аннотация,
    покрывающая весь отрезок с label ``"drone"``.

    Parameters
    ----------
    drone_dir : str | Path
        Путь к каталогу с WAV-файлами дрона (рекурсивный поиск).
    duration_s : float
        Длина извлекаемого отрезка (секунды).
    resolution_ms : float
        Шаг квантизации аннотаций (миллисекунды).
    sr : int
        Целевая частота дискретизации.
    rng : numpy.random.Generator
        Генератор случайных чисел.
    lowpass_hz : float | None
        Частота среза ФНЧ (None → не применять).

    Returns
    -------
    signal : np.ndarray (float32, mono)
        Длина = round(duration_s * sr) сэмплов.
    annotations : list[Annotation]
        Одна запись: ``Annotation(onset=0, offset=duration_s, label="drone")``.
    """
    drone_dir = Path(files['wav_dir'])
    wav_files = sorted(drone_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"WAV-файлы дрона не найдены в {drone_dir}")

    idx = int(rng.integers(0, len(wav_files)))
    filepath = wav_files[idx]

    y = _read_mono(filepath, sr, lowpass_hz=lowpass_hz)
    y, crop_start_sec = _random_crop(y, sr, duration_s, rng)
    y_duration_s = len(y) / sr
    onset = _quantize_time(0.0, resolution_ms)
    offset = _quantize_time(y_duration_s, resolution_ms)
    annotations = [Annotation(onset=onset, offset=offset, label="drone")]

    info = {
        'fname':filepath,
        'start': crop_start_sec
    }
    return y.astype(np.float32), annotations, info


def extract_datased_segment(
    files:dict,
    duration_s: float,
    resolution_ms: float,
    sr: int,
    rng: np.random.Generator,
    lowpass_hz: Optional[float] = None,
    skip_overlapping: bool = True,
) -> Tuple[np.ndarray, List[Annotation]]:
    """Извлечь случайный сегмент из DataSED с посекундной разметкой.

    Гарантирует, что старт кропа попадает внутрь активного события, т.е.
    в любой момент времени отрезка звучит **не более одного** источника
    (файлы с пересекающимися событиями отбрасываются).

    CSV-файл должен содержать столбцы:
    ``sound_name, start_time, end_time, class_name``.

    Parameters
    ----------
    datased_dir : str | Path
        Путь к каталогу с WAV-файлами DataSED.
    annotations_csv : str | Path
        Путь к ``Polyphonic_sound_detection.csv``.
    duration_s : float
        Длина извлекаемого отрезка (секунды).
    resolution_ms : float
        Шаг квантизации аннотаций (миллисекунды).
    sr : int
        Целевая частота дискретизации.
    rng : numpy.random.Generator
        Генератор случайных чисел.
    lowpass_hz : float | None
        Частота среза ФНЧ (None → не применять).
    skip_overlapping : bool
        Пропускать файлы с перекрывающимися интервалами событий.

    Returns
    -------
    signal : np.ndarray (float32, mono)
        Длина = round(duration_s * sr) сэмплов.
    annotations : list[Annotation]
    """
    
    datased_dir = Path(files['wav_dir'])
    annotations_csv = Path(files['annotations_csv'])

    df = pd.read_csv(annotations_csv)
    grouped = df.groupby("sound_name")

    # Собираем записи по файлам, фильтруем пересекающиеся
    file_entries: List[dict] = []
    for name, group_df in grouped:
        events: List[List[float]] = []
        labels: List[str] = []
        for _, row in group_df.iterrows():
            events.append([row["start_time"], row["end_time"]])
            labels.append(row["class_name"])

        if skip_overlapping and _has_overlap(events):
            continue

        # Пропускаем файлы без событий — не из чего выбирать кроп
        if not events:
            continue

        file_entries.append({"file": name, "events": events, "labels": labels})

    if not file_entries:
        raise RuntimeError(
            "Не найдено подходящих файлов DataSED после фильтрации."
        )

    idx = int(rng.integers(0, len(file_entries)))
    entry = file_entries[idx]
    filepath = datased_dir / entry["file"]
    if not filepath.exists():
        raise FileNotFoundError(f"WAV-файл не найден: {filepath}")

    y = _read_mono(filepath, sr, lowpass_hz=lowpass_hz)

    # Кроп внутри случайного активного события (не в тишине между ними)
    y, crop_start_sec = _random_crop_within_event(
        y, sr, duration_s, entry["events"], rng
    )
    y_duration_s = len(y) / sr
    annotations = _shift_and_clip_annotations(
        entry["events"],
        entry["labels"],
        crop_start_sec,
        y_duration_s,
        resolution_ms,
    )

    info = {
        'fname':filepath,
        'start': crop_start_sec
    }
    return y.astype(np.float32), annotations, info


def extract_hifitts_segment(
    files:dict,
    duration_s: float,
    resolution_ms: float,
    sr: int,
    rng: np.random.Generator,
    lowpass_hz: Optional[float] = None,
) -> Tuple[np.ndarray, List[Annotation]]:
    """Извлечь случайный сегмент из Hifitts с разметкой голосовых событий.

    JSON-файл должен иметь структуру::

        {
            "filename.wav": [
                {"start": 0.5, "end": 2.3},
                {"start": 5.0, "end": 7.1}
            ],
            ...
        }

    Всем событиям назначается label ``"Voices"``.

    Parameters
    ----------
    hifitts_dir : str | Path
        Путь к каталогу с WAV-файлами Hifitts.
    annotations_json : str | Path
        Путь к ``hifitts_clean.json``.
    duration_s : float
        Длина извлекаемого отрезка (секунды).
    resolution_ms : float
        Шаг квантизации аннотаций (миллисекунды).
    sr : int
        Целевая частота дискретизации.
    rng : numpy.random.Generator
        Генератор случайных чисел.
    lowpass_hz : float | None
        Частота среза ФНЧ (None → не применять).

    Returns
    -------
    signal : np.ndarray (float32, mono)
        Длина = round(duration_s * sr) сэмплов.
    annotations : list[Annotation]
    """
    hifitts_dir = Path(files['wav_dir'])
    annotations_json = Path(files['annotations_json'])

    with open(annotations_json, "r", encoding="utf-8") as f:
        voice_dict: dict = json.load(f)

    file_entries: List[dict] = []
    for filename, events_list in voice_dict.items():
        events: List[List[float]] = []
        labels: List[str] = []
        for ev in events_list:
            events.append([ev["start"], ev["end"]])
            labels.append("Voices")
        file_entries.append({"file": filename, "events": events, "labels": labels})

    if not file_entries:
        raise RuntimeError("В JSON-файле Hifitts не найдено записей.")

    idx = int(rng.integers(0, len(file_entries)))
    entry = file_entries[idx]
    filepath = hifitts_dir / entry["file"]
    if not filepath.exists():
        raise FileNotFoundError(f"WAV-файл не найден: {filepath}")

    y = _read_mono(filepath, sr, lowpass_hz=lowpass_hz)
    y, crop_start_sec = _random_crop(y, sr, duration_s, rng)

    y_duration_s = len(y) / sr
    annotations = _shift_and_clip_annotations(
        entry["events"],
        entry["labels"],
        crop_start_sec,
        y_duration_s,
        resolution_ms,
    )
    info = {
        'fname':filepath,
        'start': crop_start_sec
    }
    return y.astype(np.float32), annotations, info

