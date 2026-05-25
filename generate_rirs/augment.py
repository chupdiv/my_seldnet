"""
augment.py
~~~~~~~~~~

Функция аугментации аудио, работающая **исключительно с numpy-массивами**.
Все параметры берутся из секции ``augmentation`` конфигурационного словаря,
который возвращает :func:`config.build_config`.

Пример использования::

    from config import build_config
    from augment import augment_audio, build_augment_pipeline

    cfg = build_config()

    # Вариант 1: однократный вызов (pipeline пересобирается каждый раз)
    y_aug = augment_audio(y, fs=44100, aug_cfg=cfg["augmentation"])

    # Вариант 2: собрать pipeline один раз и переиспользовать
    pipeline = build_augment_pipeline(cfg["augmentation"], fs=44100)
    y_aug = pipeline(samples=y, sample_rate=44100)

    # Вариант 3: воспроизводимость через внешний RNG
    rng = np.random.default_rng(42)
    y_aug = augment_audio(y, fs=44100, aug_cfg=cfg["augmentation"], rng=rng)
    # Второй вызов с тем же RNG даёт тот же результат
    rng2 = np.random.default_rng(42)
    y_aug2 = augment_audio(y, fs=44100, aug_cfg=cfg["augmentation"], rng=rng2)
    assert np.array_equal(y_aug, y_aug2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from audiomentations import (
    AddGaussianNoise,
    AddBackgroundNoise,
    BandPassFilter,
    ClippingDistortion,
    Compose,
    Gain,
    PitchShift,
    PolarityInversion,
    Shift,
    TimeStretch,
)

logger = logging.getLogger(__name__)


# ==========================================================================
#  СБОРКА ПАЙПЛАЙНА
# ==========================================================================


def _resolve_seed(
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[int]:
    """Привести ``seed`` / ``rng`` к целочисленному seed для audiomentations.

    Приоритет: ``rng`` > ``seed`` > ``None``.

    Parameters
    ----------
    seed : int | None
        Прямое целочисленное зерно.
    rng : numpy.random.Generator | None
        Внешний генератор; из него извлекается ``int32``.

    Returns
    -------
    int | None
    """
    if rng is not None:
        return int(rng.integers(0, 2**31, dtype=np.int32))
    return seed


def build_augment_pipeline(
    aug_cfg: Dict[str, Any],
    fs: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Собрать ``audiomentations.Compose`` пайплайн из конфигурации.

    Parameters
    ----------
    aug_cfg : dict
        Секция ``augmentation`` из конфигурационного словаря.
    fs : int
        Частота дискретизации (передаётся в каждый трансформ).
    seed : int | None
        Прямое целочисленное зерно для ``audiomentations.Compose``.
        Игнорируется, если передан ``rng``.
    rng : numpy.random.Generator | None
        Внешний генератор. Если передан, из него извлекается зерно,
        что позволяет контролировать воспроизводимость из вызывающего кода.

    Returns
    -------
    audiomentations.Compose
        Готовый пайплайн аугментаций. Если ``aug_cfg["enabled"]`` равно
        ``False``, возвращается no-op пайплайн (пустой ``Compose(p=0.0)``).
    """
    if not aug_cfg.get("enabled", True):
        logger.debug("Аугментация отключена в конфигурации")
        return Compose([], p=0.0)

    transforms = _build_transforms(aug_cfg, fs)
    #TODO ВНИМАНИЕ!!! модуль audiomentation не поддерживает передачу seed генератора случайных xbctk.
    # aug_seed = _resolve_seed(seed=seed, rng=rng)
    return Compose(transforms, p=aug_cfg['prob'])


def _build_transforms(
    aug_cfg: Dict[str, Any],
    fs: int,
) -> list:
    """Построить список трансформов из конфигурации."""
    transforms: list = []

    # --- Gaussian noise ---
    gn = aug_cfg.get("gaussian_noise")
    if gn:
        transforms.append(
            AddGaussianNoise(
                min_amplitude=gn["min_amplitude"],
                max_amplitude=gn["max_amplitude"],
                p=gn["p"],
            )
        )

    # --- Time stretch ---
    ts = aug_cfg.get("time_stretch")
    if ts:
        transforms.append(
            TimeStretch(
                min_rate=ts["min_rate"],
                max_rate=ts["max_rate"],
                p=ts["p"],
            )
        )

    # --- Pitch shift ---
    ps = aug_cfg.get("pitch_shift")
    if ps:
        transforms.append(
            PitchShift(
                min_semitones=ps["min_semitones"],
                max_semitones=ps["max_semitones"],
                p=ps["p"],
            )
        )

    # --- Shift ---
    sh = aug_cfg.get("shift")
    if sh:
        transforms.append(
            Shift(
                min_shift=sh["min_shift"],
                max_shift=sh["max_shift"],
                p=sh["p"],
            )
        )

    # --- Gain ---
    ga = aug_cfg.get("gain")
    if ga:
        transforms.append(
            Gain(
                min_gain_db=ga["min_gain_db"],
                max_gain_db=ga["max_gain_db"],
                p=ga["p"],
            )
        )

    # --- Clipping distortion ---
    cd = aug_cfg.get("clipping_distortion")
    if cd:
        transforms.append(
            ClippingDistortion(
                min_percentile_threshold=cd["min_percentile_threshold"],
                max_percentile_threshold=cd["max_percentile_threshold"],
                p=cd["p"],
            )
        )

    # --- Band-pass filter ---
    bp = aug_cfg.get("band_pass_filter")
    if bp:
        transforms.append(
            BandPassFilter(
                min_center_freq=bp["min_center_freq"],
                max_center_freq=bp["max_center_freq"],
                p=bp["p"],
            )
        )

    # --- Polarity inversion ---
    pi = aug_cfg.get("polarity_inversion")
    if pi:
        transforms.append(
            PolarityInversion(p=pi["p"])
        )

    # --- Background noise (только если включён и путь валиден) ---
    bn = aug_cfg.get("background_noise")
    if bn and bn.get("enabled") and bn.get("noise_dir"):
        noise_path = Path(bn["noise_dir"])
        if noise_path.is_dir():
            transforms.append(
                AddBackgroundNoise(
                    sounds_path=str(noise_path),
                    min_snr_db=bn["min_snr_db"],
                    max_snr_db=bn["max_snr_db"],
                    p=bn["p"],
                )
            )
        else:
            logger.warning(
                "Background noise: папка не найдена (%s), трансформ пропущен",
                noise_path,
            )

    return transforms


# ==========================================================================
#  ГЛАВНАЯ ФУНКЦИЯ
# ==========================================================================


def augment_audio(
    y: np.ndarray,
    fs: int,
    aug_cfg: Dict[str, Any],
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Применить аугментацию к numpy-массиву аудио.

    Parameters
    ----------
    y : np.ndarray
        1-D массив сэмплов (моно). Форма ``(n_samples,)``.
    fs : int
        Частота дискретизации.
    aug_cfg : dict
        Секция ``augmentation`` из ``config.build_config()``.
    seed : int | None
        Прямое целочисленное зерно. Игнорируется, если передан ``rng``.
    rng : numpy.random.Generator | None
        Внешний генератор для воспроизводимости. Из него извлекается
        зерно для ``audiomentations.Compose``, так что два вызова
        с одинаковым состоянием ``rng`` дают идентичный результат.

    Returns
    -------
    np.ndarray
        Аугментированный массив той же длины (TimeStretch может немного
        изменить её; при ``clip_after=True`` сигнал нормализуется по пику).

    Examples
    --------
    Воспроизводимость через RNG::

        rng_a = np.random.default_rng(42)
        y1 = augment_audio(y, fs, aug_cfg, rng=rng_a)

        rng_b = np.random.default_rng(42)
        y2 = augment_audio(y, fs, aug_cfg, rng=rng_b)
        assert np.array_equal(y1, y2)

    Notes
    -----
    Функция создаёт пайплайн при каждом вызове. Для повторного применения
    к множеству файлов эффективнее собрать пайплайн один раз через
    :func:`build_augment_pipeline` и вызывать его напрямую.
    """
    if not aug_cfg.get("enabled", True):
        return y.copy()

    pipeline = build_augment_pipeline(aug_cfg, fs=fs, seed=seed, rng=rng)

    y_aug = pipeline(samples=y, sample_rate=fs)

    # Защита от клиппинга после аугментаций
    if aug_cfg.get("clip_after", True):
        max_val = np.max(np.abs(y_aug))
        if max_val > 1.0:
            y_aug = y_aug / max_val

    return y_aug
