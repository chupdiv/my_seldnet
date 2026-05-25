"""
config.py
~~~~~~~~~

Выделенные из generate_gpuRIR_3c.py глобальные константы, пути и
обработка параметров командной строки.

Значения по умолчанию загружаются из ``config.yaml`` (в той же директории)
и могут быть переопределены через CLI-аргументы.

Единственная точка входа — функция :func:`build_config`,
которая парсит CLI-аргументы (или принимает значения из YAML)
и возвращает словарь со **всеми** параметрами генерации.

Пример использования::

    from config import build_config

    cfg = build_config()                    # значения из config.yaml
    cfg = build_config(["--duration", "30"])  # CLI-стиль (переопределяет YAML)

    print(cfg["fs"])            # 44100
    print(cfg["room_min"])      # (4.0, 4.0, 4.0)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import extractors

# ==========================================================================
#  ФАЙЛ КОНФИГУРАЦИИ
# ==========================================================================

_CONF_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_yaml_defaults(conf_path: Path = _CONF_PATH) -> Dict[str, Any]:
    """Загрузить словарь значений по умолчанию из YAML-файла."""
    if not conf_path.exists():
        raise FileNotFoundError(
            f"Конфигурационный файл не найден: {conf_path}\n"
            f"Создайте его на основе шаблона или укажите путь через --conf."
        )
    with open(conf_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================================================
#  ARGPARSE
# ==========================================================================


def _make_parser(
    defaults: Dict[str, Any],
) -> argparse.ArgumentParser:
    """Создать парсер CLI-аргументов; дефолты берутся из *defaults*."""
    gen = defaults["generation"]
    sig = defaults["signal"]
    room = defaults["room"]
    rev = defaults["reverb"]
    mix = defaults["mixing"]
    utils = defaults["utils"]
    paths = defaults["paths"]

    p = argparse.ArgumentParser(
        description="Generate synthetic multichannel reverb dataset."
    )

    # ---- Конфигурационный файл ----
    p.add_argument(
        "--conf",
        type=str,
        default=None,
        help="Путь к YAML-файлу конфигурации (переопределяет config.yaml по умолчанию).",
    )

    # ---- Базовые параметры генерации ----
    p.add_argument("--fs", type=int, default=gen["fs"],
                   help=f"Target sampling rate. Default: {gen['fs']}")
    p.add_argument("--n-mics", type=int, default=gen["n_mics"],
                   help=f"Number of microphones. Default: {gen['n_mics']}")
    p.add_argument("--seed", type=int, default=gen["seed"],
                   help=f"Random seed. Default: {gen['seed']}")
    p.add_argument("--out-prefix", type=str, default=gen["out_prefix"],
                   help=f"Prefix for output filenames. Default: {gen['out_prefix']}")
    p.add_argument("--duration", type=int, default=gen["duration_sec"],
                   help=f"Duration of each audio fragment (seconds). Default: {gen['duration_sec']}")
    p.add_argument("--count", type=int, default=gen["files_count"],
                   help=f"Number of target files to generate. Default: {gen['files_count']}")

    # ---- Пути ----
    p.add_argument("--datasets-path", type=str, default=None,
                   help="Root path to datasets directory.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output dataset directory. Default: auto from fs and n-mics")

    # ---- Фильтрация / разметка ----
    p.add_argument("--lowpass-hz", type=float, default=sig["lowpass_hz"],
                   help=f"Butterworth LPF cutoff (Hz); 0 disables. Default: {sig['lowpass_hz']}")
    p.add_argument("--frame-step", type=float, default=sig["frame_step"],
                   help=f"Annotation time grid in seconds. Default: {sig['frame_step']}")

    # ---- Микширование ----
    p.add_argument("--train-fraction", type=float, default=mix["train_fraction"],
                   help=f"Fraction of files for train split. Default: {mix['train_fraction']}")
    # p.add_argument("--snr-min", type=float, default=sig["snr_min_db"],
    #                help=f"Min SNR (dB). Default: {sig['snr_min_db']}")
    # p.add_argument("--snr-max", type=float, default=sig["snr_max_db"],
    #                help=f"Max SNR (dB). Default: {sig['snr_max_db']}")
    p.add_argument("--overlap-touch-counts", action="store_true",
                   default=mix["overlap_touch_counts"],
                   help="Treat adjacent intervals as overlap when filtering.")
    p.add_argument("--rms-normalize", action="store_true",
                   default=sig["rms_normalize"],
                   help="Target RMS normalization.")
    p.add_argument("--rms-max", type=float, default=sig["rms_max"],
                   help=f"Max RMS for target WAV. Default: {sig['rms_max']}")
    p.add_argument("--rms-min", type=float, default=sig["rms_min"],
                   help=f"Min RMS for target WAV. Default: {sig['rms_min']}")

    # ---- Классы ----
    # p.add_argument("--voice-prob", type=float, default=cls["voice_prob"],
    #                help=f"Probability of choosing a voice file for env. Default: {cls['voice_prob']}")

    # ---- Комната ----
    p.add_argument("--room-min", type=float, nargs=3, default=None,
                   help="Room min dimensions [Lx Ly Lz].")
    p.add_argument("--room-max", type=float, nargs=3, default=None,
                   help="Room max dimensions [Lx Ly Lz].")
    p.add_argument("--room-delta", type=float, default=room["delta"],
                   help=f"Min wall distance (m). Default: {room['delta']}")
    p.add_argument("--air-temp-min", type=float, default=room["air_temperature_min"],
                   help=f"Min air temperature (deg C). Default: {room['air_temperature_min']}")
    p.add_argument("--air-temp-max", type=float, default=room["air_temperature_max"],
                   help=f"Max air temperature (deg C). Default: {room['air_temperature_max']}")

    # ---- Реверберация ----
    p.add_argument("--rt60", type=float, default=rev["rt60"],
                   help=f"Reverberation time (s). Default: {rev['rt60']}")
    p.add_argument("--att-max", type=float, default=rev["att_max"],
                   help=f"Attenuation at end of simulation (dB). Default: {rev['att_max']}")
    p.add_argument("--att-diff", type=float, default=rev["att_diff"],
                   help=f"Diffuse attenuation (dB). Default: {rev['att_diff']}")
    p.add_argument("--max-rirs-length", type=int, default=rev["max_rirs_length"],
                   help=f"Max RIR length (samples). Default: {rev['max_rirs_length']}")
    p.add_argument("--chunk-size", type=int, default=rev["chunk_size"],
                   help=f"gpuRIR batch size. Default: {rev['chunk_size']}")

    # ---- Утилиты ----
    p.add_argument("--run-sanity-check", action="store_true",
                   default=utils["run_sanity_check"],
                   help="Run dataset sanity checks after generation.")

    return p


# ==========================================================================
#  ВСПOMOGATEЛЬНЫЕ
# ==========================================================================


def _build_mics(
    mics_cfg: Dict[str, Any], n_mics: int,
) -> np.ndarray:
    """Построить numpy-массив микрофонной решётки по конфигурации."""
    key = f"mics_{n_mics}"
    if key not in mics_cfg:
        raise ValueError(
            f"В конфигурации отсутствует геометрия для {n_mics} микрофонов "
            f"(ожидается ключ '{key}' в секции microphones)."
        )
    return np.array(mics_cfg[key], dtype=np.float32)


def _build_classes_configs(
    datasets_path: Path, paths_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List]:
    """Собрать словарь классов (пути + экстракторы) из YAML-путей.

    Ссылки на функции-экстракторы жестко зашиты здесь, т.к. YAML
    не может хранить коллбэки.
    """
    drones = {
        "index": 0,
        "wav_dir": datasets_path / paths_cfg["drone_wav"],
        "extractor": extractors.extract_drone_segment,
        "transfer": {},
    }
    envs = {
        "index": 1,
        "wav_dir": datasets_path / paths_cfg["env_wav"],
        "annotations_csv": datasets_path / paths_cfg["env_csv"],
        "extractor": extractors.extract_datased_segment,
        "transfer": {"Voices": "voice"},
    }
    voices = {
        "index": 2,
        "wav_dir": datasets_path / paths_cfg["voice_wav"],
        "annotations_json": datasets_path / paths_cfg["voice_json"],
        "extractor": extractors.extract_hifitts_segment,
        "transfer": {},
    }
    classes_configs = {"drone": drones, "env": envs, "voice": voices}
    classes_plan = [([0], 0.5), ([1, 2], 0.5)]
    return classes_configs, classes_plan


# ==========================================================================
#                     ГЛАВНАЯ ФУНКЦИЯ
# ==========================================================================


def build_config(
    argv: Optional[Sequence[str]] = None,
    conf_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Построить полный словарь конфигурации.

    1. Загружает значения по умолчанию из ``config.yaml``.
    2. Парсит CLI-аргументы и переопределяет соответствующие значения.
    3. Вычисляет производные параметры.
    4. Возвращает единый словарь.
    

    Parameters
    ----------
    argv : list[str] | None
        Аргументы командной строки (как ``sys.argv[1:]``).
        Если ``None`` — берётся ``sys.argv[1:]``.
    conf_path : Path | str | None
        Путь к YAML-файлу конфигурации. Если ``None`` — используется
        ``config.yaml`` рядом с данным модулем.

    Returns
    -------
    dict
        Полный набор параметров генерации.

    Raises
    ------
    ValueError
        При некорректных значениях параметров.
    FileNotFoundError
        Если YAML-файл не найден.
    """
    # ---- 1. Загрузка YAML ----
    yaml_path = Path(conf_path) if conf_path else _CONF_PATH
    defaults = _load_yaml_defaults(yaml_path)

    # ---- 2. Парсинг CLI (дефолты из YAML) ----
    parser = _make_parser(defaults)
    args = parser.parse_args(argv)

    # Если передан --conf, перезагрузить defaults оттуда и перепарсить
    if args.conf and args.conf != str(yaml_path):
        defaults = _load_yaml_defaults(Path(args.conf))
        parser = _make_parser(defaults)
        args = parser.parse_args(argv)

    gen = defaults["generation"]
    sig = defaults["signal"]
    room = defaults["room"]
    mix = defaults["mixing"]
    utils = defaults["utils"]
    paths = defaults["paths"]

    # ---- 3. Валидация ----
    if args.frame_step <= 0:
        raise ValueError(f"--frame-step должен быть > 0, получено {args.frame_step}")
    # if args.snr_min > args.snr_max:
    #     raise ValueError(f"--snr-min ({args.snr_min}) должен быть <= --snr-max ({args.snr_max})")
    if args.lowpass_hz is not None and args.lowpass_hz < 0:
        raise ValueError(f"--lowpass-hz должен быть >= 0, получено {args.lowpass_hz}")
    if args.train_fraction < 0 or args.train_fraction > 1:
        raise ValueError(f"--train-fraction должен быть в [0, 1], получено {args.train_fraction}")
    # if args.voice_prob < 0 or args.voice_prob > 1:
    #     raise ValueError(f"--voice-prob должен быть в [0, 1], получено {args.voice_prob}")

    # ---- 4. Микрофоны ----
    mics = _build_mics(defaults["microphones"], args.n_mics)

    # ---- 5. Пути ----
    datasets_path = (
        Path(args.datasets_path) if args.datasets_path
        else Path(paths["datasets"])
    )

    classes_configs, classes_plan = _build_classes_configs(
        datasets_path=datasets_path, 
        paths_cfg=paths,
    )

    # ---- 6. Выходные пути ----
    if args.output_dir is None:
        output_dir = datasets_path / f"reverb_fs{args.fs}Hz_{args.n_mics}mics"
    else:
        output_dir = Path(args.output_dir)

    audio_dir = output_dir / "mic_dev"
    meta_dir = output_dir / "metadata_dev"

    # ---- 7. Производные ----
    hop = int(args.frame_step * args.fs)
    lowpass = None if (args.lowpass_hz is None or args.lowpass_hz == 0) else float(args.lowpass_hz)

    room_min = tuple(args.room_min) if args.room_min else tuple(room["min"])
    room_max = tuple(args.room_max) if args.room_max else tuple(room["max"])

    # ---- 8. Сборка словаря ----
    cfg: Dict[str, Any] = {
        # Параметры генерации
        "fs": args.fs,
        "mics": mics,
        "n_mics": args.n_mics,
        "seed": args.seed,
        "out_prefix": args.out_prefix,
        "duration_sec": int(args.duration),
        "files_count": int(args.count),

        # Параметры сигнала
        "frame_step": args.frame_step,
        "hop": hop,
        "lowpass": lowpass,
        "rms_normalize": args.rms_normalize,
        "rms_min": args.rms_min,
        "rms_max": args.rms_max,

        # Классы
        "classes_plan": classes_plan,
        "classes_configs": classes_configs,

        # Микширование
        "train_fraction": args.train_fraction,
        "overlap_touch_counts": args.overlap_touch_counts,

        # Параметры комнаты
        "room_min": room_min,
        "room_max": room_max,
        "room_delta": args.room_delta,
        "air_temperature_min": args.air_temp_min,
        "air_temperature_max": args.air_temp_max,

        # Реверберация
        "rt60": args.rt60,
        "att_max": args.att_max,
        "att_diff": args.att_diff,
        "max_rirs_length": args.max_rirs_length,
        "chunk_size": args.chunk_size,

        # Пути
        "datasets_path": datasets_path,
        "output_dir": output_dir,
        "audio_dir": audio_dir,
        "meta_dir": meta_dir,

        # Аугментация
        "augmentation": defaults["augmentation"],

        # Утилиты
        "run_sanity_check": args.run_sanity_check,
    }

    return cfg


# ==========================================================================
#                          УТИЛИТЫ
# ==========================================================================
def _to_str(value: Any) -> Any:
    """Рекурсивно преобразовать Path-объекты в строки для печати."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_str(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_str(v) for v in value)
    return value


def _print_nested(d: Dict[str, Any], indent: str = "") -> None:
    """Рекурсивно вывести вложенный словарь по строкам с отступами."""
    for key, val in d.items():
        if callable(val):
            continue
        val_str = _to_str(val)
        if isinstance(val, dict) and val_str:
            print(f"{indent}{key}:")
            _print_nested(val_str, indent + "  ")
        else:
            print(f"{indent}{key:28s} = {val_str}")


def print_config(cfg: Dict[str, Any]) -> None:
    """Вывести конфигурацию в читаемом виде."""
    print("=== CONFIG ===")
    groups = {
        "Generation": [
            "seed", 
            "fs", 
            "n_mics", 
            "mics", 
            "duration_sec", 
            "files_count", 
            "frame_step", 
            "hop", 
            "out_prefix",
        ],
        "Input paths": [
            "datasets_path", 
        ],
        "Output paths": [
            "output_dir", 
            "audio_dir", 
            "meta_dir"
        ],
        "Signal": [
            "lowpass",
            "rms_normalize",
            "rms_min",
            "rms_max", 
        ],
        "Mixing": [
            "classes_plan", 
            "train_fraction", 
            "overlap_touch_counts"
        ],
        "Classes": [
            "classes_configs", 
        ],
        "Room": [
            "room_min", 
            "room_max", 
            "room_delta",
            "air_temperature_min", 
            "air_temperature_max"
        ],
        "Reverb": [
            "rt60", 
            "att_max", 
            "att_diff", 
            "max_rirs_length", 
            "chunk_size"
        ],
        "Augmentation": ["augmentation"],
        "Utils": ["run_sanity_check"],
    }
    for group_name, keys in groups.items():
        print(f"\n  {group_name}:")
        for k in keys:
            if k not in cfg:
                continue
            if k in ("classes_configs", "augmentation"):
                # Иерархический вывод: вложенный словарь по строкам
                if isinstance(cfg[k], dict):
                    _print_nested(cfg[k], indent="    ")
            else:
                print(f"    {k:30s} = {_to_str(cfg[k])}")
    print("===============")

