"""
scene.py
~~~~~~~~

Модуль формирования акустической сцены.

Все параметры акустической сцены инкапсулированы в датакласс :class:`SceneAcoustics`,
который генерируется функцией :func:`sample_scene_acoustics`.


    scene  = sample_scene_acoustics(rng, n_mics=5)

Зависимости: numpy, scipy,  gpuRIR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal as ss


# ==========================================================================
#                        МАКРО-ПАРАМЕТРЫ СЦЕНЫ
# ==========================================================================

# Пределы размеров комнаты (метры)
ROOM_MIN: Tuple[float, float, float] = (4.0, 4.0, 4.0)
ROOM_MAX: Tuple[float, float, float] = (14.0, 10.0, 8.0)

# Диапазон температуры воздуха (°C)
AIR_TEMPERATURE_MIN: float = 15.0
AIR_TEMPERATURE_MAX: float = 25.0

# Параметры реверберации
RT60: float = 0.6                  # Время реверберации (с)
ATT_MAX: float = 40.0              # Затухание на конце симуляции (дБ)
ATT_DIFF: float = 15.0             # Затухание при переходе к поздним отражениям (дБ)
MAX_RIRS_LENGTH: int = 8196        # Макс. длина RIR (сэмплов)




# ==========================================================================
#                        DATA STRUCTURES
# ==========================================================================


@dataclass
class SceneAcoustics:
    """Полное описание акустической сцены для рендеринга.

    Содержит все параметры, необходимые для расчёта RIR и рендеринга
    многоканального аудио: размеры комнаты, коэффициенты отражения,
    скорость звука, позиции микрофонов.

    Attributes
    ----------
    room : list[float]
        Размеры комнаты [Lx, Ly, Lz] в метрах.
    beta : np.ndarray
        Коэффициенты отражения 6 стен (float32, shape (6,)).
    c : float
        Скорость звука в воздухе (м/с).
    max_n_rirs : np.ndarray
        Максимальное число изображений-источников по трём осям (int, shape (3,)).
    mics : np.ndarray
        Абсолютные позиции микрофонов (float32, shape (n_mics, 3)).
    mic_orientation : np.ndarray
        Ориентация каждого микрофона — единичный вектор (float32, shape (n_mics, 3)).
    mic_pattern : str
        Диаграмма направленности микрофона (по умолчанию ``"omni"``).
    n_mics : int
        Количество микрофонов.
    """

    room: List[float]
    beta: np.ndarray
    c: float
    max_n_rirs: np.ndarray
    mic_center:np.ndarray
    mics: np.ndarray
    mic_orientation: np.ndarray = field(default=None)  # type: ignore[assignment]
    mic_pattern: str = "omni"
    n_mics: int = 5

    def __post_init__(self):
        # Инициализация mic_orientation по умолчанию, если не передан
        if self.mic_orientation is None:
            self.mic_orientation = np.tile(
                np.array([[1, 0, 0]], dtype=np.float32), (self.n_mics, 1)
            )


def sound_speed(t: float) -> float:
    """Скорость звука в воздухе при температуре *t* (°C).

    Формула: ``331.3 * sqrt(1 + t / 273.15)``.
    """
    return 331.3 * np.sqrt(1 + t / 273.15)

# ==========================================================================
#                        ГЕНЕРАЦИЯ СЦЕНЫ
# ==========================================================================


def random_room_dim(
    rng: Optional[np.random.Generator] = None,
    room_min: Tuple[float, float, float] = ROOM_MIN,
    room_max: Tuple[float, float, float] = ROOM_MAX,
) -> List[float]:
    """Случайные размеры комнаты в заданных пределах.

    Parameters
    ----------
    rng : numpy.random.Generator | None
        Генератор. Если ``None``, создаётся с seed по умолчанию.
    room_min, room_max : tuple[float, float, float]
        Минимальные и максимальные размеры по осям x, y, z.

    Returns
    -------
    list[float]
        [Lx, Ly, Lz].
    """
    if rng is None:
        rng = np.random.default_rng()
    return [
        float(rng.uniform(room_min[0], room_max[0])),
        float(rng.uniform(room_min[1], room_max[1])),
        float(rng.uniform(room_min[2], room_max[2])),
    ]


def random_position(
    room_dim: List[float],
    rng: Optional[np.random.Generator] = None,
    margin: float = 0.5,
) -> List[float]:
    """Случайная позиция внутри комнаты с отступом *margin* от стен.

    Parameters
    ----------
    room_dim : list[float]
        Размеры комнаты [Lx, Ly, Lz].
    rng : numpy.random.Generator | None
    margin : float
        Минимальное расстояние до каждой стены (метры).

    Returns
    -------
    list[float]
        [x, y, z].
    """
    if rng is None:
        rng = np.random.default_rng()
    return [
        float(rng.uniform(margin, room_dim[0] - margin)),
        float(rng.uniform(margin, room_dim[1] - margin)),
        float(rng.uniform(margin, room_dim[2] - margin)),
    ]


def sample_scene_acoustics(
    cfg, 
    rng: Optional[np.random.Generator] = None,
) -> SceneAcoustics:
    """Сгенерировать случайную акустическую сцену.

    Создаёт полный набор параметров: случайную комнату, коэффициенты
    отражения, температуру (→ скорость звука), позиции микрофонов.

    Parameters
    ----------
    cfg : Dict
        Конфигурация    
    rng : numpy.random.Generator | None
        Генератор случайных чисел.

    Returns
    -------
    SceneAcoustics
        Готовый объект сцены для передачи в :func:`render_source`.

    Raises
    ------
    ValueError
        Если ``n_mics`` не найдена в :data:`MIC_GEOMETRIES`.
    """
    if rng is None:
        rng = np.random.default_rng()

    room = random_room_dim(rng)
    reflectivity = float(rng.uniform(
        low=0.5, 
        high=0.8
    ))
    beta = np.full(6, reflectivity, dtype=np.float32)

    air_temperature = float(rng.uniform(
        low=cfg.get('air_temperature_min', AIR_TEMPERATURE_MIN),
        high=cfg.get('air_temperature_max', AIR_TEMPERATURE_MAX),
    ))
    c = sound_speed(air_temperature)

    max_n_rirs = np.full((3,), cfg.get('max_rirs_length', MAX_RIRS_LENGTH))


    mics = np.asarray(cfg['mics'], dtype=np.float32)
    n_mics= mics.shape[0]
    mic_center = np.asarray(random_position(room, rng), dtype=np.float32)
    mics_pos = (mics + mic_center).astype(np.float32)
    mics_orientation = np.tile(
        np.array([[1, 0, 0]], dtype=np.float32), (n_mics, 1)
    )

    return SceneAcoustics(
        room=room,
        beta=beta,
        c=c,
        max_n_rirs=max_n_rirs,
        mic_center = mic_center,
        mics=mics_pos,
        mic_orientation=mics_orientation,
        mic_pattern="omni",
        n_mics=n_mics,
    )
