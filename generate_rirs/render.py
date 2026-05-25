"""
render.py
~~~~~~~~

Модуль генерации траекторий движения источников, расчёта отражений
(RIR) и рендеринга многоканального аудио с реверберацией.

Основной конвейер:

    traj   = generate_linear_trajectory(duration, scene.room, rng)
    y_mult = render_source(x_mono, fs, traj, scene)

Зависимости: numpy, scipy, torch, gpuRIR.
"""


from __future__ import annotations


from typing import List, Optional, Tuple,Any
from annotation import Annotation

import numpy as np
import scipy.signal as ss
import torch
import gpuRIR

from scene import random_position, SceneAcoustics

# # Оптимизация gpuRIR (вызывать один раз при импорте)
# gpuRIR.activateMixedPrecision(False)   # LUT не работает со смешанной точностью
# gpuRIR.activateLUT(True)

# Пакетная обработка gpuRIR
CHUNK_SIZE: int = 500
# Параметры нормализации пиков при рендеринге
FLOAT_WAV_PEAK_TARGET_MAX: float = 0.98


def normalize_rms(y, sigma=1.):
    k = sigma / (np.std(y) + 1e-12)
    return k * y


def normalize_to_random_rms(y, rms_min=0, rms_max=1.0, rng = None):
    if rng is None:
        rng = np.random.default_rng()        
    sigma = np.sqrt(rng.uniform(rms_min ** 2, rms_max ** 2))
    return normalize_rms(y, sigma), sigma

def clipping(y):
    y_clipped = np.clip(y, -1.0, 1.0)
    return y_clipped

def limit_float_wav_peak(
    y: np.ndarray,
    peak_target: float = FLOAT_WAV_PEAK_TARGET_MAX,
) -> np.ndarray:
    """Уменьшить амплитуду только при риске клиппинга в float WAV.

    Не выравнивает тихие записи — масштабирует вниз только если
    ``max(|y|) > peak_target``.
    """
    y = np.asarray(y, dtype=np.float32)
    p = float(np.max(np.abs(y)))
    if p > peak_target:
        y = y * (peak_target / p)
    return y.astype(np.float32)


# ==========================================================================
#                        ТРАЕКТОРИИ
# ==========================================================================


def generate_linear_trajectory(
    duration: float,
    frame_step:float,
    room_dim: List[float],
    rng: Optional[np.random.Generator] = None,
    start_time: float = 0.0,
) -> List[List[float]]:
    """Сгенерировать линейную 3D-траекторию внутри комнаты.

    Начальная и конечная точки выбираются случайно.
    Траектория равномерно дискретизируется с шагом 0.1 с
    (``duration / 0.1 + 1`` точек — чтобы не терять последний кадр).

    Parameters
    ----------
    duration : float
        Длительность траектории (секунды).
    room_dim : list[float]
        Размеры комнаты [Lx, Ly, Lz].
    rng : numpy.random.Generator | None
    start_time : float
        Абсолютное время начала (по умолчанию 0).

    Returns
    -------
    list[list[float]]
        Точки траектории ``[[x, y, z, t], ...]``.
    """
    if rng is None:
        rng = np.random.default_rng()

    start_pos = random_position(room_dim, rng)
    end_pos = random_position(room_dim, rng)

    num_steps = int(np.ceil(duration / frame_step)) + 1
    times = np.linspace(start_time, start_time + duration, num_steps)

    traj = []
    n = len(times)
    for i, t in enumerate(times):
        alpha = i / max(1, n - 1)
        x = (1 - alpha) * start_pos[0] + alpha * end_pos[0]
        y = (1 - alpha) * start_pos[1] + alpha * end_pos[1]
        z = (1 - alpha) * start_pos[2] + alpha * end_pos[2]
        traj.append([x, y, z, float(t)])

    return traj


def interpolate_trajectory_gpu(
    traj: List[List[float]],
    signal_len: int,
    hop: int,
    fs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Интерполировать траекторию для gpuRIR с заданным шагом *hop* (сэмплы).

    Линейная интерполяция между ключевыми точками траектории.
    Возвращает массив позиций и временных меток для каждой позиции.

    Parameters
    ----------
    traj : list[list[float]]
        Траектория ``[[x, y, z, t], ...]``.
    signal_len : int
        Длина сигнала в сэмплах.
    hop : int
        Шаг интерполяции в сэмплах.
    fs : float | None
        Частота дискретизации. Если ``None``, используется равномерная сетка.

    Returns
    -------
    positions : np.ndarray (n_positions, 3), float32
    timestamps : np.ndarray (n_positions,), float64
    """
    flight_path = np.asarray(traj, dtype=np.float32)
    time_points = flight_path[:, 3].copy()

    n_positions = (signal_len + hop - 1) // hop
    if fs is not None:
        sample_indices = np.arange(0, signal_len, hop)[:n_positions]
        target_times = sample_indices / fs
        target_times = np.clip(target_times, time_points[0], time_points[-1])
    else:
        target_times = np.linspace(time_points[0], time_points[-1], n_positions)

    indices = np.searchsorted(time_points, target_times, side="right") - 1
    indices = np.clip(indices, 0, len(time_points) - 2)

    p_start = flight_path[indices, :3]
    p_end = flight_path[indices + 1, :3]
    t_start = time_points[indices]
    t_end = time_points[indices + 1]

    dt = t_end - t_start
    dt[dt == 0] = 1e-9
    t = (target_times - t_start) / dt

    positions = (1 - t)[:, np.newaxis] * p_start + t[:, np.newaxis] * p_end
    return positions, target_times


# ==========================================================================
#                        РАСЧЁТ ОТРАЖЕНИЙ (RIR)
# ==========================================================================


def calculate_RIRS(
    scene: SceneAcoustics,
    sp_path: np.ndarray,
    fs: int,
    chunk_size:int,
    rt60:float,
    att_diff:float,
    att_max: float,
) -> np.ndarray:
    """Рассчитать Room Impulse Responses с помощью gpuRIR.

    Обрабатывает позиции источника пакетами по :data:`CHUNK_SIZE`
    для экономии GPU-памяти.

    Parameters
    ----------
    scene : SceneAcoustics
        Параметры сцены (комната, бета, микрофоны и т.д.).
    sp_path : np.ndarray (n_points, 3)
        Позиции источника вдоль траектории.
    fs : int
        Частота дискретизации.

    Returns
    -------
    np.ndarray (n_points, n_mics, n_samples), float32
    """
    beta = gpuRIR.beta_SabineEstimation(
        room_sz=scene.room,
        T60=rt60,
        abs_weights=scene.beta,
    )
    Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, rt60)
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt60)
    nb_img = gpuRIR.t2n(Tdiff, scene.room)
    nb_img = np.minimum(nb_img, scene.max_n_rirs)

    n_points = sp_path.shape[0]
    RIRs = None

    for i in range(0, n_points, chunk_size):
        chunk = sp_path[i : i + chunk_size]
        chunk_RIRs = gpuRIR.simulateRIR(
            room_sz=scene.room,
            beta=beta,
            pos_src=chunk,
            pos_rcv=scene.mics,
            nb_img=nb_img,
            Tmax=Tmax,
            fs=fs,
            Tdiff=Tdiff,
            orV_rcv=scene.mic_orientation,
            mic_pattern=scene.mic_pattern,
            c=scene.c,
        )
        if RIRs is None:
            shape = list(chunk_RIRs.shape)
            shape[0] = n_points
            RIRs = np.zeros(shape, dtype=np.float32)
        RIRs[i : i + CHUNK_SIZE, :, :] = chunk_RIRs[:, :, :]

    return RIRs


# ==========================================================================
#                        СИМУЛЯЦИЯ ТРАЕКТОРИИ
# ==========================================================================


def simulateTrajectory(
    source_signal: np.ndarray,
    RIRs: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
) -> np.ndarray:
    """Фильтрация аудиосигнала RIR вдоль траектории движения.

    Замена ``gpuRIR.simulateTrajectory`` с ручным управлением памятью
    (через ``scipy.signal.oaconvolve`` вместо C++ -расширения gpuRIR,
    чтобы избежать проблем с GPU-памятью).

    Parameters
    ----------
    source_signal : np.ndarray (n_samples,)
        Моносигнал движущегося источника.
    RIRs : np.ndarray (n_pts, n_rcv, lenRIR)
        Импульсные характеристики для каждой точки траектории.
    timestamps : np.ndarray | None
        Временные метки каждого RIR (с). По умолчанию — равномерная сетка.
    fs : float | None
        Частота дискретизации (нужна только для кастомных timestamps).

    Returns
    -------
    np.ndarray (len_filtered, n_rcv), float32
        Сигналы, записанные каждым микрофоном.
    """
    nSamples = len(source_signal)
    nPts, nRcv, lenRIR = RIRs.shape

    assert timestamps is None or fs is not None, "fs must be indicated for custom timestamps"
    assert timestamps is None or timestamps[0] == 0, "The first timestamp must be 0"

    if timestamps is None:
        timestamps = np.arange(nPts)
    fs_val = nSamples / nPts

    RIRs = np.asarray(RIRs, dtype=np.float32)
    w_ini = np.append((timestamps * fs_val).astype(int), nSamples)

    len_filtered = nSamples + lenRIR - 1
    filtered_signal = np.zeros((len_filtered, nRcv), dtype=np.float32)

    for n in range(nPts):
        seg = source_signal[w_ini[n] : w_ini[n + 1]].astype(np.float32, copy=False)
        conv = ss.oaconvolve(
            seg[np.newaxis, :], RIRs[n], mode="full", axes=-1
        )  # (nRcv, L_seg + lenRIR - 1)
        len_conv = conv.shape[1]
        filtered_signal[w_ini[n] : w_ini[n] + len_conv, :] += conv.T

    return filtered_signal


# ==========================================================================
#                        РЕНДЕРИНГ ИСТОЧНИКА
# ==========================================================================


def render_source(
    x: np.ndarray,
    fs: int,
    traj: List[List[float]],
    scene: SceneAcoustics,
    chunk_size:int,
    rt60:float,
    att_diff:float,
    att_max: float,
    hop: Optional[int] = None,
    peak_normalize: bool = False,
    peak_target: float = FLOAT_WAV_PEAK_TARGET_MAX,
) -> np.ndarray:
    """Отрендерить моноисточник в многоканальный сигнал с реверберацией.

    Полный пайплайн: интерполяция траектории → расчёт RIR → свёртка.
    Все параметры комнаты и микрофонов берутся из ``scene``.

    Parameters
    ----------
    x : np.ndarray (n_samples,)
        Моно-сигнал источника (float32).
    fs : int
        Частота дискретизации (Гц).
    traj : list[list[float]]
        Траектория источника ``[[x, y, z, t], ...]``.
    scene : SceneAcoustics
        Полное описание акустической сцены.
    hop : int | None
        Шаг интерполяции траектории в сэмплах.
        По умолчанию ``max(1, int(frame_step * fs) // 10)`` с frame_step=0.1.
    peak_normalize : bool
        Если ``True`` — нормировать каждый источник по пику.
    peak_target : float
        Целевой максимум амплитуды (по умолчанию 0.98).

    Returns
    -------
    np.ndarray (n_samples, n_mics), float32
        Многоканальный реверберантный сигнал.
    """
    x_numpy = np.asarray(x, dtype=np.float32).flatten()
    signal_len = len(x_numpy)

    flight_path = np.asarray(traj, dtype=np.float32)
    # Для линейной траектории достаточно начальной и конечной точек
    flight_path = flight_path[[0, -1], ...]

    # Шаг интерполяции по умолчанию
    if hop is None:
        hop = max(1, int(0.1 * fs) // 10)

    sp_path, timestamps = interpolate_trajectory_gpu(
        flight_path, signal_len, hop, fs=fs
    )

    # Расчёт RIR
    RIRs = calculate_RIRS(
        scene=scene, 
        sp_path=sp_path, 
        fs=fs,
        chunk_size=chunk_size,
        rt60=rt60,
        att_diff=att_diff,
        att_max=att_max,
    )
    torch.cuda.empty_cache()

    # Симуляция траектории
    receiver_signals = simulateTrajectory(
        source_signal=x_numpy,
        RIRs=RIRs,
        timestamps=timestamps,
        fs=fs,
    )
    torch.cuda.empty_cache()

    # Корректируем длину до исходного сигнала
    receiver_signals = receiver_signals[:signal_len, :]

    # Нормализация
    if peak_normalize:
        receiver_signals = receiver_signals / (np.max(np.abs(receiver_signals)) + 1e-9)
    else:
        receiver_signals = limit_float_wav_peak(receiver_signals, peak_target=peak_target)

    return receiver_signals

def annotate_trajectory(
    label:str,
    trajectory: List[List[Any]],
    annotations: List[Annotation],
    classes_configs:dict,
    
) -> List[Tuple[int, str]]:
    """
    Сопоставляет моменты времени с событиями из списка аннотаций.
    """
    result = []
    for current_pos in trajectory:
        current_time = current_pos[3]
        for ann in annotations:
            if current_time in ann:  
                lbl = classes_configs[label]['transfer'].get(ann.label, label)
                idx = classes_configs[lbl]['index']
                result.append(current_pos + [idx])
    return result
