import random
from pathlib import Path
import argparse

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import math
import json
import scipy.signal as ss


import torch
import gpuRIR
## Используется https://github.com/DavidDiazGuerra/gpuRIR

# Оптимизация производительности gpuRIR
gpuRIR.activateMixedPrecision(False)  # Отключить смешанную точность, с ней не работает таблица поиска
gpuRIR.activateLUT(True)              # Включить таблицу поиска для ускорения

# ---------------------------- BASE CONFIG PARAMS ----------------------------
OUT_PREFIX = "20260405"
FS = 44100
LOWPASS = 5000
MAX_DURATION_SEC = 15 # cекунд
FILES_COUNT = 2000 
N_MICS = 5

# ---------------------------- INPUT PATHS ----------------------------
CURRENT_DIR = Path.cwd()
DATASETS_PATH  = CURRENT_DIR / "../datasets/seld_data"
ENV_DATASET_PATH = DATASETS_PATH / "DataSED/SED_wav"
ENV_CSV_PATH = DATASETS_PATH / "DataSED/SED_ground_truth/Polyphonic_sound_detection.csv"
DRONE_PATH_TRAIN = DATASETS_PATH / "drone_sounds_250kHz"
DRONE_PATH_TEST = DATASETS_PATH / "drone_sounds_test"

VOICE_JSON_PATH = DATASETS_PATH / "Hifitts/hifitts_clean.json"
VOICE_DATASET_PATH = DATASETS_PATH / "Hifitts/hifitts_clean"


# ---------------------------- OUTPUT PATHS ----------------------------
OUTPUT_DIR = DATASETS_PATH / f"reverb_all_inclusive_voice_fs{FS}Hz_{N_MICS}mics"
AUDIO_DIR = OUTPUT_DIR / "mic_dev"
META_DIR = OUTPUT_DIR / "metadata_dev"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------- CONFIG  ----------------------------
# Доля обучающей выборки
TRAIN_FRACTION = 0.85

# Смесь «дрон + окружение»: SNR = 10*log10(P_drone / P_env) по активным интервалам окружения.
# Умеренный диапазон для аугментации без доминирования одного класса.
SNR_MIN_DB = -6.0
SNR_MAX_DB = 6.0
# False: не выравнивать каждый источник по пику (сохраняет относительную энергетику для SNR).
# True: прежнее поведение — нормализация по максимуму в render_source.
PEAK_NORMALIZE_RENDER = False
FLOAT_WAV_PEAK_TARGET_MAX = 0.98
FLOAT_WAV_PEAK_TARGET_MIN = 0.10
# Считать «стык» интервалов не пересечением (смежные [a,b) и [b,c) не отбрасываются).
OVERLAP_TOUCH_COUNTS = False

ROOM_MIN = (4.0, 4.0, 4.0)
ROOM_MAX = (14.0, 10.0, 8.0)
ROOM_DELTA = 0.5  # Минимальное расстояние от стен (в метрах)

# Температура помещения в °C
AIR_TEMPERATURE_MIN = 15.0 
AIR_TEMPERATURE_MAX = 25.0 


# Настройки производительности генератора отражений
MAX_RIRS_LENGTH = 8196
RT60 = 0.6

ATT_MAX = 40.0  # Attenuation at the end of the simulation [dB]
ATT_DIFF = 15.0

CHUNK_SIZE = 500  # Размер пакета для расчёта RIR


MIC_GEOMETRIES = {
    4: np.array(
        [
            [ 0.0,      0.0,   0.08016],
            [ 0.0577,   0.0,   0.0],
            [-0.02885, -0.05, 0.0],
            [-0.02885,  0.05, 0.0],
        ],
        dtype=np.float32,
    ),
    5: np.array(
        [
            [0.0, -0.05, -0.05],
            [0.0, -0.05, 0.05],
            [0.0, 0.05, 0.05],
            [0.0, 0.05, -0.05],
            [0.0707, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
}
MIC_POS_FIXED = MIC_GEOMETRIES[N_MICS]
MIC_ORIENTATION = np.tile(np.array([[1, 0, 0]], dtype=np.float32), (N_MICS, 1))

MIC_PATTERN = "omni"


CLASSES_DATASED_DICT = {"Voices": 2}    # Классы из DataSED, для которых делаем специфический класс
VOICE_PROB = 0.5 # Вероятность того, что выберем файл с голосом при условии, что выбран файл окружения



FRAME_STEP = 0.1  # seconds
HOP = int(FRAME_STEP * FS)  # Шаг интерполяции в сэмплах

RNG = 20260405
random.seed(RNG)
np.random.seed(RNG)

def sound_speed(t):
    return 331.3 * np.sqrt(1 + t / 273.15)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic multichannel reverb dataset.")
    parser.add_argument("--fs", type=int, default=FS, help="Target sampling rate.")
    parser.add_argument("--n-mics", type=int, default=N_MICS, choices=sorted(MIC_GEOMETRIES.keys()), help="Number of microphones.")
    parser.add_argument("--seed", type=int, default=RNG, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dataset directory.")
    parser.add_argument("--run-sanity-check", action="store_true", help="Run dataset sanity checks after generation.")
    parser.add_argument(
        "--snr-min",
        type=float,
        default=SNR_MIN_DB,
        help="Min SNR (dB) for drone vs environment in 'both' (active env intervals). Default: -6.",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=SNR_MAX_DB,
        help="Max SNR (dB) for drone vs environment in 'both'. Default: 6.",
    )
    parser.add_argument(
        "--peak-normalize-render",
        action="store_true",
        help="Legacy: per-source peak normalization inside render_source (worse for SNR mixing).",
    )
    parser.add_argument(
        "--overlap-touch-counts",
        action="store_true",
        help="Treat adjacent event intervals as overlap when filtering good_env_files (default: touch is not overlap).",
    )
    parser.add_argument(
        "--frame-step",
        type=float,
        default=FRAME_STEP,
        metavar="SEC",
        help="Annotation time grid in seconds (metadata frame index × this). Default: %(default)s (100 ms).",
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=LOWPASS,
        metavar="HZ",
        help="Butterworth LPF cutoff (Hz) on written WAV and when reading sources; 0 disables. Default: %(default)s.",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=MAX_DURATION_SEC,
        help="The duration of the generated audio fragment in seconds.  Default: %(default)s (15 sec).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=FILES_COUNT,
        help="The count of target files.  Default: %(default)s (2000).",
    )

    return parser.parse_args()


def configure_from_args(args):
    global FS, HOP, FRAME_STEP, RNG, N_MICS, MIC_POS_FIXED, MIC_ORIENTATION, OUTPUT_DIR, AUDIO_DIR, META_DIR
    global SNR_MIN_DB, SNR_MAX_DB, PEAK_NORMALIZE_RENDER, OVERLAP_TOUCH_COUNTS, LOWPASS, MAX_DURATION_SEC
    global FILES_COUNT

    if args.frame_step <= 0:
        raise ValueError(f"--frame-step must be positive, got {args.frame_step}")
    FRAME_STEP = args.frame_step

    FS = args.fs
    HOP = int(FRAME_STEP * FS)
    RNG = args.seed
    N_MICS = args.n_mics
    MIC_POS_FIXED = MIC_GEOMETRIES[N_MICS]
    MIC_ORIENTATION = np.tile(np.array([[1, 0, 0]], dtype=np.float32), (N_MICS, 1))

    if args.snr_min > args.snr_max:
        raise ValueError(f"--snr-min ({args.snr_min}) must be <= --snr-max ({args.snr_max})")
    SNR_MIN_DB = args.snr_min
    SNR_MAX_DB = args.snr_max
    PEAK_NORMALIZE_RENDER = args.peak_normalize_render
    OVERLAP_TOUCH_COUNTS = args.overlap_touch_counts

    if args.lowpass_hz < 0:
        raise ValueError(f"--lowpass-hz must be >= 0, got {args.lowpass_hz}")
    LOWPASS = None if args.lowpass_hz == 0 else float(args.lowpass_hz)

    if args.output_dir is None:
        OUTPUT_DIR = DATASETS_PATH / f"reverb_all_inclusive_voice_fs{FS}Hz_{N_MICS}mics"
    else:
        OUTPUT_DIR = Path(args.output_dir)
    AUDIO_DIR = OUTPUT_DIR / "mic_dev"
    META_DIR = OUTPUT_DIR / "metadata_dev"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    MAX_DURATION_SEC = int(args.duration)
    FILES_COUNT = int(args.count)

    random.seed(RNG)
    np.random.seed(RNG)


def write_soundfile(fname, y, fs):
    nyq = fs / 2.0
    if LOWPASS is not None and LOWPASS > 0 and LOWPASS < nyq:
        sos = ss.butter(2, LOWPASS, btype='low', fs=fs, output='sos')
        y = ss.sosfiltfilt(sos, y, axis=0)
    y = np.clip(y, -1.0, 1.0)
    sf.write(fname, y, fs, format="WAV", subtype="FLOAT")

def read_soundfile(fname, dtype = 'float32'):
    y, fs = sf.read(fname, dtype=dtype)
    if y.ndim > 1:
        y = y[:, 0]
    nyq = fs / 2.0
    max_duration = int(MAX_DURATION_SEC*fs)
    if max_duration < y.shape[0]:
        max_start = y.shape[0]- max_duration
        start = np.random.randint(0, max_start)
        y = y[start:start+max_duration]

    if fs != FS:
        y = librosa.resample(y=y, orig_sr=fs, target_sr=FS)
        fs = FS
    nyq = FS / 2.0
    if LOWPASS is not None and LOWPASS > 0 and LOWPASS < nyq:
        sos = ss.butter(2, LOWPASS, btype='low', fs = FS, output='sos')
        y = ss.sosfiltfilt(sos, y)
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    # Нормирование к энергетике 1
    y = y / (rms(y) + 1e-12)
    return y, FS


# Выдернул функцию из GPURIR и сделал затычку через signal, так как ругается на память (хотя реально занимает 10%)
# Переписал функцию так, чтобы создавалось как можно меньше массивов в памяти
def simulateTrajectory(source_signal, RIRs, timestamps=None, fs=None):
    ''' Filter an audio signal by the RIRs of a motion trajectory recorded with a microphone array.
    Parameters
    ----------
    source_signal : array_like
    	Signal of the moving source.
    RIRs : 3D ndarray
    	Room Impulse Responses generated with simulateRIR.
    timestamps : array_like, optional
    	Timestamp of each RIR [s]. By default, the RIRs are equispaced through the trajectory.
    fs : float, optional
    	Sampling frequency (in Hertz). It is only needed for custom timestamps.
    Returns
    -------
       2D ndarray
           Matrix with the signals captured by each microphone in each column.
       '''
    nSamples = len(source_signal)
    nPts, nRcv, lenRIR = RIRs.shape

    assert timestamps is None or fs is not None, "fs must be indicated for custom timestamps"
    assert timestamps is None or timestamps[0] == 0, "The first timestamp must be 0"
    if timestamps is None:
        fs = nSamples / nPts
        timestamps = np.arange(nPts)
    RIRs = np.asarray(RIRs, dtype = np.float32)
    w_ini = np.append((timestamps * fs).astype(int), nSamples)

    # Выходной массив инициализируем сразу в float32
    len_filtered = nSamples + lenRIR - 1
    filtered_signal = np.zeros((len_filtered, nRcv), dtype=np.float32)

    for n in range(nPts):
        # Выполняется свертка сегмента seg со всеми RIR текущей позиции n
        # seg[np.newaxis, :] -> (1, L_seg), RIRs[n] -> (nRcv, lenRIR)
        seg = source_signal[w_ini[n]:w_ini[n+1]].astype(np.float32, copy=False) # (L_seg)
        conv = ss.oaconvolve(seg[np.newaxis, :], RIRs[n], mode='full', axes=-1) # (nRcv, L_seg + lenRIR - 1)
        len_conv = conv.shape[1]
        filtered_signal[w_ini[n] : w_ini[n] + len_conv, :] += conv.T

    return filtered_signal  

# ---------------------------- Helpers ----------------------------

def choose_scenario():
    """Randomly choose a mixing scenario.

    Scenarios:
        - 'drone_only'
        - 'env_only'
        - 'both'

    Returns:
        str: Selected scenario name.
    """
    r = random.random()
    if r < 0.25:
        return "drone_only"
    elif r <= 0.7:
        return "env_only"
    else:
        return "both"

def random_room_dim():
    """Sample random room dimensions within predefined bounds.

    Returns:
        list[float]: Random room dimensions [Lx, Ly, Lz].
    """
    return [
        random.uniform(ROOM_MIN[0], ROOM_MAX[0]),
        random.uniform(ROOM_MIN[1], ROOM_MAX[1]),
        random.uniform(ROOM_MIN[2], ROOM_MAX[2]),
    ]

def random_position(room_dim):
    """Sample a random position inside the room with margin to walls.

    Args:
        room_dim (sequence[float]): Room dimensions [Lx, Ly, Lz].

    Returns:
        list[float]: Random position [x, y, z] inside the room.
    """
    return [
        random.uniform(0.5, room_dim[0] - 0.5),
        random.uniform(0.5, room_dim[1] - 0.5),
        random.uniform(0.5, room_dim[2] - 0.5),
    ]

def generate_linear_trajectory(duration, room_dim, start_time=0.0):
    """Generate a linear 3D trajectory segment inside a room.

    Args:
        duration (float): Duration in seconds.
        room_dim (sequence[float]): Room dimensions [Lx, Ly, Lz].
        start_time (float, optional): Start time of trajectory. Defaults to 0.0.

    Returns:
        list[list[float]]: Trajectory as [x, y, z, t].
    """
    start_pos = random_position(room_dim)
    end_pos = random_position(room_dim)
    
    #FIXED потеря последнего кадра аудио
    num_steps = int(np.ceil(duration / FRAME_STEP)) + 1
    times = np.linspace(start_time, start_time + duration, num_steps)    

    traj = []
    num_steps = len(times)
    for i, t in enumerate(times):
        alpha = i / max(1, num_steps - 1)
        x = (1 - alpha) * start_pos[0] + alpha * end_pos[0]
        y = (1 - alpha) * start_pos[1] + alpha * end_pos[1]
        z = (1 - alpha) * start_pos[2] + alpha * end_pos[2]
        traj.append([x, y, z, t])
    return traj


def interpolate_trajectory_gpu(traj, signal_len, hop, fs=None):
    """
    Интерполирует траекторию с заданным шагом hop (в сэмплах) для gpuRIR.
    
    Аргументы:
        traj: траектория [x, y, z, t] (список списков или np.ndarray)
        signal_len: длина сигнала в сэмплах
        hop: шаг интерполяции в сэмплах
        fs: частота дискретизации
    
    Возвращает:
        positions: np.ndarray формы (n_positions, 3)
        timestamps: np.ndarray формы (n_positions,)
    """
    # hop = hop//10
    flight_path = np.asarray(traj, dtype=np.float32)
    
    # Извлекаем временные метки
    time_points = flight_path[:, 3].copy()
    
    # Определяем целевые временные точки
    n_positions = (signal_len + hop - 1) // hop
    if fs is not None:
        sample_indices = np.arange(0, signal_len, hop)[:n_positions]
        target_times = sample_indices / fs
        target_times = np.clip(target_times, time_points[0], time_points[-1])
    else:
        target_times = np.linspace(time_points[0], time_points[-1], n_positions)
    
    # Поиск индексов сегментов для линейной интерполяции
    indices = np.searchsorted(time_points, target_times, side='right') - 1
    indices = np.clip(indices, 0, len(time_points) - 2)
    
    # Координаты начала и конца отрезков
    p_start = flight_path[indices, :3]
    p_end = flight_path[indices + 1, :3]
    t_start = time_points[indices]
    t_end = time_points[indices + 1]
    
    # Коэффициент интерполяции
    dt = t_end - t_start
    dt[dt == 0] = 1e-9
    t = (target_times - t_start) / dt
    
    # Линейная интерполяция позиций
    positions = (1 - t)[:, np.newaxis] * p_start + t[:, np.newaxis] * p_end
    return positions, target_times

def calculate_RIRS(sp_path, room_dims, beta, max_n_rirs, receiver_positions, fs):
    """
    Расчёт Room Impulse Responses с помощью gpuRIR.
    
    Args:
        sp_path: позиции источника (n_points, 3)
        room_dims: размеры комнаты [Lx, Ly, Lz]
        beta: коэффициенты отражения стен
        max_n_rirs: максимальное количество отражений
        receiver_positions: позиции микрофонов (n_mics, 3)
        fs: частота дискретизации
    
    Returns:
        np.ndarray: RIRs (n_points, n_mics, n_samples)
    """
    beta = gpuRIR.beta_SabineEstimation(
        room_sz=room_dims,
        T60=RT60,
        abs_weights=beta
    )
    Tdiff = gpuRIR.att2t_SabineEstimator(ATT_DIFF, RT60)
    Tmax = gpuRIR.att2t_SabineEstimator(ATT_MAX, RT60)
    nb_img = gpuRIR.t2n(Tdiff, room_dims)
    nb_img = np.minimum(nb_img, max_n_rirs)

    n_points = sp_path.shape[0]
    RIRs = None
    air_temperatue = np.random.uniform(low=AIR_TEMPERATURE_MIN, high=AIR_TEMPERATURE_MAX)
    c = sound_speed(air_temperatue)
    for i in range(0, n_points, CHUNK_SIZE):
        chunk = sp_path[i:i+CHUNK_SIZE]
        chunk_RIRs = gpuRIR.simulateRIR(
            room_sz=room_dims,
            beta=beta,
            pos_src=chunk,
            pos_rcv=receiver_positions,
            nb_img=nb_img,
            Tmax=Tmax,
            fs=fs,
            Tdiff=Tdiff,
            orV_rcv=MIC_ORIENTATION,
            mic_pattern=MIC_PATTERN,
            c = c,
        )
        if RIRs is None:
            RIRs_shape = list(chunk_RIRs.shape)
            RIRs_shape[0] = n_points
            RIRs = np.zeros(RIRs_shape, dtype=np.float32)
        RIRs[i:i+CHUNK_SIZE, :, :] = chunk_RIRs[:, :, :]
    
    return RIRs


def render_source(x, fs, traj, room_dim, mic_xyz):
    """Render a single source with reverb using gpuRIR.
    Args:
        x (np.ndarray): Mono source signal.
        fs (int): Sampling rate in Hz.
        traj (array_like): Source trajectory [x, y, z, t].
        room_dim (sequence[float]): Room dimensions [Lx, Ly, Lz].
        mic_xyz (array_like): Microphone positions (M, 3).

    Returns:
        np.ndarray: Multichannel reverberant audio (N, M).
    """
    x_numpy = np.asarray(x, dtype=np.float32).flatten()
    signal_len = len(x_numpy)
    
    # Преобразуем траекторию в numpy массив
    flight_path = np.asarray(traj, dtype=np.float32)
    flight_path = flight_path[[0,-1],...] # Нам нужно только начало и конец

    # Масштабируем траекторию под комнату
    # flight_path = scale_trajectory_to_room(flight_path, room_dim, ROOM_DELTA)
    # Интерполируем траекторию для gpuRIR
    sp_path, timestamps = interpolate_trajectory_gpu(
        flight_path, 
        signal_len, 
        max(1, HOP // 10), 
        fs
    )
    
    # Параметры отражения стен (типичная комната)
    reflectivity = np.random.uniform(0.5, 0.8)
    beta = np.full(6, reflectivity)
    max_n_rirs = np.full((3,), MAX_RIRS_LENGTH)
    
    # Рассчитываем RIR
    RIRs = calculate_RIRS(sp_path, room_dim, beta, max_n_rirs, mic_xyz, fs)
    
    # Очищаем кэш GPU
    torch.cuda.empty_cache()
    
    # Симулируем траекторию источника

    # receiver_signals = gpuRIR.simulateTrajectory(
    receiver_signals = simulateTrajectory(
        source_signal=x_numpy,
        RIRs=RIRs, #RIRs[..., :MAX_RIRS_LENGTH],
        timestamps=timestamps,
        fs=fs,
        )
    
    torch.cuda.empty_cache()
    # Корректируем длину выходного сигнала
    receiver_signals = receiver_signals[:signal_len, :]

    if PEAK_NORMALIZE_RENDER:
        receiver_signals = receiver_signals / (np.max(np.abs(receiver_signals)) + 1e-9)
    else:
        receiver_signals = limit_float_wav_peak(receiver_signals)
    return receiver_signals


def generate_annotations_both(drone_traj, environ_traj, mic_center, events, classes):
    """Generate frame-wise DOA and distance annotations for scenario "both"

    Args:
        drone_traj (array_like): Drone trajectory [x, y, z, t].
        environ_traj (array_like): Environment trajectory [x, y, z, t].
        mic_center (array_like): Microphone array center [x, y, z].
        events (array_like): List of events.
        classes (array_like): List of classes.

    Returns:
        list[str]: Annotation lines 'frame, class, activity, azimuth, elevation, distance'.
    """
    drone_rel_traj = np.asarray(drone_traj, dtype=np.float32)
    drone_rel_traj[:, :3] -= mic_center

    environ_rel_traj = np.asarray(environ_traj, dtype=np.float32)
    environ_rel_traj[:, :3] -= mic_center

    len_traj = len(drone_rel_traj)

    annotations = []
    for frame_idx in range(len_traj):
        cur_time = frame_idx * FRAME_STEP

        (x_drone, y_drone, z_drone, _) = drone_rel_traj[frame_idx]
        (x_environ, y_environ, z_environ, _) = environ_rel_traj[frame_idx]

        azi_drone = int(round(math.degrees(math.atan2(y_drone, x_drone)) % 360))
        ele_drone = int(
            round(
                np.clip(
                    math.degrees(math.atan2(z_drone, math.hypot(x_drone, y_drone))),
                    -90,
                    90,
                )
            )
        )
        dist_drone = int(round(np.sqrt(x_drone ** 2 + y_drone ** 2 + z_drone ** 2) * 100))

        azi_environ = int(round(math.degrees(math.atan2(y_environ, x_environ)) % 360))
        ele_environ = int(
            round(
                np.clip(
                    math.degrees(math.atan2(z_environ, math.hypot(x_environ, y_environ))),
                    -90,
                    90,
                )
            )
        )
        dist_environ = int(round(np.sqrt(x_environ ** 2 + y_environ ** 2 + z_environ ** 2) * 100))

        annotations.append(f"{frame_idx},0,0,{azi_drone},{ele_drone},{dist_drone}")
        # Добавляем environment, если он входит в список событий
        for ev_ind in range(len(events)):
            start = events[ev_ind][0]
            end = events[ev_ind][1]
            if start <= cur_time <= end:
                if classes[ev_ind] not in CLASSES_DATASED_DICT:
                    annotations.append(f"{frame_idx},1,0,{azi_environ},{ele_environ},{dist_environ}")
                else:
                    annotations.append(f"{frame_idx},{CLASSES_DATASED_DICT[classes[ev_ind]]},0,{azi_environ},{ele_environ},{dist_environ}")
        
    return annotations


def generate_annotations_single(traj, mic_center, class_index, events, classes):
    """Generate frame-wise DOA and distance annotations for a single class.

    Args:
        traj (array_like): Source trajectory [x, y, z, t].
        mic_center (array_like): Microphone array center [x, y, z].
        class_index (int): Class label index.
        events (array_like): List of events.
        classes (array_like): List of classes.

    Returns:
        list[str]: Annotation lines 'frame, class, activity, azimuth, elevation, distance'.
    """
    rel_traj = np.asarray(traj, dtype=np.float32)
    rel_traj[:, :3] -= mic_center

    annotations = []
    for frame_idx, (x, y, z, _) in enumerate(rel_traj):
        cur_time = frame_idx * FRAME_STEP

        # Пропускаем момент времени, если он не входит в список событий
        if events is not None:
            if not any(start <= cur_time <= end for start, end in events):
                continue

        azi = int(round(math.degrees(math.atan2(y, x)) % 360))
        ele = int(
            round(
                np.clip(
                    math.degrees(math.atan2(z, math.hypot(x, y))),
                    -90,
                    90,
                )
            )
        )
        dist = int(round(np.sqrt(x ** 2 + y ** 2 + z ** 2) * 100))
        if classes is None:
            annotations.append(f"{frame_idx},{class_index},0,{azi},{ele},{dist}")
        else:
            for ev_ind in range(len(events)):
                start = events[ev_ind][0]
                end = events[ev_ind][1]
                if start <= cur_time <= end:
                    if classes[ev_ind] not in CLASSES_DATASED_DICT:
                        annotations.append(f"{frame_idx},{class_index},0,{azi},{ele},{dist}")
                    else:
                        annotations.append(f"{frame_idx},{CLASSES_DATASED_DICT[classes[ev_ind]]},0,{azi},{ele},{dist}")
            
    return annotations


def prepare_split_dirs(split_name):
    """Prepare output directories for a dataset split.

    Args:
        split_name (str): Split name (e.g., 'dev-train', 'dev-test').

    Returns:
        tuple[Path, Path]: Paths to WAV and CSV directories.
    """
    wav_dir = AUDIO_DIR / split_name
    csv_dir = META_DIR / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    return wav_dir, csv_dir


def has_overlap(intervals):
    # Сортируем по возрастанию начала интервала
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    for i in range(len(sorted_intervals) - 1):
        if OVERLAP_TOUCH_COUNTS:
            if sorted_intervals[i][1] >= sorted_intervals[i + 1][0]:
                return True
        else:
            if sorted_intervals[i][1] > sorted_intervals[i + 1][0]:
                return True

    return False


def cut_audio_by_intervals(x, sr, intervals):
    """
    x: входное аудио
    sr: частота дискретизации
    intervals: список [ [start_sec, end_sec], ... ]
    """

    pieces = []
    for start, end in intervals:
        # переводим секунды в индексы сэмплов
        i0 = int(round(start * sr))
        i1 = int(round(end * sr))
        i0 = max(i0, 0)
        i1 = min(i1, x.shape[0])
        if i1 > i0:
            pieces.append(x[i0:i1])

    if not pieces:
        # ничего не осталось — вернём тишину нулевой длины
        y = x[:0]
    else:
        y = np.concatenate(pieces, axis=0)

    return y


def has_silence(y, sr, top_db: float = 40.0, silence_ratio: float = 0.10,
                frame_length: int = 2048, hop_length: int = 512) -> bool:
    """Проверяет, содержит ли аудио много тишины (> silence_ratio)."""
    try:
        if y.ndim > 1:
            y = librosa.to_mono(y.T)

        rms = np.sqrt(np.mean(y**2))
        db = 20 * np.log10(rms + 1e-12)
        # if db < -45:
        #     return True

        if np.max(np.abs(y)) < 1e-4:
            return True

        duration = y.shape[-1] / sr
        if duration == 0:
            return True

        intervals = librosa.effects.split(
            y,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )

        if len(intervals) == 0:
            return True

        non_silent_dur = np.sum((intervals[:, 1] - intervals[:, 0]) / sr)

        silence_ratio_actual = 1 - (non_silent_dur / duration)

        return silence_ratio_actual > silence_ratio

    except Exception as e:
        print("Silence detection error:", e)
        return True


def rms(x):
    """Compute root mean square value.

    Args:
        x (array_like): Input signal.

    Returns:
        float: RMS value.
    """
    return np.sqrt(np.mean(x ** 2))

def level_scale(y, peak_target):
    p = float(np.max(np.abs(y)))
    y = y * (peak_target / p)
    return y.astype(np.float32)


def limit_float_wav_peak(y, peak_target=None):
    """Уменьшить амплитуду только при риске клиппинга в float WAV (не выравнивает тихие записи)."""
    if peak_target is None:
        peak_target = FLOAT_WAV_PEAK_TARGET_MAX
    y = np.asarray(y, dtype=np.float32)
    p = float(np.max(np.abs(y)))
    if p > peak_target:
        y = y * (peak_target / p)
    return y.astype(np.float32)


def active_interval_mask(n_samples, fs, events):
    """Булева маска сэмплов, попадающих в объединение интервалов событий [start, end] в секундах."""
    mask = np.zeros(n_samples, dtype=bool)
    if not events:
        return mask
    for ev in events:
        if len(ev) < 2:
            continue
        start, end = float(ev[0]), float(ev[1])
        i0 = int(round(start * fs))
        i1 = int(round(end * fs))
        i0 = max(0, i0)
        i1 = min(n_samples, i1)
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def mix_drone_and_environment(y_drone, y_env, fs, events, snr_db):
    """
    Сумма y_drone + g * y_env с целевым SNR (дБ) дрон/окружение.
    Мощность окружения считается по активным интервалам events; мощность дрона — по всей длине клипа.
    Один коэффициент g на все каналы. Защита от перегруза — только при |y| > peak_target.
    """
    eps = 1e-12
    y_drone = np.asarray(y_drone, dtype=np.float32)
    y_env = np.asarray(y_env, dtype=np.float32)
    n = y_env.shape[0]
    mask = active_interval_mask(n, fs, events)
    if mask.any():
        pe = np.mean(np.square(y_env[mask]))
    else:
        pe = np.mean(np.square(y_env))
    pd = np.mean(np.square(y_drone))
    g = np.sqrt(pd / (pe * (10.0 ** (snr_db / 10.0)) + eps))
    mix = y_drone + g * y_env
    return limit_float_wav_peak(mix)


def pick_drone_path(drone_train, drone_test, is_train):
    """Исключает утечку: train fold только из DRONE_PATH_TRAIN, test — из DRONE_PATH_TEST."""
    if is_train:
        pool = drone_train if drone_train else drone_test
    else:
        pool = drone_test if drone_test else drone_train
    if not pool:
        return None
    return random.choice(pool)


def run_sanity_check(expected_fs, expected_n_mics):
    """Validate generated dataset structure and basic metadata quality."""
    print("\n=== SANITY CHECK START ===")
    splits = ["dev-train", "dev-test"]
    total_wavs = 0
    total_csvs = 0
    total_rows = 0
    classes_seen = set()
    errors = []

    for split in splits:
        wav_dir = AUDIO_DIR / split
        csv_dir = META_DIR / split
        wav_files = sorted(wav_dir.glob("*.wav"))
        csv_files = sorted(csv_dir.glob("*.csv"))
        total_wavs += len(wav_files)
        total_csvs += len(csv_files)

        wav_stems = {p.stem for p in wav_files}
        csv_stems = {p.stem for p in csv_files}
        missing_csv = sorted(wav_stems - csv_stems)
        missing_wav = sorted(csv_stems - wav_stems)
        if missing_csv:
            errors.append(f"[{split}] missing csv for wav: {len(missing_csv)}")
        if missing_wav:
            errors.append(f"[{split}] missing wav for csv: {len(missing_wav)}")

        for wav_path in wav_files:
            info = sf.info(str(wav_path))
            if info.samplerate != expected_fs:
                errors.append(f"[{split}] bad fs in {wav_path.name}: {info.samplerate}")
            if info.channels != expected_n_mics:
                errors.append(f"[{split}] bad channels in {wav_path.name}: {info.channels}")

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    total_rows += 1
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) != 6:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} expected 6 columns, got {len(parts)}")
                        continue
                    try:
                        frame_i = int(parts[0])
                        class_i = int(parts[1])
                        _track_i = int(parts[2])
                        azi_i = int(parts[3])
                        ele_i = int(parts[4])
                        dist_i = int(parts[5])
                    except ValueError:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} has non-integer fields")
                        continue

                    classes_seen.add(class_i)
                    if frame_i < 0:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} negative frame")
                    if not (0 <= azi_i <= 360):
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} azimuth out of range")
                    if not (-90 <= ele_i <= 90):
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} elevation out of range")
                    if dist_i < 0:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} negative distance")

    print(f"Output dir: {OUTPUT_DIR}")
    print(f"WAV files: {total_wavs}, CSV files: {total_csvs}, annotation rows: {total_rows}")
    print(f"Classes seen: {sorted(classes_seen)}")

    if errors:
        print("SANITY CHECK FAILED")
        print(f"Errors: {len(errors)}")
        for err in errors[:30]:
            print(" -", err)
        if len(errors) > 30:
            print(f" ... and {len(errors) - 30} more")
    else:
        print("SANITY CHECK PASSED")
    print("=== SANITY CHECK END ===\n")

# -----------------------------------------------------------------

def main():

    out_wav_dir_train, out_csv_dir_train = prepare_split_dirs("dev-train")
    out_wav_dir_test, out_csv_dir_test = prepare_split_dirs("dev-test")

    df = pd.read_csv(ENV_CSV_PATH)

    grouped = df.groupby('sound_name')

    # Ищем "хорошие файлы" - без перекрывающихся интервалов событий
    good_env_files = []

    print("Ищем файлы без перекрывающихся событий\n")
    for name, group_df in tqdm(grouped):

        # Анализируем отдельный файл
        events = []
        classes = []
        for index, row in group_df.iterrows():
            events.append([row["start_time"], row["end_time"]])
            classes.append(row["class_name"])            
    
        if not has_overlap(events):
            good_env_files.append({"file": name, "events": events, "classes": classes})
    print(f"Найдено файлов без перекрывающихся событий: {len(good_env_files)}\n")
    
    voice_files = []
    
    print("Ищем файлы с голосом\n")
    with open(VOICE_JSON_PATH, 'r', encoding='utf-8') as voice_file:
        voice_files_dict = json.load(voice_file)
        for voice_filename in voice_files_dict:
            events = []
            classes = []
            for ev_ind in range(len(voice_files_dict[voice_filename])):
                events.append([voice_files_dict[voice_filename][ev_ind]["start"], voice_files_dict[voice_filename][ev_ind]["end"]])
                classes.append("Voices")
            voice_files.append({"file": voice_filename, "events": events, "classes": classes})            
    print(f"Найдено файлов с голосом: {len(voice_files)}\n")

    drone_directory_train = Path(DRONE_PATH_TRAIN)
    drone_train = list(drone_directory_train.rglob("*.wav"))

    drone_directory_test = Path(DRONE_PATH_TEST)
    drone_test = list(drone_directory_test.rglob("*.wav"))

    if not drone_train and not drone_test:
        raise RuntimeError(
            f"Не найдено ни одного .wav дронов в {drone_directory_train} и {drone_directory_test}"
        )
        
    for counter in tqdm(range(FILES_COUNT)):

        scenario = choose_scenario()

        if scenario == "drone_only":
            print(f"-- Сценарий <<Только дрон>> --")
            r_split = random.random()
            is_train_split = r_split <= TRAIN_FRACTION
            fname = pick_drone_path(drone_train, drone_test, is_train_split)
            if fname is None:
                print("Нет файлов дронов для выбранного split, пропуск.\n")
                continue
            print(f"Файл: {fname}")

            x_drone, fs = read_soundfile(fname, dtype="float32")

            room_dim = random_room_dim()
            mic_center = np.asarray(random_position(room_dim), dtype=np.float64)
            mic_xyz = MIC_POS_FIXED + mic_center

            duration = len(x_drone) / fs
            drone_traj = generate_linear_trajectory(duration, room_dim)

            # Генерация многоканального аудио
            y = render_source(x_drone, fs, drone_traj, room_dim, mic_xyz)

            # Масштабирование итогового сигнала
            peak_target = random.uniform(FLOAT_WAV_PEAK_TARGET_MIN, FLOAT_WAV_PEAK_TARGET_MAX)
            y = level_scale(y, peak_target=peak_target)

            # Пропускаем файлы в которых много тишины
            if has_silence(y, fs):
                print("В файле drone много тишины, пропускаем его.\n")
                continue

            annotations = generate_annotations_single(drone_traj, mic_center, class_index=0, events=None, classes=None)

            if is_train_split:
                name = f"fold3_{OUT_PREFIX}_drone_{counter}"
                write_soundfile(out_wav_dir_train / f"{name}.wav", y, fs)
                with open(out_csv_dir_train / f"{name}.csv", "w") as f:
                    f.write("\n".join(annotations))
            else:
                name = f"fold4_{OUT_PREFIX}_drone_{counter}"
                write_soundfile(out_wav_dir_test / f"{name}.wav", y, fs)
                with open(out_csv_dir_test / f"{name}.csv", "w") as f:
                    f.write("\n".join(annotations)) 

        elif scenario == "env_only":
            print(f"-- Сценарий <<Только окружение>> --- ")
            
            rvoice = random.random()        
            if rvoice <= VOICE_PROB:
                cur_environ_file = random.choice(voice_files)
                x_environ, fs = read_soundfile(Path(VOICE_DATASET_PATH) / cur_environ_file["file"], dtype="float32")
            else:            
                cur_environ_file = random.choice(good_env_files)
                x_environ, fs = read_soundfile(Path(ENV_DATASET_PATH) / cur_environ_file["file"], dtype="float32")                
            
            print(f'Файл: {cur_environ_file["file"]}')

            room_dim = random_room_dim()
            mic_center = np.asarray(random_position(room_dim), dtype=np.float64)
            mic_xyz = MIC_POS_FIXED + mic_center

            duration = len(x_environ) / fs
            environ_traj = generate_linear_trajectory(duration, room_dim)

            # Генерация многоканального аудио
            y = render_source(x_environ, fs, environ_traj, room_dim, mic_xyz)

            y_cut = cut_audio_by_intervals(y, fs, cur_environ_file["events"])

            # Пропускаем файлы в которых много тишины
            if has_silence(y_cut, fs):
                print("В файле environment много тишины, пропускаем его.\n")
                continue

            # Масштабирование итогового сигнала
            peak_target = random.uniform(FLOAT_WAV_PEAK_TARGET_MIN, FLOAT_WAV_PEAK_TARGET_MAX)
            y = level_scale(y, peak_target=peak_target)

            annotations = generate_annotations_single(environ_traj, mic_center, class_index=1, events=cur_environ_file["events"], classes=cur_environ_file["classes"])

            # В train пишем TRAIN_FRACTION файлов
            r = random.random()
            if r <= TRAIN_FRACTION:
                name = f"fold3_{OUT_PREFIX}_env_{counter}"
                write_soundfile(out_wav_dir_train / f"{name}.wav", y, fs)
                with open(out_csv_dir_train / f"{name}.csv", "w") as f:
                    f.write("\n".join(annotations))
            else:
                name = f"fold4_{OUT_PREFIX}_env_{counter}"
                write_soundfile(out_wav_dir_test / f"{name}.wav", y, fs)
                with open(out_csv_dir_test / f"{name}.csv", "w") as f:
                    f.write("\n".join(annotations)) 

        elif scenario == "both":
            print("-- Сценарий <<Дрон + окружение>> --")

            rvoice = random.random()            
            if rvoice <= VOICE_PROB:
                cur_environ_file = random.choice(voice_files)
                x_environ, fs_env = read_soundfile(Path(VOICE_DATASET_PATH) / cur_environ_file["file"], dtype="float32")
            else:            
                cur_environ_file = random.choice(good_env_files)
                x_environ, fs_env = read_soundfile(Path(ENV_DATASET_PATH) / cur_environ_file["file"], dtype="float32")                

            r_split = random.random()
            is_train_split = r_split <= TRAIN_FRACTION
            drone_path = pick_drone_path(drone_train, drone_test, is_train_split)
            if drone_path is None:
                print("Нет файлов дронов для выбранного split, пропуск.\n")
                continue
            x_drone, fs = read_soundfile(drone_path, dtype="float32")
            print(f"Файлы: БПЛА {drone_path}, окружение {cur_environ_file['file']}")

            room_dim = random_room_dim()
            mic_center = np.asarray(random_position(room_dim), dtype=np.float64)
            mic_xyz = MIC_POS_FIXED + mic_center

            duration_environ = len(x_environ) / fs
            duration_drone = len(x_drone) / fs

            min_samples = min(len(x_environ), len(x_drone))
            min_duration = min(duration_environ, duration_drone)

            x_environ = x_environ[:min_samples]
            x_drone = x_drone[:min_samples]

            environ_traj = generate_linear_trajectory(min_duration, room_dim)
            drone_traj = generate_linear_trajectory(min_duration, room_dim)

            y_environ = render_source(x_environ, fs, environ_traj, room_dim, mic_xyz)
            y_cut = cut_audio_by_intervals(y_environ, fs, cur_environ_file["events"])

            # Пропускаем файлы в которых много тишины
            if has_silence(y_cut, fs):
                print(f"В файле environment {cur_environ_file['file']} много тишины, пропускаем его.\n")
                continue

            y_drone = render_source(x_drone, fs, drone_traj, room_dim, mic_xyz)
            if has_silence(y_drone, fs):
                print("В файле drone много тишины, пропускаем его.\n")
                continue

            target_snr_db = random.uniform(SNR_MIN_DB, SNR_MAX_DB)
            y = mix_drone_and_environment(
                y_drone, y_environ, fs, cur_environ_file["events"], target_snr_db
            )

            # Масштабирование итогового сигнала
            peak_target = random.uniform(FLOAT_WAV_PEAK_TARGET_MIN, FLOAT_WAV_PEAK_TARGET_MAX)
            y = level_scale(y, peak_target=peak_target)

            annotations = generate_annotations_both(drone_traj, environ_traj, mic_center, events=cur_environ_file["events"], classes=cur_environ_file["classes"])

            if is_train_split:
                name = f"fold3_{OUT_PREFIX}_both_{counter}"
                write_soundfile(out_wav_dir_train / f"{name}.wav", y, fs)
                with open(out_csv_dir_train / f"{name}.csv", "w") as f:
                    f.write("\n".join(annotations))
            else:
                name = f"fold4_{OUT_PREFIX}_both_{counter}"
                write_soundfile(out_wav_dir_test / f"{name}.wav", y, fs)
                with open(out_csv_dir_test / f"{name}.csv", "w") as f:
                    f.write("\n".join(annotations))  



if __name__ == "__main__":
    args = parse_args()
    configure_from_args(args)
    main()
    if args.run_sanity_check:
        run_sanity_check(expected_fs=FS, expected_n_mics=N_MICS)
