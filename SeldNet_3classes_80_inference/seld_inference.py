"""
Модуль инференса для модели SELDNet (SeldNet_3classes_80).
Не зависит от других каталогов — содержит все необходимые классы и функции.

Использование:
    from seld_inference import SELDInference
    
    # Вариант 1: передача параметров явно
    params = {
        'fs': 250000,
        'hop_len_s': 0.02,
        'label_hop_len_s': 0.04,
        'nb_mel_bins': 256,
        'mel_fmax_hz': 5000.0,
        'n_mics': 4,
        'unique_classes': 3,
        'label_sequence_length': 5,
        'dataset': 'mic',
        'use_salsalite': False,
        'multi_accdoa': True,
        'nb_cnn2d_filt': 64,
        'f_pool_size': [4, 4, 2],
        'dropout_rate': 0.05,
        'nb_heads': 8,
        'nb_self_attn_layers': 2,
        'nb_rnn_layers': 2,
        'rnn_size': 128,
        'nb_fnn_layers': 1,
        'fnn_size': 128,
    }
    infer = SELDInference(
        weights_path='path/to/weights.pth',
        params=params,
        task_id='6'
    )
    result = infer.infer(audio_path='path/to/audio.wav')
    
    # Вариант 2: использование параметров по умолчанию
    infer = SELDInference(
        weights_path='path/to/weights.pth',
        task_id='6'
    )
    result = infer.infer(audio_path='path/to/audio.wav')
"""
from __future__ import annotations

import os
import math
import wave
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import joblib
from typing import Any, Dict, List, Optional, Tuple, Set


# =============================================================================
# Параметры по умолчанию для SeldNet_3classes_80
# =============================================================================
DEFAULT_PARAMS_80 = dict(
    fs=250000,
    hop_len_s=0.02,
    label_hop_len_s=0.04,
    nb_mel_bins=256,
    mel_fmin_hz=0.0,
    mel_fmax_hz=5000.0,
    n_mics=4,
    use_salsalite=False,
    fmin_doa_salsalite=50,
    fmax_doa_salsalite=2000,
    fmax_spectra_salsalite=5000,
    model='seldnet',
    multi_accdoa=True,
    unique_classes=3,
    label_sequence_length=5,
    dropout_rate=0.05,
    nb_cnn2d_filt=64,
    f_pool_size=[4, 4, 2],
    t_pool_size=[2, 1, 1],
    nb_heads=8,
    nb_self_attn_layers=2,
    nb_rnn_layers=2,
    rnn_size=128,
    nb_fnn_layers=1,
    fnn_size=128,
    dataset='mic',
    raw_chunks=False,
)


# =============================================================================
# Требуемые параметры для каждого task_id
# =============================================================================
REQUIRED_PARAMS_BY_TASK: Dict[str, Set[str]] = {
    '2': {  # FOA, single-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'unique_classes', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '3': {  # FOA, multi-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'unique_classes', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '4': {  # MIC, single-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'unique_classes', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '5': {  # MIC, SALSA, single-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'unique_classes', 'label_sequence_length', 'dataset', 'use_salsalite',
        'fmin_doa_salsalite', 'fmax_doa_salsalite', 'fmax_spectra_salsalite', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '6': {  # MIC, multi-ACCDOA (основная конфигурация SeldNet_3classes_80)
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'unique_classes', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '7': {  # MIC, SALSA, multi-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'unique_classes', 'label_sequence_length', 'dataset', 'use_salsalite',
        'fmin_doa_salsalite', 'fmax_doa_salsalite', 'fmax_spectra_salsalite', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
}

# Параметры, используемые непосредственно в инференсе (минимальный набор)
INFERENCE_ONLY_PARAMS = {
    'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
    'n_mics', 'dataset', 'use_salsalite', 'multi_accdoa', 'unique_classes',
    'label_sequence_length', 'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate',
    'nb_heads', 'nb_self_attn_layers', 'nb_rnn_layers', 'rnn_size',
    'nb_fnn_layers', 'fnn_size',
}


def _recompute_time_derived(params: Dict[str, Any]) -> None:
    """Вычисляет производные временные параметры."""
    params['feature_label_resolution'] = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = (
        params['label_sequence_length'] * params['feature_label_resolution']
    )


def nCr(n: int, r: int) -> int:
    """Биномиальный коэффициент."""
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


# =============================================================================
# Модели нейронной сети
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    """SELDNet модель с CNN + Bi-GRU + Multi-Head Self-Attention + FNN."""
    
    def __init__(self, in_feat_shape: Tuple[int, ...], out_shape: Tuple[int, ...], params: Dict[str, Any]):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params = params
        self.conv_block_list = nn.ModuleList()
        
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                in_ch = params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1]
                self.conv_block_list.append(ConvBlock(in_channels=in_ch, out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params['rnn_size'],
            num_layers=params['nb_rnn_layers'],
            batch_first=True,
            dropout=params['dropout_rate'],
            bidirectional=True
        )

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(
                embed_dim=self.params['rnn_size'],
                num_heads=self.params['nb_heads'],
                dropout=self.params['dropout_rate'],
                batch_first=True
            ))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                in_f = params['fnn_size'] if fc_cnt else self.params['rnn_size']
                self.fnn_list.append(nn.Linear(in_f, params['fnn_size'], bias=True))
        out_dim = out_shape[-1] if isinstance(out_shape, (tuple, list)) else out_shape
        self.fnn_list.append(nn.Linear(
            params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'],
            out_dim,
            bias=True
        ))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, vid_feat=None):
        """
        Args:
            x: тензор формы (batch_size, mic_channels, time_steps, mel_bins)
        """
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)
        return doa


# =============================================================================
# Извлечение признаков
# =============================================================================
class FeatureExtractor:
    """Класс для извлечения и нормализации признаков из аудио."""
    
    def __init__(self, params: Dict[str, Any], scaler_path: Optional[str] = None):
        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._label_hop_len_s = params['label_hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_mel_bins = params['nb_mel_bins']
        self._dataset = params['dataset']
        self._use_salsalite = params.get('use_salsalite', False)
        self._multi_accdoa = params.get('multi_accdoa', True)
        self._nb_unique_classes = params['unique_classes']
        self._n_mics = params['n_mics']
        
        # Mel filterbank
        fmin_mel = float(params.get('mel_fmin_hz', 0.0))
        fmax_mel = params.get('mel_fmax_hz', None)
        mel_kw = dict(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins, fmin=fmin_mel)
        if fmax_mel is not None:
            mel_kw['fmax'] = float(fmax_mel)
        self._mel_wts = librosa.filters.mel(**mel_kw).T
        
        # Загрузка скалера
        self._scaler = None
        if scaler_path is not None and os.path.exists(scaler_path):
            self._scaler = joblib.load(scaler_path)
    
    @staticmethod
    def _next_greater_power_of_2(x: int) -> int:
        return 2 ** (x - 1).bit_length()
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Загружает аудиофайл и возвращает сигнал и частоту дискретизации."""
        audio, fs = librosa.load(audio_path, sr=self._fs, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.T  # (samples, channels)
        return audio, fs
    
    def _spectrogram(self, audio_input: np.ndarray, _nb_frames: int) -> np.ndarray:
        """Вычисляет спектрограмму для каждого канала."""
        _nb_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(
                np.asfortranarray(audio_input[:, ch_cnt]),
                n_fft=self._nfft,
                hop_length=self._hop_len,
                win_length=self._win_len,
                window='hann'
            )
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T
    
    def _get_mel_spectrogram(self, linear_spectra: np.ndarray) -> np.ndarray:
        """Преобразует линейную спектрограмму в мел-спектрограмму."""
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat
    
    def _get_gcc(self, linear_spectra: np.ndarray) -> np.ndarray:
        """Вычисляет GCC-PHAT признаки для пар микрофонов."""
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Извлекает признаки из аудиофайла."""
        audio_input, fs = self._load_audio(audio_path)
        nb_feat_frames = int(len(audio_input) / float(self._hop_len))
        
        spect = self._spectrogram(audio_input, nb_feat_frames)
        mel_spect = self._get_mel_spectrogram(spect)
        
        if self._dataset == 'mic' and not self._use_salsalite:
            gcc = self._get_gcc(spect)
            feat = np.concatenate((mel_spect, gcc), axis=-1)
        else:
            feat = mel_spect
        
        # Нормализация
        if self._scaler is not None:
            feat = self._scaler.transform(feat)
        
        return feat
    
    def reshape_features_for_model(self, feat: np.ndarray) -> np.ndarray:
        """Формирует батчи для подачи в модель."""
        # Определяем размерность признака на один фрейм
        feat_per_frame = feat.shape[1] if feat.ndim == 3 else feat.shape[-1]
        
        # Вычисляем количество feature frames на label frame
        feature_label_resolution = int(self._label_hop_len // self._hop_len)
        feature_sequence_length = 5 * feature_label_resolution  # label_sequence_length=5
        
        # Обрезаем до кратного feature_sequence_length
        nb_frames = feat.shape[0]
        nb_chunks = nb_frames // feature_sequence_length
        
        if nb_chunks == 0:
            # Если аудио слишком короткое, дополняем нулями
            pad_len = feature_sequence_length - nb_frames
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
            nb_chunks = 1
        
        feat = feat[:nb_chunks * feature_sequence_length]
        
        # Reshape: (nb_chunks, feature_sequence_length, feat_dim)
        feat = feat.reshape((nb_chunks, feature_sequence_length, -1))
        
        # Transpose для модели: (batch, channels, time, freq)
        # Приводим к виду (batch, channels=1, time=feature_sequence_length, freq=feat_dim)
        feat = feat[:, np.newaxis, :, :]
        
        return feat.astype(np.float32)


# =============================================================================
# Постобработка результатов
# =============================================================================
def decode_multi_accdoa_output(output: np.ndarray, threshold: float = 0.5) -> List[Dict]:
    """
    Декодирует выход модели multi-ACCDOA в список детекций.
    
    Args:
        output: массив формы (frames, tracks*4*classes) или (frames, tracks, 4, classes)
        threshold: порог активности класса
    
    Returns:
        Список словарей с детекциями: [{'frame': int, 'class': int, 'x': float, 'y': float, 'z': float, 'dist': float}, ...]
    """
    if output.ndim == 2:
        # Разворачиваем (frames, tracks*4*classes) -> (frames, tracks, 4, classes)
        nb_tracks = 3
        nb_axes = 4
        nb_classes = output.shape[1] // (nb_tracks * nb_axes)
        output = output.reshape((output.shape[0], nb_tracks, nb_axes, nb_classes))
    
    detections = []
    nb_frames, nb_tracks, nb_axes, nb_classes = output.shape
    
    for frame_idx in range(nb_frames):
        for track_idx in range(nb_tracks):
            for class_idx in range(nb_classes):
                activity = output[frame_idx, track_idx, 0, class_idx]
                if activity > threshold:
                    x = output[frame_idx, track_idx, 1, class_idx]
                    y = output[frame_idx, track_idx, 2, class_idx]
                    z = output[frame_idx, track_idx, 3, class_idx]
                    dist = output[frame_idx, track_idx, 3, class_idx] if nb_axes > 3 else 1.0
                    
                    detections.append({
                        'frame': frame_idx,
                        'class': class_idx,
                        'track': track_idx,
                        'activity': float(activity),
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'dist': float(dist)
                    })
    
    return detections


def cartesian_to_polar(x: float, y: float, z: float) -> Tuple[float, float]:
    """Преобразует декартовы координаты в полярные (азимут, угол места) в градусах."""
    azimuth = np.arctan2(y, x) * 180 / np.pi
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    return float(azimuth), float(elevation)


# =============================================================================
# Основной класс инференса
# =============================================================================
class SELDInference:
    """
    Класс для выполнения инференса модели SELDNet.
    
    Пример использования:
        # Вариант 1: с явной передачей параметров
        params = {'fs': 250000, 'n_mics': 4, ...}
        infer = SELDInference(
            weights_path='models/seldnet_model_final.pth',
            params=params,
            task_id='6'
        )
        
        # Вариант 2: с использованием параметров по умолчанию
        infer = SELDInference(
            weights_path='models/seldnet_model_final.pth',
            task_id='6'
        )
        results = infer.infer('audio.wav')
    """
    
    def __init__(
        self,
        weights_path: str,
        scaler_path: Optional[str] = None,
        task_id: str = '6',
        params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            weights_path: путь к файлу весов модели (.pth)
            scaler_path: путь к файлу скалера для нормализации (опционально)
            task_id: идентификатор задачи (влияет на параметры)
            params: словарь параметров модели. Если None, используются параметры по умолчанию.
                   Должен содержать все необходимые параметры для указанного task_id.
            device: устройство для вычислений ('cuda', 'cpu', или None для автовыбора)
            threshold: порог активности для детекции событий
        """
        self.threshold = threshold
        
        # Выбор устройства
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Загрузка параметров для задачи
        self.params = self._get_params_for_task(task_id, params)
        
        # Валидация параметров (самая первая операция - до любых файловых операций)
        self._validate_params_for_task(task_id)
        
        # Создание экстрактора признаков (может загружать scaler файл)
        self.feature_extractor = FeatureExtractor(self.params, scaler_path=scaler_path)
        
        # Создание и загрузка модели
        self.model = self._create_and_load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _validate_params_for_task(self, task_id: str) -> None:
        """
        Проверяет наличие всех необходимых параметров для указанного task_id.
        
        Args:
            task_id: идентификатор задачи
            
        Raises:
            ValueError: если отсутствуют необходимые параметры
        """
        if task_id not in REQUIRED_PARAMS_BY_TASK:
            raise ValueError(f"Неподдерживаемый task_id: {task_id}. Допустимые значения: {list(REQUIRED_PARAMS_BY_TASK.keys())}")
        
        required_params = REQUIRED_PARAMS_BY_TASK[task_id]
        missing_params = required_params - set(self.params.keys())
        
        if missing_params:
            raise ValueError(
                f"Для task_id={task_id} отсутствуют следующие обязательные параметры: {sorted(missing_params)}"
            )
    
    @staticmethod
    def get_required_params(task_id: str) -> Set[str]:
        """
        Возвращает множество обязательных параметров для указанного task_id.
        
        Args:
            task_id: идентификатор задачи
            
        Returns:
            Множество имен обязательных параметров
            
        Raises:
            ValueError: если task_id не поддерживается
        """
        if task_id not in REQUIRED_PARAMS_BY_TASK:
            raise ValueError(f"Неподдерживаемый task_id: {task_id}. Допустимые значения: {list(REQUIRED_PARAMS_BY_TASK.keys())}")
        return REQUIRED_PARAMS_BY_TASK[task_id].copy()
    
    @staticmethod
    def get_inference_params() -> Set[str]:
        """
        Возвращает минимальный набор параметров, используемых непосредственно в инференсе.
        
        Returns:
            Множество имен параметров
        """
        return INFERENCE_ONLY_PARAMS.copy()
    
    def _get_params_for_task(self, task_id: str, user_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Возвращает параметры для заданной задачи."""
        # Начинаем с параметров по умолчанию
        params = DEFAULT_PARAMS_80.copy()
        
        # Переопределяем пользовательскими параметрами
        if user_params is not None:
            params.update(user_params)
        
        # Применение настроек для конкретных задач (если не переопределено пользователем)
        if task_id == '6':
            params.setdefault('dataset', 'mic')
            params.setdefault('use_salsalite', False)
            params.setdefault('multi_accdoa', True)
            params.setdefault('n_mics', 4)
        elif task_id == '4':
            params.setdefault('dataset', 'mic')
            params.setdefault('use_salsalite', False)
            params.setdefault('multi_accdoa', False)
        elif task_id == '5':
            params.setdefault('dataset', 'mic')
            params.setdefault('use_salsalite', True)
            params.setdefault('multi_accdoa', False)
        elif task_id == '7':
            params.setdefault('dataset', 'mic')
            params.setdefault('use_salsalite', True)
            params.setdefault('multi_accdoa', True)
        elif task_id == '2':
            params.setdefault('dataset', 'foa')
            params.setdefault('multi_accdoa', False)
        elif task_id == '3':
            params.setdefault('dataset', 'foa')
            params.setdefault('multi_accdoa', True)
        
        _recompute_time_derived(params)
        
        # Настройка t_pool_size
        params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]
        
        # Вычисление количества каналов
        nm = int(params['n_mics'])
        if params.get('use_salsalite', False):
            params['nb_channels'] = 7
        else:
            params['nb_channels'] = nm + (nm * (nm - 1)) // 2
        
        return params
    
    def _create_and_load_model(self, weights_path: str) -> torch.nn.Module:
        """Создаёт и загружает модель из файла весов."""
        # Входная форма: (batch, channels, time, freq)
        channels = self.params['nb_channels']
        time_steps = self.params['feature_sequence_length']
        freq_bins = self.params['nb_mel_bins'] * self.params['n_mics'] + \
                   (self.params['n_mics'] * (self.params['n_mics'] - 1) // 2) * self.params['nb_mel_bins'] \
                   if not self.params['use_salsalite'] else self.params['nb_mel_bins'] * 7
        
        # Упрощённый расчёт: mel по каждому каналу + gcc по парам
        nm = self.params['n_mics']
        mel_dim = self.params['nb_mel_bins'] * nm
        gcc_dim = self.params['nb_mel_bins'] * (nm * (nm - 1) // 2)
        freq_dim = mel_dim + gcc_dim if not self.params['use_salsalite'] else self.params['nb_mel_bins'] * 7
        
        in_feat_shape = (None, channels, time_steps, freq_dim)
        
        # Выходная форма: (batch, frames, tracks*4*classes)
        nb_tracks = 3
        nb_axes = 4  # act, x, y, z (или dist)
        nb_classes = self.params['unique_classes']
        out_dim = nb_tracks * nb_axes * nb_classes
        out_shape = (None, time_steps, out_dim)
        
        model = SeldModel(in_feat_shape, out_shape, self.params)
        
        # Загрузка весов
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
        
        return model
    
    def infer(self, audio_path: str) -> List[Dict]:
        """
        Выполняет инференс для аудиофайла.
        
        Args:
            audio_path: путь к аудиофайлу
        
        Returns:
            Список детекций: [{'frame': int, 'class': int, 'track': int, 'activity': float, 
                              'x': float, 'y': float, 'z': float, 'dist': float,
                              'azimuth': float, 'elevation': float}, ...]
        """
        # Извлечение признаков
        feat = self.feature_extractor.extract_features(audio_path)
        
        # Формирование батча
        feat_batch = self.feature_extractor.reshape_features_for_model(feat)
        feat_tensor = torch.from_numpy(feat_batch).float().to(self.device)
        
        # Инференс
        with torch.no_grad():
            output = self.model(feat_tensor)
            output = output.cpu().numpy()
        
        # Декодирование
        # output имеет форму (batch, time_steps, tracks*4*classes)
        output = output.reshape((-1, output.shape[-1]))  # (batch*time_steps, tracks*4*classes)
        
        detections = decode_multi_accdoa_output(output, threshold=self.threshold)
        
        # Добавление полярных координат
        for det in detections:
            az, el = cartesian_to_polar(det['x'], det['y'], det['z'])
            det['azimuth'] = az
            det['elevation'] = el
        
        return detections
    
    def infer_file(self, audio_path: str, output_csv: Optional[str] = None) -> str:
        """
        Выполняет инференс и сохраняет результаты в CSV файл в формате DCASE.
        
        Args:
            audio_path: путь к аудиофайлу
            output_csv: путь к выходному CSV файлу (опционально)
        
        Returns:
            Строка с результатами в формате CSV
        """
        detections = self.infer(audio_path)
        
        # Формирование строк CSV
        lines = []
        for det in detections:
            # Формат: frame, class, track, x, y, z, dist, azimuth, elevation
            line = "{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.1f},{:.1f}".format(
                det['frame'],
                det['class'],
                det['track'],
                det['x'],
                det['y'],
                det['z'],
                det['dist'],
                det['azimuth'],
                det['elevation']
            )
            lines.append(line)
        
        csv_content = "frame,class,track,x,y,z,dist,azimuth,elevation\n" + "\n".join(lines)
        
        if output_csv:
            with open(output_csv, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        
        return csv_content


# =============================================================================
# Точка входа для командной строки
# =============================================================================
if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Инференс SELDNet модели')
    parser.add_argument('--weights', type=str, required=False, default=None, help='Путь к файлу весов .pth (не требуется при --list-params)')
    parser.add_argument('--scaler', type=str, default=None, help='Путь к файлу скалера .joblib')
    parser.add_argument('--audio', type=str, required=False, default=None, help='Путь к аудиофайлу (не требуется при --list-params)')
    parser.add_argument('--output', type=str, default=None, help='Путь к выходному CSV файлу')
    parser.add_argument('--task-id', type=str, default='6', help='ID задачи (по умолчанию 6)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог детекции')
    parser.add_argument('--device', type=str, default=None, help='Устройство (cuda/cpu)')
    parser.add_argument('--params', type=str, default=None, 
                        help='Путь к JSON файлу с параметрами модели (опционально)')
    parser.add_argument('--list-params', action='store_true', 
                        help='Вывести список требуемых параметров для task_id и выйти')
    
    args = parser.parse_args()
    
    # Если запрошен список параметров, выводим и выходим
    if args.list_params:
        try:
            required = SELDInference.get_required_params(args.task_id)
            inference_only = SELDInference.get_inference_params()
            print(f"Требуемые параметры для task_id={args.task_id}:")
            for param in sorted(required):
                print(f"  - {param}")
            print(f"\nМинимальный набор для инференса ({len(inference_only)} параметров):")
            for param in sorted(inference_only):
                print(f"  - {param}")
        except ValueError as e:
            print(f"Ошибка: {e}")
        exit(0)
    
    # Проверка обязательных аргументов для режима инференса
    if not args.weights or not args.audio:
        parser.error("Для режима инференса требуются аргументы --weights и --audio")
    
    # Загрузка пользовательских параметров из JSON файла
    user_params = None
    if args.params:
        with open(args.params, 'r', encoding='utf-8') as f:
            user_params = json.load(f)
        print(f"Загружены параметры из {args.params}")
    
    infer = SELDInference(
        weights_path=args.weights,
        scaler_path=args.scaler,
        task_id=args.task_id,
        params=user_params,
        device=args.device,
        threshold=args.threshold
    )
    
    result = infer.infer_file(args.audio, args.output)
    print(result)
