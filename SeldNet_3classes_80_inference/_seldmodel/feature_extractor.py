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
import sys


def nCr(n: int, r: int) -> int:
    """Биномиальный коэффициент."""
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


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
        if fmax_mel is not None:
            fmax_mel = float(fmax_mel)
        mel_kw = dict(
            sr=self._fs, 
            n_fft=self._nfft, 
            n_mels=self._nb_mel_bins, 
            fmin=fmin_mel, 
            fmax = fmax_mel,
        )
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
    
    def extract_features(self, audio_arr) -> np.ndarray:
        """Извлекает признаки из аудиофайла."""
        nb_feat_frames = int(len(audio_arr) / float(self._hop_len))
        
        spect = self._spectrogram(audio_arr, nb_feat_frames)
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
