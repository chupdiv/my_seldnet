"""
SELDModel - универсальный модуль инференса для SELD задач.

Поддерживаемые модели:
- 6: Mel_Gcc_SeldNet
- 7: Salsa_SeldNet  
- 10: NGCC_SeldNet
- 33: GCC-CSTFormer
- 34: Salsa-CSTFormer
- 333: NGCC-CSTFormer

Тип задачи определяется параметром model_kind в разделе [tags] конфигурационного файла.
"""

import os
import numpy as np
import joblib
import torch as pt
import json
import ast
from typing import List, Dict, Any

from acoustic.doa.doa_tools import cartesian_to_spherical

from core.logger import logger

from ..protomodel import ProtoModel

# Импорт моделей и утилит из _seldnet
from ._seldnet.seldnet_model import SeldModel
from ._ngcc.ngcc_seld_model import NGCCModel
from ._seldnet.tools import (
    next_greater_power_of_2,
    seldnet_get_spectrogram_for_audio,
    get_multi_accdoa_predictions,
    extract_prediction,
    get_melgcc_feats,
    get_melfsgcc_feats,
    get_salsalite_feats,
    get_salsa_components_count,
    NFFT
)
from ._seldnet.seldnet_parser import SELDnetOutputParser
from ._seldnet.ngcc_seld_params import load_params_with_optional_task

# Импорт CSTFormer модели
from ._cstformer.CST_former_model import CST_former

EPS = 1e-10
MAX_SECONDARY_TRACKS = 2


class SELDModel(ProtoModel):
    """
    Универсальный класс для инференса SELD моделей.
    
    Поддерживает архитектуры:
    - SeldNet (Mel+GCC, SALSA-LITE)
    - NGCC-SeldNet
    - CSTFormer (GCC, SALSA-LITE, NGCC)
    """
    
    def __init__(self, path: str, device: str = 'cpu', **kwargs) -> None:
        super().__init__(path, device)
        
        if 'signal_duration_s' in kwargs.keys():
            self.signal_duration = kwargs['signal_duration_s']
        else:
            self.signal_duration = 0.2
        
        # Определяем тип модели из конфига
        self.model_kind = self._get_model_kind()
        
        self.load()
        
        # Используем парсер для разбора выхода модели
        self.parser = SELDnetOutputParser(
            nb_classes=len(self.classes_list),
            hop_length_ms=self.params['label_hop_len_s'] * 1000,
            class_names=self.classes_list
        )
    
    def _get_model_kind(self) -> str:
        """
        Извлекает model_kind из раздела [tags] конфигурационного файла.
        
        Ожидаемые значения:
        - '6' или 'mel_gcc_seldnet': Mel+GCC+SeldNet
        - '7' или 'salsa_seldnet': SALSA-LITE+SeldNet
        - '10' или 'ngcc_seldnet': NGCC-SeldNet
        - '33' или 'gcc_cstformer': GCC+CSTFormer
        - '34' или 'salsa_cstformer': SALSA-LITE+CSTFormer
        - '333' или 'ngcc_cstformer': NGCC+CSTFormer
        """
        tags = self.settings.get('tags', {})
        model_kind = tags.get('model_kind', None)
        
        if model_kind is None:
            # Пытаемся определить по другим параметрам
            model = tags.get('model', 'seldnet').lower()
            task_id = tags.get('task_id', None)
            
            if task_id is not None:
                model_kind = str(task_id)
            elif 'cstformer' in model or 'cst_former' in model:
                model_kind = '33'  # По умолчанию GCC-CSTFormer
            elif 'ngcc' in model:
                model_kind = '10'
            elif 'salsa' in model:
                model_kind = '7'
            else:
                model_kind = '6'  # По умолчанию Mel+GCC+SeldNet
            
            logger.warning(f'model_kind не задан в [tags], используем значение по умолчанию: {model_kind}')
        
        return str(model_kind).lower()
    
    def load(self):
        """Загрузка модели и параметров."""
        SIGNAL_LEN = self.signal_duration
        DEVICE = self.device
        MODELPATH = self.files['model']
        AUDIOSCALERPATH = self.files.get('scaler', None)
        
        # Загружаем параметры
        raw = {key: ast.literal_eval(val) for key, val in self.settings['params'].items()}
        self.params = load_params_with_optional_task(raw)
        
        # Базовые параметры
        self.n_mics = self.params['n_mics']
        self.n_pairs = self.n_mics * (self.n_mics - 1) // 2
        self.classes_list = self.params.get('classes_list', [])
        self.target_class = self.params.get('target_class', 0)
        self.fs = self.params['fs']
        
        N_CLASSES = len(self.classes_list)
        
        # Определяем архитектуру по model_kind
        model_kind = self.model_kind
        
        # Параметры признаков
        USE_SALSALITE = self.params.get('use_salsalite', False)
        USE_NGCC = self.params.get('use_ngcc', False)
        
        # Автоматическое определение use_salsalite для model_kind 7 и 34
        if model_kind in ['7', '34']:
            USE_SALSALITE = True
            self.params['use_salsalite'] = True
        
        # Автоматическое определение use_ngcc для model_kind 10 и 333
        if model_kind in ['10', '333']:
            USE_NGCC = True
            self.params['use_ngcc'] = True
        
        if USE_SALSALITE:
            self.n_feats = self.n_mics + (self.n_mics - 1)  # 9=5+4 для 5 микрофонов
            self.n_components = get_salsa_components_count(
                fs=self.fs,
                n_fft=NFFT,
                fmin_doa=self.params.get('fmin_doa_salsalite', 50),
                fmax_spectra=self.params.get('fmax_spectra_salsalite', 9000),
            )
        else:
            self.n_components = self.params['nb_mel_bins']
            if USE_NGCC:
                # Для NGCC количество каналов вычисляется иначе
                ngcc_out = self.params.get('ngcc_out_channels', 8)
                if self.params.get('use_mel', True):
                    self.n_feats = int(ngcc_out * self.n_mics * (self.n_mics - 1) / 2 + self.n_mics)
                else:
                    self.n_feats = int(ngcc_out * self.n_mics * (1 + (self.n_mics - 1) / 2))
            else:
                self.n_feats = self.n_mics + self.n_pairs  # 15=5+10 для 5 микрофонов
        
        N_TIMES = int(SIGNAL_LEN / self.params['label_hop_len_s'])
        MAX_SIGNAL_LEN = self.params.get('max_audio_len_s', -1)
        
        if (MAX_SIGNAL_LEN > 0) and (SIGNAL_LEN > MAX_SIGNAL_LEN):
            SIGNAL_LEN = np.float32(MAX_SIGNAL_LEN)
        
        # Загрузка весов модели
        state_dict = None
        try:
            try:
                state_dict = pt.load(MODELPATH, map_location=DEVICE, weights_only=True)
            except TypeError:
                state_dict = pt.load(MODELPATH, map_location=DEVICE)
            
            # Валидация ключей в зависимости от типа модели
            if model_kind in ['10', '333']:  # NGCC модели
                key = 'ngcc.backbone.conv.0.low_hz_'
                if key not in state_dict and 'fnn_list.1.weight' not in state_dict:
                    logger.warning(f'Модель {MODELPATH} может не быть NGCC моделью')
            else:
                key = 'fnn_list.1.weight'
                if key not in state_dict:
                    # Для CSTFormer может быть другой ключ
                    if 'fc_layer' not in str(list(state_dict.keys())):
                        logger.warning(f'Модель {MODELPATH} имеет нестандартную структуру')
            
            # Определение числа классов из весов
            if 'fnn_list.1.weight' in state_dict:
                detected_n_classes = state_dict['fnn_list.1.weight'].shape[0] // 12
                if N_CLASSES == 0:
                    logger.warning('В параметрах модели отсутствует список классов. Число классов определено автоматически.')
                    N_CLASSES = detected_n_classes
                    self.classes_list = list(range(N_CLASSES))
                elif detected_n_classes != N_CLASSES:
                    logger.warning(f'Число классов в модели ({detected_n_classes}) не совпадает с указанным в конфигурации ({N_CLASSES}).')
            
            logger.info(f'Загружена SELD модель (model_kind={model_kind}) с {N_CLASSES} классами')
            
        except Exception:
            logger.exception('Ошибка при загрузке весов модели')
            raise
        
        # Создание модели
        try:
            data_in = (1, self.n_feats, N_TIMES, self.n_components)
            data_out = (1, self.n_feats, 4 * 3 * N_CLASSES)
            
            vid_data_in = None
            
            # Загружаем scaler только для non-NGCC моделей
            self.scaler = None
            if not USE_NGCC and AUDIOSCALERPATH and os.path.exists(AUDIOSCALERPATH):
                self.scaler = joblib.load(AUDIOSCALERPATH)
            
            # Инициализация модели в зависимости от type
            if model_kind in ['33', '34', '333']:
                # CSTFormer
                self.model_type = 'cstformer'
                self.model = CST_former(data_in, data_out, self.params, vid_data_in).to(DEVICE)
            elif model_kind in ['10']:
                # NGCC-SeldNet
                self.model_type = 'ngccmodel'
                self.model = NGCCModel(data_in, data_out, self.params, vid_data_in).to(DEVICE)
            else:
                # SeldNet (Mel+GCC или SALSA)
                self.model_type = 'seldnet'
                self.model = SeldModel(data_in, data_out, self.params, vid_data_in).to(DEVICE)
            
            # Загрузка весов
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
        except Exception:
            logger.exception('Ошибка инициализации модели SELD обработки аудио')
            raise
    
    def predict(self, S):
        """
        Вычисляет направление на источник при помощи SELD модели.
        
        Args:
            S: Акустический сигнал
            
        Returns:
            dict с полями: azimuth, elevation, class, prob, distance, track_id, secondary_tracks
        """
        PROB_THRESHOLD = self.prob_threshold
        N_CLASSES = len(self.classes_list)
        
        # Проверка частоты дискретизации
        if S.fs != self.fs:
            logger.warning(f'Частота дискретизации сигнала {S.fs} отличается от частоты модели. Сигнал будет передискретизирован к частоте {self.fs}')
            S = S.resample(self.fs)
        
        # Подготовка входных данных
        USE_NGCC = self.params.get('use_ngcc', False) or self.model_kind in ['10', '333']
        
        if USE_NGCC:
            raw_chunks = self._get_raw_chunks(S)
            x = pt.from_numpy(raw_chunks.astype(np.float32)).to(self.device)
        else:
            feats = self._get_features(S)
            x = pt.from_numpy(feats.astype(np.float32)).to(self.device)
            x = pt.permute(x, (0, 1, 3, 2))  # [1, n_feats, n_times, n_components]
        
        # Получение предсказания
        with pt.no_grad():
            output = self.model(x)
            
            # Для CSTFormer берем последние кадры
            if self.model_type == 'cstformer':
                tail = int(self.params.get('cst_inference_last_frames', 2))
                tail = min(tail, output.shape[1])
                output = output[:, -tail:, :]
        
        # Парсинг выхода модели
        try:
            prediction = self.parser.parse_with_tracking(
                output.detach().cpu().numpy(),
                threshold=self.prob_threshold
            )
            
            frames = self._extract_frames_for_target_class(
                prediction=prediction,
                target_class=self.target_class,
            )
            
            if len(frames) == 0:
                return {'full_prediction': prediction}
            
            primary = frames[0]
            secondary_tracks = frames[1:]
            
            return {
                'azimuth': primary['azimuth'],
                'elevation': primary['elevation'],
                'class': primary['class'],
                'prob': primary['prob'],
                'distance': primary['distance'],
                'track_id': primary['track_id'],
                'secondary_tracks': secondary_tracks,
                'full_prediction': prediction,
            }
            
        except Exception:
            logger.exception('Ошибка парсинга выхода SELD, используется резервный парсер')
            
            confidence_level, coords = get_multi_accdoa_predictions(
                accdoa_in=output.detach().cpu().numpy(),
                nb_classes=N_CLASSES
            )
            probability, azimuth, elevation, distance = extract_prediction(
                confidence_level, coords, PROB_THRESHOLD, class_id=self.target_class
            )
            detected_class = self.classes_list[self.target_class]
            track_id = np.argmax(probability)
            
            return {
                'azimuth': azimuth[track_id],
                'elevation': elevation[track_id],
                'class': detected_class,
                'prob': probability[track_id],
                'distance': distance[track_id],
                'track_id': int(track_id),
                'secondary_tracks': [],
            }
    
    def _extract_frames_for_target_class(
        self,
        prediction,
        target_class: int,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает отсортированный список треков target_class для последнего кадра.
        """
        if prediction is None or not prediction.frames:
            return []
        
        series: List[Dict[str, Any]] = []
        for frame_id, frame in prediction.frames.items():
            class_events = [e for e in frame.events if e.class_idx == target_class]
            if not class_events:
                continue
            class_events.sort(key=lambda e: float(e.confidence), reverse=True)
            event = class_events[0]
            series.append({
                'track_id': int(event.track_id),
                'azimuth': float(event.azimuth),
                'elevation': float(event.elevation),
                'distance': float(event.distance),
                'prob': float(event.confidence),
                'class': event.class_name,
                'frame_id': frame_id,
            })
        
        series.sort(key=lambda e: float(e['prob']), reverse=True)
        return series
    
    def _get_features(self, S):
        """Вычисление признаков для SeldNet и CSTFormer (non-NGCC)."""
        USE_SALSALITE = self.params.get('use_salsalite', False)
        USE_FSGCC = self.params.get('use_fsgcc', False)
        
        HOP_LEN = int(self.params['hop_len_s'] * S.fs)
        WIN_LEN = 2 * HOP_LEN
        NFFT = next_greater_power_of_2(WIN_LEN)
        NFFT = min(NFFT, S.n_samples)
        
        spect = seldnet_get_spectrogram_for_audio(
            S.data,
            hop_len=HOP_LEN,
            win_len=WIN_LEN,
            n_fft=NFFT
        )
        
        if USE_SALSALITE:
            feat = get_salsalite_feats(
                linear_spectra=spect,
                n_fft=NFFT,
                fmin_doa=self.params.get('fmin_doa_salsalite', 50),
                fmax_doa=self.params.get('fmax_doa_salsalite', 2000),
                fmax_spectra=self.params.get('fmax_spectra_salsalite', 9000),
                fs=S.fs,
                sound_speed=343
            )
            self.n_components = feat.shape[-1] // self.n_feats
        elif USE_FSGCC:
            feat = get_melfsgcc_feats(
                linear_spectra=spect,
                n_mels=self.n_components,
                fs=self.fs,
                n_fft=NFFT,
                window_size=self.params.get('fsgcc_window_size', 256),
                hop_size=self.params.get('fsgcc_hop_size', 64),
                f_min_hz=self.params.get('f_min_hz', 0),
                f_max_hz=self.params.get('f_max_hz', None),
            )
        else:
            feat = get_melgcc_feats(
                spect,
                self.n_components,
                sr=float(self.fs),
                n_fft=NFFT,
                f_min_hz=self.params.get('f_min_hz', 0),
                f_max_hz=self.params.get('f_max_hz', None),
            )
        
        # Масштабирование признаков
        if self.scaler is not None:
            scaled_feat = self.scaler.transform(feat)
        else:
            scaled_feat = feat
        
        # Изменение формата для модели
        scaled_feat = scaled_feat.transpose()
        scaled_feat = scaled_feat.reshape(
            (self.n_feats, self.n_components, -1), order='C'
        )
        scaled_feat = np.expand_dims(scaled_feat, axis=0)
        
        return scaled_feat
    
    def _get_raw_chunks(self, S):
        """
        Возвращает сырые окна аудио в формате [B, M, T, L] для NGCC моделей.
        """
        hop = int(self.params['hop_len_s'] * S.fs)
        hop = max(hop, 1)
        
        data = S.data
        n_samples, n_mics = data.shape
        
        total = max(hop, (n_samples // hop) * hop)
        if n_samples < total:
            pad = np.zeros((total - n_samples, n_mics), dtype=data.dtype)
            data = np.vstack((data, pad))
        else:
            data = data[:total, :]
        
        n_frames = total // hop
        framed = data.reshape(n_frames, hop, n_mics)  # [T, L, M]
        framed = np.transpose(framed, (2, 0, 1))       # [M, T, L]
        framed = np.expand_dims(framed, axis=0)        # [1, M, T, L]
        
        return framed
