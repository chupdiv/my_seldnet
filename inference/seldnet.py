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

EPS = 1e-10
MAX_SECONDARY_TRACKS = 2

class SELDNet(ProtoModel):
    def __init__(self, path:str, device:str='cpu', **kwargs) -> None:
        super().__init__(path, device)
    
        if 'signal_duration_s' in kwargs.keys():
            self.signal_duration = kwargs['signal_duration_s']
        else:
            self.signal_duration = 0.2

        self.load()

        # Используем новый парсер
        self.parser = SELDnetOutputParser(
            nb_classes=len(self.classes_list),
            hop_length_ms=self.params['label_hop_len_s'] * 1000,
            class_names=self.classes_list
        )

    def load(self):
        #TODO SIGNAL_LEN должен передаваться параметром
        SIGNAL_LEN = self.signal_duration

        DEVICE = self.device
        MODELPATH = self.files['model']
        AUDIOSCALERPATH = self.files['scaler']
        # self.params = dict()
        raw = {key: ast.literal_eval(val) for key, val in self.settings['params'].items()}
        self.params = load_params_with_optional_task(raw)
        self.model_type = str(self.params.get("model", "seldnet")).lower()
        self.n_mics = self.params['n_mics']
        self.n_pairs = self.n_mics*(self.n_mics-1)//2
        self.classes_list = self.params.get('classes_list', [])
        self.target_class = self.params.get('target_class', 0)
        self.fs = self.params['fs']

        N_CLASSES = len(self.classes_list)   
        USE_SALSALITE = self.params.get('use_salsalite',False)
        if USE_SALSALITE:
            self.n_feats = self.n_mics + (self.n_mics -1) #9=5 + 4 для 5 микрофонов - 4 пары с базовым.
            self.n_components = get_salsa_components_count(
                fs=self.fs,
                n_fft = NFFT, 
                fmin_doa = self.params.get('fmin_doa_salsalite', 50), 
                fmax_spectra = self.params.get('fmax_spectra_salsalite', 9000),
            ) # 415 по умолчанию
        else:
            self.n_components = self.params['nb_mel_bins'] # 64 по умолчанию
            self.n_feats = self.n_mics+self.n_pairs # 15=5+10 для 5 микрофонов

        
        N_TIMES = int(SIGNAL_LEN/self.params['label_hop_len_s']) # 10=0.2/0.02 по умолчанию 
        MAX_SIGNAL_LEN = self.params.get('max_audio_len_s',-1)

        if (MAX_SIGNAL_LEN>0) and (SIGNAL_LEN > MAX_SIGNAL_LEN):
            SIGNAL_LEN = np.float32(MAX_SIGNAL_LEN)
        state_dict = None
        try:
            # Загружаем веса модели (один раз) и валидируем, что это ожидаемая архитектура
            try:
                state_dict = pt.load(MODELPATH, map_location=DEVICE, weights_only=True)
            except TypeError:
                state_dict = pt.load(MODELPATH, map_location=DEVICE)
            if self.model_type == "ngccmodel":
                key = 'fnn_list.1.weight'
                if key not in state_dict:
                    raise ValueError(f'Модель {MODELPATH} не является NGCC-SELD моделью (нет ключа {key})')
            else:
                key = 'fnn_list.1.weight'
                if key not in state_dict:
                    raise ValueError(f'Модель {MODELPATH} не является SeldNet-моделью (нет ключа {key})')
            if state_dict[key].shape[0] % 12 != 0:
                raise ValueError(f'Модель {MODELPATH} имеет некорректную форму выхода ({key})')

            detected_n_classes = state_dict[key].shape[0] // 12
            if N_CLASSES == 0:
                logger.warning('В параметрах модели отсутствует список классов. Число классов определено автоматически.')
                N_CLASSES = detected_n_classes
                self.classes_list = list(range(N_CLASSES))
            elif detected_n_classes != N_CLASSES:
                raise ValueError(f'Число классов в модели {MODELPATH} не совпадает с указанным в конфгурации модели.')
            logger.info(f'Загружена SeldNet-модель с {N_CLASSES} классами')

        except Exception:
            logger.exception('Ошибка при загрузке весов модели')
            raise

        try:
            # Для ngccmodel вход подается "сырым" аудио-чанком и scaler не обязателен.
            self.scaler = None
            if self.model_type != "ngccmodel":
                self.scaler = joblib.load(AUDIOSCALERPATH)
            # Создаем модель
            data_in = (1, self.n_feats, N_TIMES, self.n_components) # по умолчанию (1,15,10,64) для сигнали длинной 0.2 сек
            data_out = (1, self.n_feats, 4*3*N_CLASSES) # по умолчанию (1,15, 12) для одного определяемого класса. 
            # logger.info(f'Форма входа:{data_in}; форма выхода:{data_out}')

            # В третьей компоненте выходных данных размера data_out содержатся по 4 параметра для каждого класса: 
            # координаты x, y, z внутри единичного шара определяющие направление на объект и расстояние d до объекта
            vid_data_in = None
            if self.model_type == "ngccmodel":
                self.model = NGCCModel(data_in, data_out, self.params, vid_data_in).to(DEVICE)
            else:
                self.model = SeldModel(data_in, data_out, self.params, vid_data_in).to(DEVICE)
            # Загружаем веса модели (используем уже загруженный state_dict)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
        except Exception:
            logger.exception('Ошибка инициализации модели SeldNet обработки аудио')
            raise

    def predict(self, S):
        '''
        Вычисляет направление на источник при помощи модели SELDnet
        S - сигнал

        Возвращает азимутальный угол относительно оси зрения видеокамеры, угол возвышения, относительно плоскости видеокамеры
        и класс объекта
        '''

        PROB_THRESHOLD = self.prob_threshold # Уровень обраружения БПЛА
        N_CLASSES = len(self.classes_list)   

        # Модель обучена на сигналах с дискретизацией, заданной параметром fs 
        if S.fs != self.fs:
            logger.warning(f'Частота дискретизации сигнала {S.fs} отличается от частоты модели SeldNet. Сигнал будет передискретизирован к частоте {self.fs}')
            S = S.resample(self.fs)

        # Вычисляем признаки
        if self.model_type == "ngccmodel":
            raw_chunks = self._get_raw_chunks(S)
            x = pt.from_numpy(raw_chunks.astype(np.float32)).to(self.device)
        else:
            feats = self._get_features(S) # [1, n_feats, n_components, n_times]
            x = pt.from_numpy(feats.astype(np.float32)).to(self.device)
            x = pt.permute(x, (0, 1, 3, 2)) # [1, n_feats, n_times, n_components]
        # Получаем предсказание модели
        with pt.no_grad():
            output = self.model(x)
        try:
            # парсер от Kimi
            prediction = self.parser.parse_with_tracking(
                output.detach().cpu().numpy(),
                threshold=self.prob_threshold
            )

            frames = self._extract_frames_for_target_class(
                prediction=prediction,
                target_class=self.target_class,
            )

            if len(frames)==0:
                print('-------')
                return {'full_prediction': prediction}

            print('\n-------')
            for frame in frames:
                print(f"{frame['frame_id']}: P={frame['prob']:.2f}, az={frame['azimuth']:.2f}, el={frame['elevation']:.2f}, class={frame['class']}")
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
                'full_prediction': prediction,  # Полная структура при необходимости
            }
        except Exception:
            logger.exception('Ошибка парсинга выхода SELDNet, используется резервный парсер')

            confidence_level, coords = get_multi_accdoa_predictions(
                accdoa_in=output.detach().cpu().numpy(), 
                nb_classes = N_CLASSES)
            # confidence_level [n_tracks, n_frames, n_class]
            # coords [n_tracks, n_frames, n_class, 4]
            probability, azimuth, elevation, distance = extract_prediction(confidence_level, coords, PROB_THRESHOLD, class_id = self.target_class)
            detected_class = self.classes_list[self.target_class]
            # print('\n'*3)
            # print('*'*30)
            # print(f'Строка - трек, столбец - класс')
            # print('='*30)
            # print(f'0-100 МС')
            # print('-'*30)
            # print(f'Уровни уверенности')
            # print(confidence_level[:,0,:].round(2))
            # print(f'Азимуты')
            # print(np.rad2deg(np.arctan2(coords[:,0,:,1], coords[:,0,:,0])).round(1))
            # print(f'Взвышения')
            # print(np.rad2deg(np.arctan2(coords[:,0,:,2], np.sqrt(coords[:,0,:,0]**2 + coords[:,0,:,1]**2))).round(1))
            # print('='*30)
            # print(f'100-200 МС')
            # print('-'*30)
            # print(f'Уровни уверенности')
            # print(confidence_level[:,1,:].round(2))
            # print(f'Азимуты')
            # print(np.rad2deg(np.arctan2(coords[:,1,:,1], coords[:,1,:,0])).round(1))
            # print(f'Взвышения')
            # print(np.rad2deg(np.arctan2(coords[:,1,:,2], np.sqrt(coords[:,1,:,0]**2 + coords[:,1,:,1]**2))).round(1))
            track_id = np.argmax(probability)
            return {
                'azimuth': azimuth[track_id],
                'elevation': elevation[track_id],
                'class':detected_class,
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
        Возвращает отсортированный список треков target_class для последнего кадра:
        [primary, secondary...].
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
            series.append(
                {
                    'track_id': int(event.track_id),
                    'azimuth': float(event.azimuth),
                    'elevation': float(event.elevation),
                    'distance': float(event.distance),
                    'prob': float(event.confidence),
                    'class': event.class_name,
                    'frame_id': frame_id,
                }
            )
        series.sort(key=lambda e: float(e['prob']), reverse=True)
        return series

    
    def _get_features(self, S):
        USE_SALSALITE = self.params.get('use_salsalite',False)
        USE_FSGCC = self.params.get('use_fsgcc', False)
        HOP_LEN = int(self.params['hop_len_s']*S.fs)
        WIN_LEN = 2*HOP_LEN
        NFFT = next_greater_power_of_2(WIN_LEN)
        NFFT = min(NFFT, S.n_samples)
        spect = seldnet_get_spectrogram_for_audio(
            S.data,
            hop_len = HOP_LEN, 
            win_len = WIN_LEN,
            n_fft = NFFT
        )
        
        if USE_SALSALITE:
            feat = get_salsalite_feats(
                linear_spectra = spect,
                n_fft=NFFT, 
                fmin_doa = self.params.get('fmin_doa_salsalite', 50), 
                fmax_doa = self.params.get('fmax_doa_salsalite', 2000), 
                fmax_spectra = self.params.get('fmax_spectra_salsalite', 9000),
                fs=S.fs,
                sound_speed= 343
            )
            self.n_components = 415 #FIXME Количество компонентов должно быть задано параметрически в конфиге или вычисляться.
        elif USE_FSGCC:
            feat = get_melfsgcc_feats(
                linear_spectra = spect, 
                n_mels = self.n_components, 
                fs = self.fs, 
                n_fft=NFFT, 
                window_size = self.params.get('fsgcc_window_size', 256), 
                hop_size=self.params.get('fsgcc_hop_size', 64),
                f_min_hz=self.params.get('f_min_hz', 0), 
                f_max_hz=self.params.get('f_max_hz', None), 
            )
        else:
            feat = get_melgcc_feats(spect, 
                self.n_components, 
                sr=float(self.fs),
                n_fft=NFFT,
                f_min_hz=self.params.get('f_min_hz', 0), 
                f_max_hz=self.params.get('f_max_hz', None), 
                )

        # Масштабируем вектор признаков
        scaled_feat = self.scaler.transform(feat)

        # Изменяем формат массива так, чтобы он подходил для входа модели
        scaled_feat = scaled_feat.transpose()
        scaled_feat = scaled_feat.reshape((self.n_feats, self.n_components, -1), order='C') # [n_feats, n_components, n_times]
        scaled_feat = np.expand_dims(scaled_feat, axis=0) # [1, n_feats, n_components, n_times]
        
        return scaled_feat

    def _get_raw_chunks(self, S):
        """
        Возвращает сырые окна аудио в формате [B, M, T, L], ожидаемом NGCCModel.
        """
        hop = int(self.params['hop_len_s'] * S.fs)
        hop = max(hop, 1)
        data = S.data
        n_samples, n_mics = data.shape
        # Для стабильной совместимости берем только полные окна и дополняем при нехватке.
        total = max(hop, (n_samples // hop) * hop)
        if n_samples < total:
            pad = np.zeros((total - n_samples, n_mics), dtype=data.dtype)
            data = np.vstack((data, pad))
        else:
            data = data[:total, :]
        n_frames = total // hop
        framed = data.reshape(n_frames, hop, n_mics)      # [T, L, M]
        framed = np.transpose(framed, (2, 0, 1))          # [M, T, L]
        framed = np.expand_dims(framed, axis=0)           # [1, M, T, L]
        return framed

        
