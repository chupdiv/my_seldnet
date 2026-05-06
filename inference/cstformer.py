import os
import numpy as np
import joblib
import torch as pt
import json
import ast

from acoustic.doa.doa_tools import cartesian_to_spherical
from acoustic import Acoustic

from core.logger import logger

from ..protomodel import ProtoModel

from ._seldnet.seldnet_model import SeldModel
from ._seldnet.tools import (
    seldnet_get_spectrogram_for_audio,
    get_multi_accdoa_predictions,
    extract_prediction,
    get_melgcc_feats,
    get_salsalite_feats,
    get_salsa_components_count,
    CLASSES_LIST,
    NFFT,
)
from ._seldnet.ngcc_seld_params import load_params_with_optional_task

from ._cstformer.CST_former_model import CST_former

EPS = 1e-10


class CSTformer(ProtoModel):
    def __init__(self, path:str, device:str='cpu',**kwargs) -> None:
        super().__init__(path, device)

        if 'signal_duration_s' in kwargs.keys():
            self.signal_duration = kwargs['signal_duration_s']
        else:
            self.signal_duration = 0.2

        self.load()
    def load(self):
        #TODO SIGNAL_LEN должен передаваться параметром
        SIGNAL_LEN = self.signal_duration
        DEVICE = self.device
        MODELPATH = self.files['model']
        AUDIOSCALERPATH = self.files['scaler']
        # self.params = dict()
        # for key,val in self.settings['params'].items():
        #     self.params[key] = ast.literal_eval(val)
        #     print(f'{key}->{self.params[key]}')
        raw = {key: ast.literal_eval(val) for key, val in self.settings['params'].items()}
        self.params = load_params_with_optional_task(raw)
        self.use_ngcc = bool(self.params.get("use_ngcc", False))
        self.n_mics = self.params['n_mics']
        self.n_pairs = self.n_mics*(self.n_mics-1)//2
        self.classes_list = self.params.get('classes_list', CLASSES_LIST)
        self.target_class = self.params.get('target_class', 0)
        self.fs = self.params['fs']

        BUFFER_LEN = int(5 * self.fs)
        self.audio_buffer = Acoustic(
            data = np.zeros((BUFFER_LEN, self.n_mics), dtype=np.float32),
            fs = self.fs,
            timestamp=None
        ) 

        N_CLASSES = len(self.classes_list) if self.classes_list else int(self.params.get("unique_classes", 13))
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
            # nb_channels из ngcc-seld (task 33: 10) должен совпадать с числом каналов признаков
            self.n_feats = int(self.params.get('nb_channels', self.n_mics + self.n_pairs))

        
        # N_TIMES = int(SIGNAL_LEN/self.params['label_hop_len_s']) # 10=0.2/0.02 по умолчанию 
        N_TIMES = int(SIGNAL_LEN/self.params['hop_len_s']) # 10=0.2/0.02 по умолчанию 

        MAX_SIGNAL_LEN = self.params.get('max_audio_len_s',-1)

        if (MAX_SIGNAL_LEN>0) and (SIGNAL_LEN > MAX_SIGNAL_LEN):
            SIGNAL_LEN = np.float32(MAX_SIGNAL_LEN)
        try:
            self.scaler = None
            if not self.use_ngcc:
                self.scaler = joblib.load(AUDIOSCALERPATH)

            data_in = (1, self.n_feats, N_TIMES, self.n_components) # по умолчанию (1,15,10,64) для сигнала длинной 0.2 сек

            data_out = (1, self.n_feats, 4*3*N_CLASSES) # по умолчанию (1,15, 12) для одного определяемого класса. 
            # print(f'Форма входа:{data_in}; форма выхода:{data_out}')

            # В третьей компоненте выходных данных размера data_out содержатся по 4 параметра для каждого класса: 
            # координаты x, y, z внутри единичного шара определяющие направление на объект и расстояние d до объекта
            vid_data_in = None

            #!!!! data_in передается, но не используется
            self.model = CST_former(data_in, data_out, self.params, vid_data_in).to(DEVICE)

            try:
                state_dict = pt.load(MODELPATH, map_location=DEVICE, weights_only=True)
            except TypeError:
                state_dict = pt.load(MODELPATH, map_location=DEVICE)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        except Exception:
            logger.error('Ошибка инициализации модели CST-Former обработки аудио')
            self.model = None
            raise

    def predict(self, S):
        '''
        Вычисляет направление на источник при помощи модели SELDnet
        S - сигнал

        Возвращает азимутальный угол относительно оси зрения видеокамеры, угол возвышения, относительно плоскости видеокамеры
        и класс объекта
        '''

        PROB_THRESHOLD = self.prob_threshold # Уровень обраружения БПЛА
        N_CLASSES = len(self.classes_list) if self.classes_list else int(self.params.get("unique_classes", 13))
        TARGET_CLASS = self.target_class

        # Модель обучена на сигналах с дискретизацией, заданной параметром fs 
        if S.fs != self.fs:
            logger.warning(f'Частота дискретизации сигнала {S.fs} отличается от частоты модели SeldNet. Сигнал будет передискретизирован к частоте {self.fs}')
            S = S.resample(self.fs)

        self.audio_buffer.data = np.roll(
            self.audio_buffer.data, 
            shift = -S.n_samples,
            axis = 0
        )
        self.audio_buffer.data[-S.n_samples:, :] = S.data

        # Вычисляем признаки
        if self.use_ngcc:
            raw_chunks = self._get_raw_chunks(self.audio_buffer)
            x = pt.from_numpy(raw_chunks.astype(np.float32)).to(self.device)
        else:
            feats = self._get_features(self.audio_buffer) # [1, n_feats, n_components, n_times]
            x = pt.from_numpy(feats.astype(np.float32)).to(self.device)
            x = pt.permute(x, (0, 1, 3, 2)) # [1, n_feats, n_times, n_components]
        # Получаем предсказание модели
        try:
            with pt.no_grad():
                output = self.model(x)
                tail = int(self.params.get('cst_inference_last_frames', 2))
                tail = min(tail, output.shape[1])
                output = output[:, -tail:, :]
        except Exception as e:
            logger.error(f'Ошибка при обращении к модели CST-Former: {e}')
            return {}

        confidence_level, coords = get_multi_accdoa_predictions(
            accdoa_in=output.detach().cpu().numpy(), 
            nb_classes = N_CLASSES)
        probability, azimuth, elevation, distance = extract_prediction(
            confidence_level, coords, PROB_THRESHOLD, class_id=TARGET_CLASS)
        detected_class = self.classes_list[TARGET_CLASS]
        track_id = np.argmax(probability)
        return {
            'azimuth': azimuth[track_id],
            'elevation': elevation[track_id],
            'class':detected_class,
            'prob': probability[track_id],
            'distance': distance[track_id],
        }    
    
    def _get_features(self, S):
        USE_SALSALITE = self.params.get('use_salsalite',False)
        HOP_LEN = int(self.params['hop_len_s']*S.fs)
        
        spect = seldnet_get_spectrogram_for_audio(
            S.data,
            hop_len = HOP_LEN,
            win_len = 2*HOP_LEN,
            n_fft = NFFT)
        
        if USE_SALSALITE:
            feat = get_salsalite_feats(
                linear_spectra = spect,
                n_fft = NFFT, 
                fmin_doa = self.params.get('fmin_doa_salsalite', 50), 
                fmax_doa = self.params.get('fmax_doa_salsalite', 2000), 
                fmax_spectra = self.params.get('fmax_spectra_salsalite', 9000),
                fs=S.fs,
                sound_speed= 343
            )
            self.n_components = 415 #FIXME Количество компонентов должно быть задано параметрически в конфиге или вычисляться.
        else:
            feat = get_melgcc_feats(spect, self.n_components, sr=float(self.fs))

        # Масштабируем вектор признаков
        scaled_feat = self.scaler.transform(feat)

        # Изменяем формат массива так, чтобы он подходил для входа модели
        scaled_feat = scaled_feat.transpose()
        scaled_feat = scaled_feat.reshape((self.n_feats, self.n_components, -1), order='C') # [n_feats, n_components, n_times]
        scaled_feat = np.expand_dims(scaled_feat, axis=0) # [1, n_feats, n_components, n_times]
        
        return scaled_feat

    def _get_raw_chunks(self, S):
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
        framed = data.reshape(n_frames, hop, n_mics)
        framed = np.transpose(framed, (2, 0, 1))
        return np.expand_dims(framed, axis=0)

