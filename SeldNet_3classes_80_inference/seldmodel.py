from __future__ import annotations
from typing import Any, Dict
import numpy as np
import ast

from ._seldmodel.inference import SELDInference
from ..protomodel import ProtoModel

class SELDModel(ProtoModel):
    """
    Обертка над SELDInference
    """

    def __init__(self, path:str, device:str='cpu', **kwargs) -> None:
        super().__init__(path, device)
    
        if 'signal_duration_s' in kwargs.keys():
            self.signal_duration = kwargs['signal_duration_s']
        else:
            self.signal_duration = 0.2

        self.load()

    def load(self):
        DEVICE = self.device
        MODEL_PATH = self.files.get('model')
        AUDIOSCALERPATH = self.files.get('scaler',None)
        TASK_ID = self.tags.get('kind',None)
        params_dict = {key: ast.literal_eval(val) for key, val in self.settings['params'].items()}
        try:
            self.seld_inference = SELDInference(
                weights_path=MODEL_PATH,
                params=params_dict,
                task_id=TASK_ID,
                scaler_path=AUDIOSCALERPATH,
                device=DEVICE
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке SELD модели: {e}")

        self.classes_list = self.params.get('classes_list', [])
        unique_classes = self.params.get('unique_classes', None)
        if (len(self.classes_list) == 0) and (unique_classes is not None):
            self.classes_list = [f'Класс {i}' for i in range(int(unique_classes))]
        self.target_class = self.params.get('target_class', 0)
        

    def predict(self, S) -> Dict[str, Any]:
        """
        Выполняет инференс для входного сигнала.
        
        Args:
            S: Объект сигнала. Ожидается, что у него есть атрибут .data 
               формы (n_samples, n_channels).
        
        Returns:
            Словарь с результатами:
            {
                'azimuth': float (градусы),
                'elevation': float (градусы),
                'distance': float (метры),
                'class': str,
                'prob': float
                'full_prediction': list[dict],

            }
        """
        if self.seld_inference is None:
            raise RuntimeError("Модель не загружена. Вызовите load().")

        detections = self.seld_inference.infer(S.data, threshold=self.prob_threshold)
        # detections = [{
        #   'frame': int, 'class': int, 'track': int, 'activity': float, 
        #   'x': float, 'y': float, 'z': float, 'dist': float, 'azimuth': float, 'elevation': float}, ...]
        detected_class = self.classes_list[self.target_class]

        # Агрегация результатов
        # Если детекций несколько, можно взять наиболее вероятную или усреднить координаты активного класса
        if not detections:
            # Возвращаем дефолтное значение, если ничего не найдено
            return {
                'azimuth': 0.0,
                'elevation': 0.0,
                'distance': 1.0,
                'class': detected_class,
                'prob': 0.0,
                'full_prediction':[],

            }

       # Получаем доминирующее предсказание для target_class
        class_detections = [ el for el in detections if (el['class']==self.target_class)]
        best_detection = max(class_detections, key=lambda x: x.get('prob', 0.0))
        
        
        if distance < 1e-6:
            distance = 0.0

        return {
            'azimuth': np.float32(best_detection['azimuth']),
            'elevation': np.float32(best_detection['elevation']),
            'distance': np.float32(best_detection['dist']),
            'class': best_detection.get('class_name', 'unknown'),
            'prob': np.float32(best_detection.get('prob', 0.0)),
            'full_prediction': detections,
        }
