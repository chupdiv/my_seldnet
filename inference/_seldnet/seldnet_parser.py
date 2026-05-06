"""
SELDnet Output Parser
Парсит выходные данные модели SELDnet в структурированные предсказания
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import warnings


class OutputFormat(Enum):
    """Поддерживаемые форматы выхода SELDnet"""
    MULTI_ACCDOA = "multi_accdoa"  # Основной формат DCASE2024
    ACCDOA = "accdoa"              # Классический ACCDOA
    ADPIT = "adpit"                # Auxiliary Duplication PIT


@dataclass
class SELDEvent:
    """Одно событие/объект, обнаруженное моделью"""
    class_idx: int           # Индекс класса
    class_name: str          # Имя класса
    track_id: int            # ID трека (0-2 для multi-accdoa)
    azimuth: float           # Азимут в градусах [-180, 180]
    elevation: float         # Угол возвышения в градусах [-90, 90]
    distance: float          # Расстояние в метрах
    confidence: float        # Уровень уверенности [0, 1]
    x: float = 0.0          # Декартова координата X
    y: float = 0.0          # Декартова координата Y  
    z: float = 0.0          # Декартова координата Z
    
    def to_dict(self) -> Dict:
        return {
            'class_idx': self.class_idx,
            'class_name': self.class_name,
            'track_id': self.track_id,
            'azimuth': self.azimuth,
            'elevation': self.elevation,
            'distance': self.distance,
            'confidence': self.confidence,
            'cartesian': (self.x, self.y, self.z)
        }


@dataclass 
class SELDFrame:
    """Все события в одном временном фрейме"""
    frame_idx: int
    timestamp_ms: float
    events: List[SELDEvent] = field(default_factory=list)
    
    def get_active_events(self, threshold: float = 0.0) -> List[SELDEvent]:
        """Вернуть события с confidence > threshold"""
        return [e for e in self.events if e.confidence > threshold]
    
    def get_events_by_class(self, class_idx: int) -> List[SELDEvent]:
        """Фильтр по классу"""
        return [e for e in self.events if e.class_idx == class_idx]


@dataclass
class SELDPrediction:
    """Полное предсказание для аудио"""
    nb_frames: int
    hop_length_ms: float
    nb_classes: int
    frames: Dict[int, SELDFrame] = field(default_factory=dict)
    
    def get_all_events(self) -> List[SELDEvent]:
        """Все события из всех фреймов"""
        events = []
        for frame in self.frames.values():
            events.extend(frame.events)
        return events
    
    def get_events_timeline(self) -> Dict[int, List[Tuple[int, SELDEvent]]]:
        """Таймлайн событий по классам: {class_idx: [(frame_idx, event), ...]}"""
        timeline = {}
        for frame_idx, frame in self.frames.items():
            for event in frame.events:
                if event.class_idx not in timeline:
                    timeline[event.class_idx] = []
                timeline[event.class_idx].append((frame_idx, event))
        return timeline


class SELDnetOutputParser:
    """
    Парсер выходных данных SELDnet
    
    Поддерживает:
    - Multi-ACCDOA (3 трека, 4 параметра: x, y, z, distance)
    - Классический ACCDOA 
    - ADPIT формат
    """
    
    def __init__(
        self,
        nb_classes: int = 12,
        hop_length_ms: float = 100.0,
        output_format: OutputFormat = OutputFormat.MULTI_ACCDOA,
        class_names: Optional[List[str]] = None
    ):
        self.nb_classes = nb_classes
        self.hop_length_ms = hop_length_ms
        self.output_format = output_format
        self.class_names = class_names or [f"class_{i}" for i in range(nb_classes)]
        
        # Параметры multi-ACCDOA
        self.nb_tracks = 3  # Фиксировано для multi-ACCDOA
        self.params_per_track = 4  # x, y, z, distance
        
    def _validate_output_shape(self, output: np.ndarray) -> Tuple[int, int, int]:
        """Проверка и извлечение размерностей"""
        if output.ndim == 3:
            # [batch, frames, features] -> берём первый batch
            output = output[0]
        
        n_frames, n_features = output.shape
        
        expected_features = self.nb_tracks * self.params_per_track * self.nb_classes
        
        if n_features != expected_features:
            # Пробуем определить формат автоматически
            if n_features % 12 == 0:
                detected_classes = n_features // 12
                warnings.warn(
                    f"Ожидалось {expected_features} признаков, получено {n_features}. "
                    f"Автоопределение: {detected_classes} классов"
                )
                self.nb_classes = detected_classes
            else:
                raise ValueError(
                    f"Некорректная размерность выхода: {n_features}, "
                    f"ожидалось кратное 12"
                )
        
        return 1, n_frames, n_features  # batch=1 после squeeze
    
    def _extract_multi_accdoa(
        self, 
        output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение предсказаний из multi-ACCDOA формата
        
        Returns:
            confidence: [n_tracks, n_frames, n_classes]
            coords: [n_tracks, n_frames, n_classes, 4] (x, y, z, distance)
        """
        if output.ndim==3:
            output = output[0,...]
        n_frames = output.shape[0]
        
        # Разбираем выход по трекам
        # Формат: [track0_x, track0_y, track0_z, track0_d, 
        #          track1_x, track1_y, track1_z, track1_d,
        #          track2_x, track2_y, track2_z, track2_d] для каждого класса
        
        tracks = []
        for track_id in range(self.nb_tracks):
            start_idx = track_id * self.params_per_track * self.nb_classes
            
            x = output[:, start_idx + 0*self.nb_classes:start_idx + 1*self.nb_classes]
            y = output[:, start_idx + 1*self.nb_classes:start_idx + 2*self.nb_classes]
            z = output[:, start_idx + 2*self.nb_classes:start_idx + 3*self.nb_classes]
            d = output[:, start_idx + 3*self.nb_classes:start_idx + 4*self.nb_classes]
            
            # Нормализация направления (единичный вектор)
            norm = np.sqrt(x**2 + y**2 + z**2)
            confidence = np.clip(norm, 0, 1)  # Уверенность = длина вектора
            
            # Избегаем деления на ноль
            norm_safe = np.where(norm == 0, 1, norm)
            x_norm = x / norm_safe
            y_norm = y / norm_safe
            z_norm = z / norm_safe
            
            # Расстояние (ReLU активация подразумевает >= 0)
            distance = np.clip(d, 0, np.inf)
            
            track_coords = np.stack([x_norm, y_norm, z_norm, distance], axis=-1)
            tracks.append((confidence, track_coords))
        
        # Объединяем треки
        all_confidence = np.stack([t[0] for t in tracks], axis=0)  # [3, frames, classes]
        all_coords = np.stack([t[1] for t in tracks], axis=0)      # [3, frames, classes, 4]
        
        return all_confidence, all_coords
    
    @staticmethod
    def cartesian_to_spherical(
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразование декартовых координат в сферические
        
        Returns:
            azimuth: [-180, 180] градусов
            elevation: [-90, 90] градусов
        """
        # Азимут: arctan2(y, x)
        azimuth = np.rad2deg(np.arctan2(y, x))
        
        # Угол возвышения: arcsin(z / r), но z уже нормализовано
        # Для единичного вектора: elevation = arcsin(z)
        elevation = np.rad2deg(np.arcsin(np.clip(z, -1, 1)))
        
        return azimuth, elevation
    
    def parse(
        self,
        model_output: Union[np.ndarray, 'torch.Tensor'],
        threshold: float = 0.3,
        apply_per_class_threshold: bool = False
    ) -> SELDPrediction:
        """
        Основной метод парсинга выхода модели
        
        Args:
            model_output: Выход модели [batch, frames, features] или [frames, features]
            threshold: Порог уверенности для фильтрации событий
            apply_per_class_threshold: Использовать разные пороги для разных классов
            
        Returns:
            SELDPrediction со структурированными данными
        """
        
        # Конвертация из torch если нужно
        if hasattr(model_output, 'detach'):
            model_output = model_output.detach().cpu().numpy()
        
        # Валидация формы
        batch, n_frames, _ = self._validate_output_shape(model_output)
        
        # Извлечение данных в зависимости от формата
        if self.output_format == OutputFormat.MULTI_ACCDOA:
            confidence, coords = self._extract_multi_accdoa(model_output)
        else:
            raise NotImplementedError(f"Формат {self.output_format} не реализован")
        
        # Парсинг по фреймам
        frames = {}
        
        for frame_idx in range(n_frames):
            frame_events = []
            timestamp_ms = frame_idx * self.hop_length_ms
            
            for track_id in range(self.nb_tracks):
                for class_idx in range(self.nb_classes):
                    conf = confidence[track_id, frame_idx, class_idx]
                    
                    # Пропускаем если уверенность ниже порога
                    if conf < threshold:
                        continue
                    
                    # Извлекаем координаты
                    x, y, z, distance = coords[track_id, frame_idx, class_idx]
                    
                    # Преобразование в сферические координаты
                    azimuth, elevation = self.cartesian_to_spherical(x, y, z)
                    
                    event = SELDEvent(
                        class_idx=class_idx,
                        class_name=self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}",
                        track_id=track_id,
                        azimuth=float(azimuth),
                        elevation=float(elevation),
                        distance=float(distance),
                        confidence=float(conf),
                        x=float(x),
                        y=float(y),
                        z=float(z)
                    )
                    frame_events.append(event)
            
            frames[frame_idx] = SELDFrame(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                events=frame_events
            )
        
        return SELDPrediction(
            nb_frames=n_frames,
            hop_length_ms=self.hop_length_ms,
            nb_classes=self.nb_classes,
            frames=frames
        )
    
    def parse_with_tracking(
        self,
        model_output: Union[np.ndarray, 'torch.Tensor'],
        threshold: float = 0.3,
        max_distance: float = 30.0  # максимальное расстояние для валидации
    ) -> SELDPrediction:
        """
        Парсинг с дополнительной фильтрацией и валидацией
        
        Args:
            max_distance: Максимально допустимое расстояние (фильтр выбросов)
        """
        prediction = self.parse(model_output, threshold)
        
        # Фильтрация выбросов по расстоянию
        for frame in prediction.frames.values():
            frame.events = [
                e for e in frame.events 
                if 0 <= e.distance <= max_distance and
                -180 <= e.azimuth <= 180 and
                -90 <= e.elevation <= 90
            ]
        
        return prediction
    
    def get_dominant_prediction(
        self,
        model_output: Union[np.ndarray, 'torch.Tensor'],
        target_class: Optional[int] = None
    ) -> Optional[SELDEvent]:
        """
        Получить доминирующее предсказание (с максимальной уверенностью)
        
        Если target_class указан — ищем максимум только в этом классе
        """
        prediction = self.parse(model_output, threshold=0.0)
        
        all_events = prediction.get_all_events()
        if not all_events:
            return None
        
        if target_class is not None:
            class_events = [e for e in all_events if e.class_idx == target_class]
            if not class_events:
                return None
            return max(class_events, key=lambda e: e.confidence)
        
        return max(all_events, key=lambda e: e.confidence)


# Утилиты для работы с предсказаниями

def merge_predictions(
    predictions: List[SELDPrediction],
    overlap_ms: float = 50.0
) -> SELDPrediction:
    """
    Объединение перекрывающихся предсказаний (для скользящего окна)
    """
    # TODO: Реализация слияния с учётом трекинга
    pass


def prediction_to_json(prediction: SELDPrediction) -> Dict:
    """Сериализация в JSON-friendly формат"""
    return {
        'nb_frames': prediction.nb_frames,
        'hop_length_ms': prediction.hop_length_ms,
        'nb_classes': prediction.nb_classes,
        'frames': {
            str(idx): {
                'timestamp_ms': frame.timestamp_ms,
                'events': [e.to_dict() for e in frame.events]
            }
            for idx, frame in prediction.frames.items()
        }
    }


# Пример использования и тестирование
if __name__ == "__main__":
    # Тестовые данные
    nb_classes = 2
    n_frames = 10
    
    # Создаём тестовый выход модели
    # [frames, 4*3*nb_classes] = [10, 24]
    test_output = np.random.randn(n_frames, 4 * 3 * nb_classes).astype(np.float32)
    # Нормализуем для реалистичности
    test_output = np.tanh(test_output) * 0.5 + 0.5
    
    # Парсинг
    parser = SELDnetOutputParser(
        nb_classes=nb_classes,
        hop_length_ms=100.0,
        class_names=['car', 'distraction']
    )
    
    prediction = parser.parse(test_output, threshold=0.3)
    
    print(f"Всего фреймов: {prediction.nb_frames}")
    print(f"Всего событий: {len(prediction.get_all_events())}")
    
    # Вывод первого фрейма с событиями
    for frame_idx, frame in prediction.frames.items():
        if frame.events:
            print(f"\nFrame {frame_idx} (t={frame.timestamp_ms}ms):")
            for event in frame.events:
                print(f"  {event.class_name} (track {event.track_id}): "
                      f"azi={event.azimuth:.1f}°, ele={event.elevation:.1f}°, "
                      f"dist={event.distance:.2f}m, conf={event.confidence:.2f}")