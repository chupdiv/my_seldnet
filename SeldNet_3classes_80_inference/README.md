# Модуль инференса для SELDNet_3classes_80

Этот каталог содержит независимый модуль для выполнения инференса модели SELDNet, обученной в конфигурации SeldNet_3classes_80.

## Файлы

- `seld_inference.py` — основной модуль с классом `SELDInference` и всеми необходимыми зависимостями
- `README.md` — этот файл

## Установка зависимостей

```bash
pip install torch numpy librosa joblib scikit-learn
```

## Использование

### Через Python API

```python
from seld_inference import SELDInference

# Создание объекта инференса
infer = SELDInference(
    weights_path='path/to/weights.pth',  # Путь к весам модели
    scaler_path='path/to/scaler.joblib',  # Путь к скалеру (опционально)
    task_id='6',  # ID задачи (по умолчанию '6')
    threshold=0.5  # Порог детекции
)

# Инференс для аудиофайла
detections = infer.infer('path/to/audio.wav')

# Вывод результатов
for det in detections:
    print(f"Frame {det['frame']}: Class {det['class']}, "
          f"Azimuth={det['azimuth']:.1f}°, Elevation={det['elevation']:.1f}°")

# Сохранение в CSV (формат DCASE)
csv_result = infer.infer_file('path/to/audio.wav', output_csv='results.csv')
```

### Через командную строку

```bash
python seld_inference.py \
    --weights path/to/weights.pth \
    --scaler path/to/scaler.joblib \
    --audio path/to/audio.wav \
    --output results.csv \
    --task-id 6 \
    --threshold 0.5
```

## Параметры задач

| task_id | dataset | use_salsalite | multi_accdoa | n_mics |
|---------|---------|---------------|--------------|--------|
| 2       | foa     | False         | False        | 4      |
| 3       | foa     | False         | True         | 4      |
| 4       | mic     | False         | False        | 4      |
| 5       | mic     | True          | False        | 4      |
| 6       | mic     | False         | True         | 4      |
| 7       | mic     | True          | True         | 4      |

## Формат выходных данных

Результат инференса — список словарей, каждый из которых содержит:

- `frame`: номер фрейма
- `class`: класс события (0, 1, 2)
- `track`: трек (0, 1, 2)
- `activity`: активность (0-1)
- `x`, `y`, `z`: декартовы координаты направления
- `dist`: расстояние (нормализованное)
- `azimuth`: азимут в градусах (-180...180)
- `elevation`: угол места в градусах (-90...90)

CSV файл имеет следующий формат:
```
frame,class,track,x,y,z,dist,azimuth,elevation
0,1,0,0.5,0.3,0.7,0.8,30.0,15.0
...
```

## Отличия от SeldNet_3classes_200

Модуль настроен на параметры SeldNet_3classes_80:
- Частота дискретизации: 250 kHz
- Hop length: 20 ms
- Label hop length: 40 ms
- Количество mel-бинов: 256
- Длительность клипа: 80 ms (2 label frame по 40 ms)

Для использования с моделью из SeldNet_3classes_200 необходимо изменить параметры в `DEFAULT_PARAMS_80`:
- fs=44100
- hop_len_s=0.02
- label_hop_len_s=0.1
- nb_mel_bins=64
- label_sequence_length=50
