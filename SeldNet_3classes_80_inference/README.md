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

#### Вариант 1: С явной передачей параметров

```python
from seld_inference import SELDInference

params = {
    'fs': 250000,
    'hop_len_s': 0.02,
    'label_hop_len_s': 0.04,
    'nb_mel_bins': 256,
    'mel_fmax_hz': 5000.0,
    'n_mics': 4,
    'classes_list': ['класс_0', 'класс_1', 'класс_2'],  # список классов
    'target_class': 0,  # целевой класс (индекс в classes_list)
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

# unique_classes будет вычислен автоматически как len(classes_list) = 3
infer = SELDInference(
    weights_path='path/to/weights.pth',
    params=params,
    task_id='6'
)

# Инференс для аудиофайла
detections = infer.infer('path/to/audio.wav')

# Вывод результатов
for det in detections:
    print(f"Frame {det['frame']}: Class {det['class']} ({params['classes_list'][det['class']]}), "
          f"Azimuth={det['azimuth']:.1f}°, Elevation={det['elevation']:.1f}°")

# Сохранение в CSV (формат DCASE)
csv_result = infer.infer_file('path/to/audio.wav', output_csv='results.csv')
```

#### Вариант 2: С использованием параметров по умолчанию

```python
from seld_inference import SELDInference

# Параметры по умолчанию уже содержат classes_list=['class_0', 'class_1', 'class_2']
infer = SELDInference(
    weights_path='path/to/weights.pth',
    task_id='6'
)

detections = infer.infer('path/to/audio.wav')
```

### Через командную строку

```bash
# Базовый вариант
python seld_inference.py \
    --weights path/to/weights.pth \
    --scaler path/to/scaler.joblib \
    --audio path/to/audio.wav \
    --output results.csv \
    --task-id 6 \
    --threshold 0.5

# С загрузкой параметров из JSON файла
python seld_inference.py \
    --weights path/to/weights.pth \
    --audio path/to/audio.wav \
    --params my_params.json \
    --task-id 6

# Просмотр требуемых параметров для task_id
python seld_inference.py --list-params --task-id 6
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

## Параметры модели

### classes_list и target_class

- **`classes_list`**: Список названий классов (например, `['шум', 'речь', 'музыка']`). 
  - Длина списка автоматически определяет параметр `unique_classes`.
  - Индексы классов в результатах инференса соответствуют индексам в этом списке.
  
- **`target_class`**: Индекс целевого класса в `classes_list` для фильтрации результатов (опционально).
  - Если требуется обработка всех классов, можно игнорировать этот параметр.

Пример:
```python
params = {
    'classes_list': ['шум', 'речь', 'музыка'],  # 3 класса
    'target_class': 1,  # целевой класс - 'речь' (индекс 1)
    ...
}
# unique_classes будет вычислен как 3
```

## Формат выходных данных

Результат инференса — список словарей, каждый из которых содержит:

- `frame`: номер фрейма
- `class`: индекс класса события (соответствует индексу в `classes_list`)
- `track`: трек (0, 1, 2)
- `activity`: активность (0-1)
- `x`, `y`, `z`: декартовы координаты направления
- `dist`: расстояние (нормализованное)
- `azimuth`: азимут в градусах (-180...180)
- `elevation`: угол места в градусах (-90...90)

Для получения названия класса из результата:
```python
class_name = params['classes_list'][det['class']]
```

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
