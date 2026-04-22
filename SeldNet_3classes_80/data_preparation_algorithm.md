# Пошаговый алгоритм подготовки данных для обучения SeldNet (argv=6)

## Конфигурация для argv=6 (MIC + GCC + multi-ACCDOA)

### Ключевые временные параметры:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `fs` | 250000 Гц | Частота дискретизации |
| `hop_len_s` | 0.02 с (20 мс) | Шаг STFT (спектрального анализа) |
| `label_hop_len_s` | 0.04 с (40 мс) | Шаг меток (временное разрешение аннотаций) |
| `feature_label_resolution` | 2 | Сколько спектральных кадров в одном лейбл-фрейме |
| `label_sequence_length` | 5 | Количество лейбл-фреймов во входной последовательности сети |
| `feature_sequence_length` | 10 | Количество спектральных кадров во входной последовательности сети |
| `max_audio_len_s` | 15 с | Максимальная длина аудиофайла для обработки |

---

## Связь между параметрами времени

### Формулы расчёта:

```python
feature_label_resolution = label_hop_len_s // hop_len_s
                         = 0.04 // 0.02 = 2

feature_sequence_length = label_sequence_length × feature_label_resolution
                        = 5 × 2 = 10 спектральных кадров
```

### Временное соответствие:

- **1 спектральный кадр** = 20 мс аудио
- **1 лейбл-фрейм** = 40 мс аудио = 2 спектральных кадра
- **Вход сети** = 10 спектральных кадров = 200 мс аудио
- **Выход сети** = 5 лейбл-фреймов = 200 мс аудио

> **Важно:** Вход и выход покрывают одинаковый временной интервал (200 мс), но с разным временным разрешением!

### Расчёт для `max_audio_len_s=15` с:

- Количество лейбл-фреймов: `15 / 0.04 = 375` лейбл-фреймов
- Количество спектральных кадров: `15 / 0.02 = 750` спектральных кадров

---

## Пошаговый алгоритм подготовки данных

### Шаг 1: Подготовка исходного датасета

**Структура директорий:**

```
datasets/seld_data/reverb_fs250000Hz_4mics/
├── mic_dev/          # Аудиофайлы для обучения (4 микрофона)
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   └── fold4/
├── metadata_dev/     # CSV-файлы с аннотациями
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   └── fold4/
└── mic_eval/         # Аудиофайлы для валидации
    └── ...
```

**Требования к аудио:**
- Формат: WAV, многоканальный (4 канала для 4 микрофонов)
- Частота дискретизации: 250000 Гц
- Разрядность: 16-bit или 32-bit float

**Требования к метаданным (CSV):**

Формат: `frame, class_id, azimuth, elevation, distance, event_type`

- `frame` — номер фрейма (на основе `label_hop_len_s`)
- `class_id` — ID класса звука (0, 1, 2 для 3 классов)
- `azimuth` — азимут в градусах [0, 360]
- `elevation` — угол места в градусах [-90, 90]
- `distance` — расстояние в метрах
- `event_type` — тип события (onset/offset)

---

### Шаг 2: Извлечение признаков (`batch_feature_extraction.py`)

**Команда:**

```bash
cd /workspace/SeldNet_3classes_80
python batch_feature_extraction.py 6
```

**Процесс извлечения признаков:**

#### 2.1. Чтение аудиофайла

```python
audio_input, fs = wav.read(audio_path)  # shape: [nb_samples, 4]
```

#### 2.2. Расчёт количества фреймов

```python
nb_feat_frames = int(len(audio_input) / hop_len)      # hop_len = fs * hop_len_s = 5000 samples
nb_label_frames = int(len(audio_input) / label_hop)   # label_hop = fs * label_hop_len_s = 10000 samples
```

#### 2.3. Вычисление спектрограммы (STFT)

Для каждого из 4 каналов:

```python
stft_ch = librosa.core.stft(
    audio_input[:, ch],
    n_fft=2*hop_len,          # 10000 точек
    hop_length=hop_len,        # 5000 samples (20 мс)
    win_length=2*hop_len,      # окно Ханна
    window='hann'
)
# Результат: [nb_bins, nb_feat_frames]
```

#### 2.4. Извлечение Mel-спектрограммы

```python
mel_wts = librosa.filters.mel(
    sr=250000,
    n_fft=10000,
    n_mels=256,
    fmin=0.0,
    fmax=5000.0  # Важно: соответствует lowpass фильтру при генерации
).T

mag_spectra = np.abs(stft)**2
mel_spectra = np.dot(mag_spectra, mel_wts)
log_mel_spectra = librosa.power_to_db(mel_spectra)
```

#### 2.5. Вычисление GCC-признаков (для microphone dataset)

Для каждой пары микрофонов (всего 6 пар для 4 микрофонов):

```python
# GCC-PHAT между микрофонами m и n
R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
cc = np.fft.irfft(np.exp(1.j * np.angle(R)))  # Cross-correlation
cc = np.concatenate((cc[:, -nb_mel_bins//2:], cc[:, :nb_mel_bins//2]), axis=-1)
```

#### 2.6. Конкатенация признаков

```python
# Для MIC + GCC: Mel (256) + GCC (6 пар × 256) = 7 каналов по 256 признаков
feat = np.concatenate((mel_spect, gcc), axis=-1)
# Итоговая форма: [nb_feat_frames, 7 × 256] = [nb_feat_frames, 1792]
```

#### 2.7. Сохранение признаков

```python
np.save(feat_path, feat)  # shape: [nb_feat_frames, nb_features]
```

---

### Шаг 3: Нормализация признаков (`preprocess_features`)

#### 3.1. Вычисление статистик нормализации

```python
spec_scaler = preprocessing.StandardScaler()
for feat_file in os.listdir(feat_dir):
    feat = np.load(feat_file)
    spec_scaler.partial_fit(feat)  # Накопление mean и std
    
joblib.dump(spec_scaler, normalized_features_wts_file)
```

#### 3.2. Применение нормализации

```python
for feat_file in os.listdir(feat_dir):
    feat = np.load(feat_file)
    feat_normalized = spec_scaler.transform(feat)
    np.save(feat_dir_norm/filt_file, feat_normalized)
```

**Результат:** Нормализованные признаки имеют `mean=0`, `std=1` по каждому признаку

---

### Шаг 4: Извлечение меток (`extract_all_labels`)

#### 4.1. Чтение CSV-файла метаданных

```python
desc_file_polar = load_output_format_file(csv_path)
```

#### 4.2. Конвертация из полярных в декартовы координаты

```python
x = sin(elevation_rad) * cos(azimuth_rad)
y = sin(elevation_rad) * sin(azimuth_rad)
z = cos(elevation_rad)
```

#### 4.3. Создание матрицы меток (для multi-ACCDOA с ADPIT)

```python
# Форма: [nb_label_frames, 6, 5, nb_classes]
# 6 треков, 5 компонентов (act + x + y + z + dist), 3 класса
se_label = np.zeros((nb_label_frames, 6, nb_classes))
x_label = np.zeros((nb_label_frames, 6, nb_classes))
y_label = np.zeros((nb_label_frames, 6, nb_classes))
z_label = np.zeros((nb_label_frames, 6, nb_classes))
dist_label = np.zeros((nb_label_frames, 6, nb_classes))

# Заполнение активных событий по фреймам
for frame_ind, active_event_list in desc_file.items():
    if frame_ind < nb_label_frames:
        for active_event in active_event_list:
            se_label[frame_ind, track_idx, class_id] = 1
            x_label[frame_ind, track_idx, class_id] = active_event.x
            y_label[frame_ind, track_idx, class_id] = active_event.y
            z_label[frame_ind, track_idx, class_id] = active_event.z
            dist_label[frame_ind, track_idx, class_id] = active_event.distance / 100.0

label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)
```

#### 4.4. Сохранение меток

```python
np.save(label_path, label_mat)  # shape: [nb_label_frames, 6, 5, 3]
```

---

### Шаг 5: Подготовка данных для DataLoader (`cls_data_generator.py`)

#### 5.1. Загрузка и нарезка на последовательности

```python
# Загрузка признаков и меток
feat = np.load(feat_path)       # [nb_feat_frames, nb_features]
label = np.load(label_path)     # [nb_label_frames, 6, 5, 3]

# Обрезка до кратного размера
label = label[:label.shape[0] - (label.shape[0] % label_seq_len)]
feat = feat[:len(label) * feature_label_resolution, :]

# Нарезка на последовательности
# feat: [nb_seqs, label_seq_len, feature_label_resolution, nb_features]
# После reshape: [nb_seqs, feature_seq_len, nb_channels, nb_features_per_channel]
feat = feat.reshape((
    feat.shape[0] // feature_seq_len,
    feature_seq_len,           # 10
    nb_channels,               # 7
    nb_mel_bins                # 256
))

# label: [nb_seqs, label_seq_len, 6, 5, 3]
label = label.reshape((
    label.shape[0] // label_seq_len,
    label_seq_len,             # 5
    6, 5, 3
))
```

#### 5.2. Формирование батчей

```python
# Для training:
# feat_shape: (batch_size, 7, 10, 256)
# label_shape: (batch_size, 5, 6, 5, 3)

# Где:
# - batch_size = 128
# - 7 каналов (1 Mel + 6 GCC пар)
# - 10 = feature_sequence_length (200 мс)
# - 256 = nb_mel_bins
# - 5 = label_sequence_length (200 мс)
# - 6 = количество треков ADPIT
# - 5 = act + x + y + z + dist
# - 3 = количество классов
```

---

## Итоговая структура подготовленных данных

```
datasets/seld_data/reverb_fs250000Hz_4mics/
├── seld_feat_label_6/              # Извлечённые признаки и метки
│   ├── dev/
│   │   ├── fold1/
│   │   │   ├── *.npy (features)
│   │   │   └── *.npy (labels)
│   │   └── ...
│   └── eval/
│       └── ...
├── mic_dev/                        # Исходные WAV файлы
└── metadata_dev/                   # Исходные CSV метаданные
```

---

## Критические зависимости параметров

### 1. `hop_len_s` ↔ `label_hop_len_s`

```
label_hop_len_s ДОЛЖЕН быть кратен hop_len_s
feature_label_resolution = label_hop_len_s / hop_len_s = 2 (целое!)
```

### 2. `label_sequence_length` ↔ `feature_sequence_length`

```
feature_sequence_length = label_sequence_length × feature_label_resolution
5 × 2 = 10 кадров

Оба представляют ОДИН И ТОТ ЖЕ временной интервал:
- 5 лейбл-фреймов × 40 мс = 200 мс
- 10 спектральных кадров × 20 мс = 200 мс
```

### 3. `max_audio_len_s` и память

```
При max_audio_len_s = 15 с:
- Лейбл-фреймов: 375
- Спектральных кадров: 750
- Последовательностей для обучения: 375 / 5 = 75 на файл

Увеличение max_audio_len_s линейно увеличивает потребление памяти!
```

### 4. `t_pool_size` для архитектуры сети

```python
# Для argv=6 (не CST-former):
t_pool_size = [feature_label_resolution, 1, 1] = [2, 1, 1]

# Это означает pooling по 2 спектральным кадрам для 
# согласования с разрешением меток (40 мс)
```

---

## Проверка корректности подготовки

Запустите `smoke_check`:

```bash
python smoke_check_tasks_80.py 6
```

Ожидаемый вывод:

```
Task 6: label_seq=5, feat_seq=10, fl_res=2
```

Это подтверждает правильность связи:
- `label_sequence_length = 5`
- `feature_sequence_length = 10`
- `feature_label_resolution = 2`
