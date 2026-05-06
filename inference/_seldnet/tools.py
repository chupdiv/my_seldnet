    # Параметры для модели SELDnet
import numpy as np
import librosa
import warnings

import acoustic 

NB_MEL_BINS = 64
HOP_LEN = 882 # 0.02 cек
WIN_LEN = 1764 # 2*HOP_LENGTH
NFFT = 2048

# Имена классов по умолчанию, если не заданы в model.conf [params] classes_list
CLASSES_LIST = ["Drone", "Wind"]

def next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


def seldnet_get_spectrogram_for_audio(audio_in, hop_len, win_len, n_fft):
    n_feat_frames = int(len(audio_in) / float(hop_len))
    n_channels = audio_in.shape[1]
    spectra = []
    for ch_cnt in range(n_channels):
        stft_ch = librosa.core.stft(
            np.asfortranarray(audio_in[:, ch_cnt]), 
            n_fft=n_fft, 
            hop_length=hop_len,
            win_length=win_len, 
            window='hann',
        )
        spectra.append(stft_ch[:, :n_feat_frames])
    return np.array(spectra).T



def seldnet_get_fsgcc(linear_spectra, nb_mel_bins, 
                    fs:float = 44100., 
                    window:str = "hann", 
                    window_size:int = 256,  
                    hop_size:int = 32, ):

    # linear_spectra [n_times, n_fft, n_channels]
    n_times = linear_spectra.shape[0]
    n_mics = linear_spectra.shape[-1]

    pairs = acoustic.doa.doa_tools.all_pairs(n_mics)[:,::-1]
    _, ar_fsgcc, tpwin = acoustic.doa.gcc.fsgcc_matrix_for_spec(
            spec = linear_spectra.transpose([1,0,2]), # [n_fft, n_times, n_channels]
            fs = fs,
            pairs = pairs, # Берем все пары
            window = window,
            window_size = window_size,
            hop_size = hop_size,
        )
    _, gcc_feat = acoustic.doa.gcc._fsgcc_wsvd_core(
        ar_fsgcc = ar_fsgcc,  # [n_time, n_bands, n_lags, n_pairs] 
        tpwin = tpwin,
        fs = fs,
        pt_max = nb_mel_bins//2,
        normalize = False,
        argmax = False,
        use_weights = False,
    )
    # gcc_feat # [n_time, n_lags, n_pairs]
    gcc_feat = gcc_feat[:, :nb_mel_bins, :] # (n_times, nb_mel_bins, gcc_channels)
    gcc_feat = gcc_feat.transpose((0, 2, 1)).reshape((n_times, -1))
    return gcc_feat

def seldnet_get_gcc(linear_spectra, nb_mel_bins):
    n_times = linear_spectra.shape[0]
    n_mics = linear_spectra.shape[-1]
    gcc_channels = int(n_mics*(n_mics-1)/2)    # Число неупорядоченных пар микрофонов
    gcc_feat = np.zeros((n_times, nb_mel_bins, gcc_channels))
    cnt = 0
    for m in range(n_mics):
        for n in range(m+1, n_mics):
            R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
            cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
            cc = np.concatenate((cc[:, -nb_mel_bins//2:], cc[:, :nb_mel_bins//2]), axis=-1)
            gcc_feat[:, :, cnt] = cc
            cnt += 1
    return gcc_feat.transpose((0, 2, 1)).reshape((n_times, -1))

def seldnet_get_mel_spectrogram(linear_spectra, nb_mel_bins, sr: float, n_fft: int = NFFT, f_min_hz = None, f_max_hz = None):
    if f_min_hz is None :
        f_min_hz = 0.
    n_times = linear_spectra.shape[0]
    n_mics = linear_spectra.shape[-1]
    mel_wts = librosa.filters.mel(
        sr=sr, 
        n_fft=n_fft, 
        n_mels=nb_mel_bins,
        fmin = f_min_hz,
        fmax = f_max_hz,
    ).T
    mel_feat = np.zeros((n_times, nb_mel_bins, n_mics))
    for ch_cnt in range(n_mics):
        mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
        mel_spectra = np.dot(mag_spectra, mel_wts)
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        mel_feat[:, :, ch_cnt] = log_mel_spectra
    mel_feat = mel_feat.transpose((0, 2, 1)).reshape((n_times, -1))
    return mel_feat


def get_melgcc_feats(linear_spectra, n_mels, sr: float, n_fft: int = NFFT,f_min_hz = None, f_max_hz = None):
    mel_spect = seldnet_get_mel_spectrogram(linear_spectra, 
        nb_mel_bins=n_mels, 
        sr=sr, 
        n_fft=n_fft,
        f_min_hz = f_min_hz, 
        f_max_hz = f_max_hz
    )
    gcc = seldnet_get_gcc(linear_spectra, nb_mel_bins=n_mels)
    return np.concatenate((mel_spect, gcc), axis=-1)

def get_melfsgcc_feats(linear_spectra, n_mels, fs, window_size, hop_size, n_fft: int = NFFT, f_min_hz = None, f_max_hz = None):
    mel_spect = seldnet_get_mel_spectrogram(linear_spectra, nb_mel_bins=n_mels, sr=fs, n_fft=n_fft, f_min_hz = f_min_hz, f_max_hz = f_max_hz)
    gcc = seldnet_get_fsgcc(linear_spectra, nb_mel_bins=n_mels, fs=fs, window_size=window_size,hop_size=hop_size)
    return np.concatenate((mel_spect, gcc), axis=-1)

def get_salsalite_feats(linear_spectra, fs, n_fft, fmin_doa, fmax_doa, fmax_spectra, sound_speed):
    # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
    # spatial features
    fs = np.float32(fs)

    LOWER_BIN = int(np.floor(fmin_doa * n_fft / float(fs)))
    LOWER_BIN = np.max((1, LOWER_BIN))
    UPPER_BIN = int(np.floor(np.min((fmax_doa, fs//2)) * n_fft / fs))
    CUTOFF_BIN = int(np.floor(fmax_spectra * n_fft / fs))
    # assert UPPER_BIN <= CUTOFF_BIN, 'Upper bin for doa featurei {} is higher than cutoff bin for spectrogram {}!'.format()

    DELTA = 2 * np.pi * fs / (n_fft * sound_speed)
    FREQ_VECTOR = np.arange(n_fft//2 + 1)
    FREQ_VECTOR[0] = 1
    FREQ_VECTOR = FREQ_VECTOR[None, :, None]  # 1 x n_bins x 1

    phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
    phase_vector = phase_vector / (DELTA * FREQ_VECTOR)
    phase_vector = phase_vector[:, LOWER_BIN : CUTOFF_BIN, :]
    phase_vector[:, UPPER_BIN:, :] = 0
    
    # spectral features
    linear_spectra = np.abs(linear_spectra)**2
    for ch_cnt in range(linear_spectra.shape[-1]):
        linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
    linear_spectra = linear_spectra[:, LOWER_BIN:CUTOFF_BIN, :]

    phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))
    linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    return np.concatenate((linear_spectra, phase_vector), axis=-1)

def get_salsa_components_count(fs, n_fft, fmin_doa, fmax_spectra):
    LOWER_BIN = int(np.floor(fmin_doa * n_fft / float(fs)))
    LOWER_BIN = np.max((1, LOWER_BIN))
    CUTOFF_BIN = int(np.floor(fmax_spectra * n_fft / fs))
    return  CUTOFF_BIN-LOWER_BIN



def get_multi_accdoa_predictions(accdoa_in, nb_classes:int):
    '''
    Вспомогательная функция для разбора инференса модели SELDnet
    accdoa_in - [batch_size, frames, num_track*num_axis*num_class=3*3*12]
    nb_classes - количество классов
    
    Возвращает:
    confidence_level - [n_tracks, n_frames, n_class]
    coords - [n_tracks, n_frames, n_class, 4]
    '''
    # batch_size для инференса = 1.
    # Трек 0
    x0 = accdoa_in[0, :, 0*nb_classes:1*nb_classes]
    y0 = accdoa_in[0, :, 1*nb_classes:2*nb_classes] 
    z0 = accdoa_in[0, :, 2*nb_classes:3*nb_classes]
    d0 = accdoa_in[0, :, 3*nb_classes:4*nb_classes]
    # Трек 1
    x1 = accdoa_in[0, :, 4*nb_classes:5*nb_classes]
    y1 = accdoa_in[0, :, 5*nb_classes:6*nb_classes] 
    z1 = accdoa_in[0, :, 6*nb_classes:7*nb_classes]
    d1 = accdoa_in[0, :, 7*nb_classes:8*nb_classes]
    # Трек 2
    x2 = accdoa_in[0, :, 8*nb_classes:9*nb_classes]
    y2 = accdoa_in[0, :, 9*nb_classes:10*nb_classes] 
    z2 = accdoa_in[0, :, 10*nb_classes:11*nb_classes]
    d2 = accdoa_in[0, :, 11*nb_classes:12*nb_classes]

    # Декартовы координаты accdoa-оценки
    x = np.stack([x0, x1, x2], axis = 0) # [n_tracks, n_frames, n_class]
    y = np.stack([y0, y1, y2], axis = 0) # [n_tracks, n_frames, n_class]
    z = np.stack([z0, z1, z2], axis = 0) # [n_tracks, n_frames, n_class]
    l = np.sqrt(x**2 + y**2 + z**2) # [n_tracks, n_frames, n_class]


    distance = np.stack([d0, d1, d2], axis = 0) # [n_tracks, n_frames, n_class]
    distance = np.clip(distance, a_min = 0., a_max = np.inf)

    # Единичный вектор предсказания или нулевой вектор
    x = x / (l + 1e-12)# [n_tracks, n_frames, n_class]
    y = y / (l + 1e-12) # [n_tracks, n_frames, n_class]
    z = z / (l + 1e-12) # [n_tracks, n_frames, n_class]

    # # Сферические координаты предсказания
    # l_xy = np.sqrt(x**2 + y**2) 
    # azimuth = np.arctan2(y,x)
    # elevation = np.arctan2(l_xy, z)

    # уровень уверенности модели
    confidence_level = np.clip(l, a_min = 0., a_max=1.)
    coords = np.stack([x,y,z,distance], axis = 3)    
    return confidence_level, coords


def extract_prediction(confidence_level, coords, treshold, class_id = 0):
    local_coords = coords.copy()
    sed = confidence_level > treshold
    sed = np.stack([sed]*local_coords.shape[-1], axis= 3)
    if np.max(sed) != 0:
        local_coords[~sed] = np.nan
    
    # Центроид точек прогноза только тех, где объект обнаружен
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local_coords = np.nanmean(local_coords, axis=1) # [n_tracks, n_class, 4]
        confidence_level = np.nanmax(confidence_level, axis=1) # [n_tracks, n_class]
    
    # Берем  нужный класс
    local_coords = local_coords[:, class_id, ...] # [n_tracks, 4]
    confidence_level = confidence_level[:, class_id] # [n_tracks]

    # Сферические координаты предсказания
    x = local_coords[:,0]
    y = local_coords[:,1]
    z = local_coords[:,2]
    distance = local_coords[:,3]
    l_xy = np.sqrt(x**2 + y**2) 
    azimuth = np.rad2deg(np.arctan2(y,x))
    elevation = np.rad2deg(np.arctan2(z, l_xy))
    return confidence_level, azimuth, elevation, distance

