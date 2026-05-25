import numpy as np
from tqdm import tqdm

from annotation import Annotation
from config import build_config, print_config
from planning import get_dataset
from render import (
    generate_linear_trajectory, 
    render_source, 
    annotate_trajectory, 
    normalize_to_random_rms,
    normalize_rms, 
    limit_float_wav_peak,
    scale_to_target_peak,
    is_silent,
)

from tools import (
    ensure_output_dirs, 
    convert_to_SELDformat, 
    filtering, 
    write_annotations, 
    write_soundfile,
    write_plan,
    make_output_filenames,
    run_sanity_check)
from augment import build_augment_pipeline

def make_single_signal(elem, label, cfg, rng=None, aug_pipeline=None):
    if rng is None:
        rng = np.random.default_rng()

    c_cfg = cfg['classes_configs'][label]
    idx_label = c_cfg['index']
    extractor = c_cfg['extractor']

    # Формирование траектории    
    traj = generate_linear_trajectory(
        duration = cfg['duration_sec'],
        frame_step = cfg['frame_step'],
        room_dim = elem['scene'].room, 
        start_time = 0.0,
        rng = rng,
    )

    # Преобразование сигнала
    y, ann, info = extractor(
        files = c_cfg,
        duration_s = cfg['duration_sec'],
        resolution_ms = cfg['frame_step'],
        sr = cfg['fs'],
        lowpass_hz = cfg['lowpass'],
        rng = rng,
    )
    if aug_pipeline is not None:
        y = aug_pipeline(samples=y, sample_rate=cfg['fs'])     # Аугментация

    y_mult = render_source(
        x = y,
        fs = cfg['fs'],
        traj = traj,
        scene = elem['scene'],
        hop = max(1, int(cfg['frame_step'] * cfg['fs']) // 10),
        peak_normalize = False,
        # peak_target = 1.0,
        chunk_size=cfg['chunk_size'],
        rt60=cfg['rt60'],
        att_diff= cfg['att_diff'],
        att_max = cfg['att_max'],
    )

    return y_mult, ann, info, traj


def mix_signals(elem, cfg, rng=None, aug_pipeline=None):
    """Смикшировать сигналы от активных источников с контролем SNR и пикового уровня.
    
    Исправления по сравнению с оригинальной версией:
    1. Нормализация RMS применяется к каждому источнику ДО микширования (не после)
    2. Сохраняется относительная энергетика источников через коэффициенты levels
    3. Итоговый сигнал масштабируется к целевому пику (не просто clipping)
    4. Добавлена проверка на тишину
    """
    signal = None
    trajectories = []
    description = {}
    
    for label, conf in cfg['classes_configs'].items():
        i = conf['index']
        active = elem['sample'][i]
        if active:
            y_mult, ann, info, traj = make_single_signal(elem, label, cfg, rng, aug_pipeline)
            
            # Нормализация каждого источника к единичному RMS перед микшированием
            # Это сохраняет относительную энергетику при последующем масштабировании coef
            y_mult = normalize_rms(y_mult, sigma=1.0)
            
            description[label] = info 
            trajectories += annotate_trajectory(label, traj, ann, cfg['classes_configs'])

            coef = elem['levels'][i]
            if signal is None:
                signal = coef * y_mult
                l_signal = len(signal)
            else:
                l_signal = min(len(signal), len(y_mult))
                signal = signal[:l_signal, :] + coef * y_mult[:l_signal, :]

    # Проверка на тишину
    if signal is None or is_silent(signal, threshold=0.001):
        # Создаём тихий сигнал вместо пустого
        signal = np.zeros((l_signal, cfg['n_mics']), dtype=np.float32)
    
    # Масштабирование к целевому пику вместо простого clipping
    # Это обеспечивает адекватный уровень сигнала для обучения
    signal = scale_to_target_peak(
        signal,
        peak_target_min=cfg.get('peak_target_min', 0.10),
        peak_target_max=cfg.get('peak_target_max', 0.98),
        rng=rng,
    )
    
    # Финальный lowpass фильтр и ограничение пика (защита от клиппинга)
    signal = filtering(signal, cfg['fs'], cfg['lowpass'])
    signal = limit_float_wav_peak(signal, peak_target=0.98)

    elem['description'] = description
    elem['duration'] = l_signal / cfg['fs']

    trajectories.sort(key=lambda x: x[3])
    trajectories = np.asarray(trajectories, dtype=np.float32)
    return signal, trajectories

def process_program(task_type:str, dataset, cfg, rng=None, aug_pipeline = None):
    i = 0
    for elem in tqdm(dataset):
        signal, trajectories = mix_signals(
                elem = elem, 
                cfg = cfg,
                rng=rng,
                aug_pipeline = aug_pipeline)
     
        annotations = convert_to_SELDformat(
            annotations=trajectories, 
            mic_center=elem['scene'].mic_center,
            frame_step=cfg['frame_step'],
        )
        wav_fname, annot_fname = make_output_filenames(i, elem['sample'], task_type, cfg)
        write_soundfile(wav_fname, signal, cfg['fs'])
        write_annotations(annot_fname, annotations)
        i += 1
    write_plan(task_type, dataset, cfg)
    run_sanity_check(cfg)
    return None

def main():
    cfg = build_config()        # вместо parse_args() + configure_from_args()
    print_config(cfg)
    rng = np.random.default_rng(cfg["seed"])
    ensure_output_dirs(cfg)

   
    # Набор классов, которые будут представлены в сцене.
    nb_train = int(round(cfg['files_count']*cfg['train_fraction']))
    nb_test = int(round(cfg['files_count']*(1-cfg['train_fraction'])))

    aug_pipeline = build_augment_pipeline(cfg["augmentation"], fs=cfg['fs'], rng=rng)
    train_program = get_dataset(
        plan = cfg['classes_plan'],
        count = nb_train,
        cfg = cfg,
        rng=rng
    )
    print(f'Формирование обучающей выборки: {len(train_program)} файлов')
    process_program(
        task_type= 'train',
        dataset=train_program, 
        cfg=cfg, 
        rng=rng,
        aug_pipeline=aug_pipeline,
    )

    test_program = get_dataset(
        plan = cfg['classes_plan'],
        count = nb_test,
        cfg = cfg,
        rng=rng
    )
    print(f'Формирование тестовой выборки: {len(test_program)} файлов')
    process_program(
        task_type= 'test',
        dataset=test_program, 
        cfg=cfg, 
        rng=rng,
        aug_pipeline=None
    )
main()

    