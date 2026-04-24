from typing import Any, Dict, List, Optional, Tuple, Set

# =============================================================================
# Параметры по умолчанию для SeldNet_3classes_80 (Task IDs 2-7)
# =============================================================================
DEFAULT_PARAMS_80 = dict(
    fs=250000,
    hop_len_s=0.02,
    label_hop_len_s=0.04,
    nb_mel_bins=256,
    mel_fmin_hz=0.0,
    mel_fmax_hz=5000.0,
    n_mics=4,
    classes_list=['class_0', 'class_1', 'class_2'],  # список классов
    target_class=0,  # целевой класс (индекс в classes_list)
    use_salsalite=False,
    fmin_doa_salsalite=50,
    fmax_doa_salsalite=2000,
    fmax_spectra_salsalite=5000,
    model='seldnet',
    multi_accdoa=True,
    label_sequence_length=5,
    dropout_rate=0.05,
    nb_cnn2d_filt=64,
    f_pool_size=[4, 4, 2],
    t_pool_size=[2, 1, 1],
    nb_heads=8,
    nb_self_attn_layers=2,
    nb_rnn_layers=2,
    rnn_size=128,
    nb_fnn_layers=1,
    fnn_size=128,
    dataset='mic',
    raw_chunks=False,
)


# =============================================================================
# Параметры по умолчанию для CSTFormer (Task ID 32, 33, 34, 333)
# =============================================================================
DEFAULT_PARAMS_CST = dict(
    fs=250000,
    hop_len_s=0.02,
    label_hop_len_s=0.04,
    nb_mel_bins=256,
    mel_fmin_hz=0.0,
    mel_fmax_hz=5000.0,
    n_mics=4,
    classes_list=['class_0', 'class_1', 'class_2'],  # список классов
    target_class=0,  # целевой класс (индекс в classes_list)
    model='cstformer',
    multi_accdoa=True,
    label_sequence_length=5,
    dropout_rate=0.1,
    nb_cnn2d_filt=64,
    f_pool_size=[4, 4, 2],
    use_salsalite=False,  # по умолчанию не используем SALSA
    # Специфичные параметры для Трансформера
    patch_size=16,
    num_heads=8,
    embed_dim=256,
    num_layers=4,
    # Параметры для совместимости с SeldModel
    rnn_size=128,
    nb_rnn_layers=2,
    nb_self_attn_layers=2,
    nb_fnn_layers=1,
    fnn_size=128,
    dataset='mic',
    raw_chunks=False,
    # CST-specific parameters
    t_pooling_loc='begin',
    ChAtten_DCA=False,
    ChAtten_ULE=False,
    CMT_block=False,
    use_ngcc=False,
    use_mfcc=False,
    predict_tdoa=False,
    max_tau=128,
    ngcc_channels=32,
    ngcc_out_channels=16,
    tracks=False,
    fixed_tdoa=False,
)


# =============================================================================
# Параметры по умолчанию для NGCC Model (Task ID 9, 10)
# =============================================================================
DEFAULT_PARAMS_NGCC = dict(
    fs=250000,
    hop_len_s=0.02,
    label_hop_len_s=0.04,
    nb_mel_bins=256,
    mel_fmin_hz=0.0,
    mel_fmax_hz=5000.0,
    n_mics=4,
    classes_list=['class_0', 'class_1', 'class_2'],  # список классов
    target_class=0,  # целевой класс (индекс в classes_list)
    model='ngccmodel',
    multi_accdoa=False,
    label_sequence_length=5,
    dropout_rate=0.05,
    nb_cnn2d_filt=64,
    f_pool_size=[4, 4, 2],
    use_salsalite=False,
    dataset='mic',
    raw_chunks=False,
    # NGCC-specific parameters
    use_ngcc=True,
    use_mfcc=False,
    predict_tdoa=False,
    max_tau=128,
    ngcc_channels=32,
    ngcc_out_channels=16,
    tracks=False,
    fixed_tdoa=False,
    # Для совместимости
    rnn_size=128,
    nb_rnn_layers=2,
    nb_heads=8,
    nb_self_attn_layers=2,
    nb_fnn_layers=1,
    fnn_size=128,
)


# =============================================================================
# Требуемые параметры для каждого task_id
# =============================================================================
REQUIRED_PARAMS_BY_TASK: Dict[str, Set[str]] = {
    '2': {  # FOA, single-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '3': {  # FOA, multi-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '4': {  # MIC, single-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '5': {  # MIC, SALSA, single-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'use_salsalite',
        'fmin_doa_salsalite', 'fmax_doa_salsalite', 'fmax_spectra_salsalite', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '6': {  # MIC, multi-ACCDOA (основная конфигурация SeldNet_3classes_80)
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '7': {  # MIC, SALSA, multi-ACCDOA
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'use_salsalite',
        'fmin_doa_salsalite', 'fmax_doa_salsalite', 'fmax_spectra_salsalite', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
    },
    '33': {  # CSTFormer - Transformer-based SELD
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'patch_size', 'num_heads', 'embed_dim', 'num_layers', 'dropout_rate',
        'nb_cnn2d_filt', 'f_pool_size', 'use_salsalite',
        't_pooling_loc', 'ChAtten_DCA', 'ChAtten_ULE', 'CMT_block',
        'use_ngcc', 'max_tau', 'ngcc_channels', 'ngcc_out_channels',
        # Параметры для совместимости с SeldModel (хотя CST может использовать другую архитектуру)
        'rnn_size', 'nb_rnn_layers', 'nb_heads', 'nb_self_attn_layers', 'nb_fnn_layers', 'fnn_size',
    },
    '9': {  # NGCC Model, label_seq=1
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
        'use_ngcc', 'max_tau', 'ngcc_channels', 'ngcc_out_channels',
    },
    '10': {  # NGCC Model, label_seq=5
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate', 'nb_heads', 'nb_self_attn_layers',
        'nb_rnn_layers', 'rnn_size', 'nb_fnn_layers', 'fnn_size',
        'use_ngcc', 'max_tau', 'ngcc_channels', 'ngcc_out_channels',
    },
    '32': {  # CSTFormer
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'patch_size', 'num_heads', 'embed_dim', 'num_layers', 'dropout_rate',
        'nb_cnn2d_filt', 'f_pool_size', 'use_salsalite',
        't_pooling_loc', 'ChAtten_DCA', 'ChAtten_ULE', 'CMT_block',
        'use_ngcc', 'max_tau', 'ngcc_channels', 'ngcc_out_channels',
        # Параметры для совместимости
        'rnn_size', 'nb_rnn_layers', 'nb_heads', 'nb_self_attn_layers', 'nb_fnn_layers', 'fnn_size',
    },
    '34': {  # CSTFormer variant
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'patch_size', 'num_heads', 'embed_dim', 'num_layers', 'dropout_rate',
        'nb_cnn2d_filt', 'f_pool_size', 'use_salsalite',
        't_pooling_loc', 'ChAtten_DCA', 'ChAtten_ULE', 'CMT_block',
        'use_ngcc', 'max_tau', 'ngcc_channels', 'ngcc_out_channels',
        # Параметры для совместимости
        'rnn_size', 'nb_rnn_layers', 'nb_heads', 'nb_self_attn_layers', 'nb_fnn_layers', 'fnn_size',
    },
    '333': {  # CSTFormer variant
        'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
        'n_mics', 'classes_list', 'target_class', 'label_sequence_length', 'dataset', 'multi_accdoa',
        'patch_size', 'num_heads', 'embed_dim', 'num_layers', 'dropout_rate',
        'nb_cnn2d_filt', 'f_pool_size', 'use_salsalite',
        't_pooling_loc', 'ChAtten_DCA', 'ChAtten_ULE', 'CMT_block',
        'use_ngcc', 'max_tau', 'ngcc_channels', 'ngcc_out_channels',
        # Параметры для совместимости
        'rnn_size', 'nb_rnn_layers', 'nb_heads', 'nb_self_attn_layers', 'nb_fnn_layers', 'fnn_size',
    },
}

# Параметры, используемые непосредственно в инференсе (минимальный набор)
INFERENCE_ONLY_PARAMS = {
    'fs', 'hop_len_s', 'label_hop_len_s', 'nb_mel_bins', 'mel_fmin_hz', 'mel_fmax_hz',
    'n_mics', 'dataset', 'use_salsalite', 'multi_accdoa', 'classes_list', 'target_class',
    'label_sequence_length', 'nb_cnn2d_filt', 'f_pool_size', 'dropout_rate',
    'nb_heads', 'nb_self_attn_layers', 'nb_rnn_layers', 'rnn_size',
    'nb_fnn_layers', 'fnn_size',
}

