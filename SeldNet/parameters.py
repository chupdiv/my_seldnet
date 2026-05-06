# === Параметры для управления моделями ===

# --- Для скоростного АЦП ---
MIC_LOCS = [
    [ 0.0,      0.0,   0.08016],
    [ 0.0577,   0.0,   0.0],
    [-0.02885, -0.05, 0.0],
    [-0.02885,  0.05, 0.0],
]
FS = 250000

HOP_LEN_S=0.025
LABEL_HOP_LEN_S=0.05
LABEL_SEQUENCE_LENGTH = 5
NB_MEL_BINS=256
MAX_TAU = 128

# --- Общие параметры ---
MAX_AUDIO_LEN_S = 5

MEL_FMIN_HZ = 0.0
MEL_FMAX_HZ = 5000.0
LOWPASS_HZ = 20000.0

BATCH_SIZE = 128
NB_EPOCHS = 800


# =========================================

from pathlib import Path

CURRENT_DIR = Path.cwd()
DATASETS_DIR = CURRENT_DIR / '../datasets/seld_data'

def _recompute_time_derived(params):
    """Call after label_hop_len_s and label_sequence_length are final for the chosen task."""
    params['feature_label_resolution'] = int(params['label_hop_len_s'] / params['hop_len_s'])
    params['feature_sequence_length'] = (
        params['label_sequence_length'] * params['feature_label_resolution']
    )

def get_params(argv='1'):
    N_MICS = len(MIC_LOCS)
    WORK_DIR = DATASETS_DIR / f"reverb_fs{FS}Hz_{N_MICS}mics"

    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights

        # INPUT PATH
        dataset_dir=str(WORK_DIR),

        # OUTPUT PATHS
        feat_label_dir = str(WORK_DIR / f'seld_feat_label_{argv}'),  # Directory to dump extracted features and labels

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        mic_locs = [
            [ 0.0,      0.0,   0.08016],
            [ 0.0577,   0.0,   0.0],
            [-0.02885, -0.05, 0.0],
            [-0.02885,  0.05, 0.0],
        ],
        fs=FS,
        hop_len_s=HOP_LEN_S,
        label_hop_len_s=LABEL_HOP_LEN_S,
        max_audio_len_s=MAX_AUDIO_LEN_S,
        nb_mel_bins=NB_MEL_BINS,
        # Mel: при узкополосном датасете (напр. --lowpass-hz 5000) задайте mel_fmax_hz (Гц) по срезу ФНЧ.
        # None = до Найквиста fs/2 (как раньше). Для генерации с --lowpass-hz 5000 обычно: mel_fmax_hz=5000.0
        mel_fmin_hz=MEL_FMIN_HZ,
        mel_fmax_hz=MEL_FMAX_HZ,
        lowpass_hz=LOWPASS_HZ,
        n_mics=N_MICS,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        raw_chunks=False,
        saved_chunks=False,
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        # Must be <= effective audio bandwidth (e.g. LPF 5 kHz in the generator).
        fmax_spectra_salsalite=5000,

        # MODEL TYPE
        model = 'seldnet',
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        # label_sequence_length = 5,    # Feature sequence length
        label_sequence_length = LABEL_SEQUENCE_LENGTH,    # Feature sequence length
        batch_size=BATCH_SIZE,              # Batch size
        eval_batch_size=BATCH_SIZE,
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,
        nb_rnn_layers=2,
        rnn_size=128,
        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=NB_EPOCHS,  # Train for maximum epochs
        eval_freq=5, # evaluate every x epochs
        lr=1e-3,
        final_lr=1e-5, # final learning rate in cosine scheduler
        weight_decay=0.05,
        predict_tdoa=False,
        warmup=5, #number of warmup epochs
        relative_dist = True, # scales MSE loss with 1/d
        no_dist = True, # removes distance from loss, can be used if we don't want to perform distance estimation

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics

        #CST-former params
        encoder = 'conv',           # ['conv', 'ResNet', 'SENet']
        LinearLayer = False,        # Linear Layer right after attention layers (usually not used/employed in baseline model)
        FreqAtten = False,          # Use of Divided Spectro-Temporal Attention (DST Attention)
        ChAtten_DCA = False,        # Use of Divided Channel-S-T Attention (CST Attention)
        ChAtten_ULE = False,        # Use of Divided C-S-T attention with Unfold (Unfolded CST attention)
        CMT_block = False,          # Use of LPU & IRFNN
        CMT_split = False,          # Apply LPU & IRFNN on S, T attention layers independently
        use_ngcc = False,
        use_mfcc = False,
        
    )

    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['n_mics'] = N_MICS

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '9': # TDOA pre-training
        print("RAW AUDIO CHUNKS w/ NGCC model + multi ACCDOA, TDOA-pretraining\n")
        params['label_sequence_length'] = 1 # use only one time frame for tdoa training
        params['raw_chunks'] = True
        # params['pretrained_model_weights'] = 'blah.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['n_mics'] = N_MICS
        params['model'] = 'ngccmodel'
        params['ngcc_channels'] = 32
        params['ngcc_out_channels'] = 16 
        params['saved_chunks'] = True
        params['use_mel'] = False
        params['nb_epochs'] = 1
        params['predict_tdoa'] = False
        params['lambda'] = 1.0 # set to 1.0 to only train tdoa, and 0.0 to only train SELD
        params['max_tau'] = MAX_TAU
        params['tracks'] = 3
        params['fixed_tdoa'] = False
        params['batch_size'] = 32
        params['lr'] = 1e-4
        params['warmup'] = 5

    elif argv == '10': # fine-tuning from tdoa-pretrained model
        print("RAW AUDIO CHUNKS w/ NGCC model + multi ACCDOA, pre-trained TDOA features\n")
        params['finetune_mode'] = False
        params['raw_chunks'] = True
        # params['pretrained_model_weights'] = 'models/9_tdoa-3tracks-16channels.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['n_mics'] = N_MICS
        params['model'] = 'ngccmodel'
        params['ngcc_channels'] = 32
        params['ngcc_out_channels'] = 16
        params['saved_chunks'] = True
        params['use_mel'] = True

        params['predict_tdoa'] = False
        params['lambda'] = 0.0 # set to 1.0 to only train tdoa, and 0.0 to only train SELD
        params['max_tau'] = MAX_TAU
        params['tracks'] = 3
        params['fixed_tdoa'] = True


    elif argv == '32':
        print("[CST-former: Divided Channel Attention] FOA + Multi-ACCDOA + CST_DCA + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_DCA'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [2, 2, 1]
        params['fnn_size'] = 256

    elif argv == '33':
        print("[CST-former: Unfolded Local Embedding] GCC + Multi-ACCDOA + CST Unfold + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [1,2,2]
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 256
        params['nb_channels'] = N_MICS*(N_MICS-1)//2 + N_MICS

    elif argv == '34':
        print("[CST-former: Unfolded Local Embedding] SALSA-LITE + Multi-ACCDOA + CST Unfold + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['use_salsalite'] = True
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [1,4,6]
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 256

    elif argv == '333': #CST former with NGCC-PHAT
        print("[CST-former: Unfolded Local Embedding] FOA + Multi-ACCDOA + CST Unfold + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['use_ngcc'] = True
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [1,2,2] # change to [1, 1, 1] to use the "Large" version
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 256

        params['finetune_mode'] = True
        params['raw_chunks'] = True
        params['pretrained_model_weights'] = 'models/9_tdoa-3tracks-16channels.h5' 
        params['dataset'] = 'mic'
        params['n_mics'] = N_MICS
        params['ngcc_channels'] = 32
        params['ngcc_out_channels'] = 16
        params['saved_chunks'] = True
        params['use_mel'] = True
        params['use_mfcc'] = False

        params['predict_tdoa'] = False
        params['lambda'] = 0.0 # set to 1.0 to only train tdoa, and 0.0 to only train SELD
        params['max_tau'] = MAX_TAU
        params['tracks'] = 3
        params['fixed_tdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    _recompute_time_derived(params)
    if argv == '32':
        params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]
    elif argv in ('33', '34', '333'):
        params['t_pool_size'] = [1, 1, params['feature_label_resolution']]
    else:
        params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]

    if params['dataset'] == 'mic':
        if params['use_ngcc']:
            if params['use_mel']:
                params['nb_channels'] = int(params['ngcc_out_channels'] * params['n_mics'] * (params['n_mics'] - 1) / 2 +  params['n_mics'])
            else:
                params['nb_channels'] = int(params['ngcc_out_channels'] * params['n_mics'] * ( 1 + (params['n_mics'] - 1) / 2))
        elif params['use_salsalite']:
            params['nb_channels'] = 7
        else:
            # Mel по каждому каналу + GCC по парам микрофонов (см. cls_feature_class GCC+mel)
            nm = int(params['n_mics'])
            params['nb_channels'] = nm + (nm * (nm - 1)) // 2
    else:
        params['nb_channels'] = 7


    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2024' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif 'sim' in params['dataset_dir']:
        params['unique_classes'] = 13
    else:
        params['unique_classes'] = 3
        
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
