"""
Параметры задач (task id), совместимые с parameters.py репозитория ngcc-seld:
https://github.com/axeber01/ngcc-seld

Используются как базовые значения для инференса; значения из model.conf [params]
переопределяют их (см. merge_ngcc_params_with_user).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Optional


def _default_params() -> Dict[str, Any]:
    return dict(
        quick_test=False,
        finetune_mode=False,
        dataset_dir="./data_2024/",
        feat_label_dir="./data_2024/seld_feat_label/",
        model_dir="models",
        dcase_output_dir="results",
        mode="dev",
        dataset="mic",
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,
        use_salsalite=False,
        raw_chunks=False,
        saved_chunks=False,
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,
        model="seldnet",
        modality="audio",
        multi_accdoa=False,
        thresh_unify=15,
        label_sequence_length=50,
        batch_size=64,
        eval_batch_size=64,
        dropout_rate=0.05,
        nb_cnn2d_filt=64,
        f_pool_size=[4, 4, 2],
        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,
        nb_rnn_layers=2,
        rnn_size=128,
        nb_fnn_layers=1,
        fnn_size=128,
        nb_epochs=300,
        eval_freq=25,
        lr=1e-3,
        final_lr=1e-5,
        weight_decay=0.05,
        predict_tdoa=False,
        warmup=5,
        relative_dist=True,
        no_dist=False,
        average="macro",
        segment_based_metrics=False,
        evaluate_distance=True,
        lad_doa_thresh=20,
        lad_dist_thresh=float("inf"),
        lad_reldist_thresh=float("1"),
        encoder="conv",
        LinearLayer=False,
        FreqAtten=False,
        ChAtten_DCA=False,
        ChAtten_ULE=False,
        CMT_block=False,
        CMT_split=False,
        use_ngcc=False,
        use_mfcc=False,
        baseline=False,
        t_pooling_loc="front",
        MSCAM=False,
        nb_resnet_filt=64,
    )


def _apply_derived(params: MutableMapping[str, Any]) -> None:
    params["feature_label_resolution"] = int(params["label_hop_len_s"] // params["hop_len_s"])
    params["feature_sequence_length"] = params["label_sequence_length"] * params["feature_label_resolution"]
    params["t_pool_size"] = [params["feature_label_resolution"], 1, 1]
    params["patience"] = int(params["nb_epochs"])
    params["model_dir"] = str(params["model_dir"]) + "_" + str(params["modality"])
    params["dcase_output_dir"] = str(params["dcase_output_dir"]) + "_" + str(params["modality"])


def _set_unique_classes(params: MutableMapping[str, Any], dataset_dir: str) -> None:
    if "2020" in dataset_dir:
        params["unique_classes"] = 14
    elif "2021" in dataset_dir:
        params["unique_classes"] = 12
    elif "2022" in dataset_dir:
        params["unique_classes"] = 13
    elif "2023" in dataset_dir:
        params["unique_classes"] = 13
    elif "2024" in dataset_dir:
        params["unique_classes"] = 13
    elif "sim" in dataset_dir:
        params["unique_classes"] = 13
    else:
        params["unique_classes"] = 13


def _set_nb_channels_mic(params: MutableMapping[str, Any]) -> None:
    if params.get("dataset") != "mic":
        return
    # Явно заданные в ветке task (например 33: nb_channels=10) не перезаписываем
    if "nb_channels" in params:
        return
    if params.get("use_ngcc"):
        if params.get("use_mel"):
            params["nb_channels"] = int(
                params["ngcc_out_channels"] * params["n_mics"] * (params["n_mics"] - 1) / 2
                + params["n_mics"]
            )
        else:
            params["nb_channels"] = int(
                params["ngcc_out_channels"]
                * params["n_mics"]
                * (1 + (params["n_mics"] - 1) / 2)
            )
    elif params.get("use_salsalite"):
        params["nb_channels"] = 7
    else:
        params["nb_channels"] = 7


def get_ngcc_seld_params(argv: str = "1", dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Возвращает словарь параметров для номера задачи, как в ngcc-seld/parameters.py get_params().
    """
    params: Dict[str, Any] = _default_params()
    if dataset_dir is not None:
        params["dataset_dir"] = dataset_dir

    _apply_derived(params)

    a = str(argv).strip()

    if a == "1":
        pass
    elif a == "2":
        params["quick_test"] = False
        params["dataset"] = "foa"
        params["multi_accdoa"] = False
    elif a == "3":
        params["quick_test"] = False
        params["dataset"] = "foa"
        params["multi_accdoa"] = True
    elif a == "4":
        params["quick_test"] = False
        params["dataset"] = "mic"
        params["use_salsalite"] = False
        params["multi_accdoa"] = False
    elif a == "5":
        params["quick_test"] = False
        params["dataset"] = "mic"
        params["use_salsalite"] = True
        params["multi_accdoa"] = False
    elif a == "6":
        params["quick_test"] = False
        params["dataset"] = "mic"
        params["use_salsalite"] = False
        params["multi_accdoa"] = True
        params["n_mics"] = 4
    elif a == "7":
        params["quick_test"] = False
        params["dataset"] = "mic"
        params["use_salsalite"] = True
        params["multi_accdoa"] = True
    elif a == "9":
        params["label_sequence_length"] = 1
        params["feature_sequence_length"] = params["label_sequence_length"] * params["feature_label_resolution"]
        params["raw_chunks"] = True
        params["pretrained_model_weights"] = "blah.h5"
        params["quick_test"] = False
        params["dataset"] = "mic"
        params["use_salsalite"] = False
        params["multi_accdoa"] = True
        params["n_mics"] = 4
        params["model"] = "ngccmodel"
        params["ngcc_channels"] = 32
        params["ngcc_out_channels"] = 16
        params["saved_chunks"] = True
        params["use_mel"] = False
        params["nb_epochs"] = 1
        params["predict_tdoa"] = True
        params["lambda"] = 1.0
        params["max_tau"] = 6
        params["tracks"] = 3
        params["fixed_tdoa"] = False
        params["batch_size"] = 32
        params["lr"] = 1e-4
        params["warmup"] = 0
    elif a == "10":
        params["finetune_mode"] = True
        params["raw_chunks"] = True
        params["pretrained_model_weights"] = "models/9_tdoa-3tracks-16channels.h5"
        params["quick_test"] = False
        params["dataset"] = "mic"
        params["use_salsalite"] = False
        params["multi_accdoa"] = True
        params["n_mics"] = 4
        params["model"] = "ngccmodel"
        params["ngcc_channels"] = 32
        params["ngcc_out_channels"] = 16
        params["saved_chunks"] = True
        params["use_mel"] = True
        params["predict_tdoa"] = False
        params["lambda"] = 0.0
        params["max_tau"] = 6
        params["tracks"] = 3
        params["fixed_tdoa"] = True
    elif a == "32":
        params["model"] = "cstformer"
        params["quick_test"] = False
        params["multi_accdoa"] = True
        params["t_pooling_loc"] = "front"
        params["FreqAtten"] = True
        params["ChAtten_DCA"] = True
        params["CMT_block"] = True
        params["f_pool_size"] = [2, 2, 1]
        params["t_pool_size"] = [params["feature_label_resolution"], 1, 1]
        params["fnn_size"] = 256
    elif a == "33":
        params["model"] = "cstformer"
        params["quick_test"] = False
        params["multi_accdoa"] = True
        params["t_pooling_loc"] = "front"
        params["n_mics"] = 4
        params["FreqAtten"] = True
        params["ChAtten_ULE"] = True
        params["CMT_block"] = True
        params["f_pool_size"] = [1, 2, 2]
        params["t_pool_size"] = [1, 1, params["feature_label_resolution"]]
        params["nb_fnn_layers"] = 1
        params["fnn_size"] = 256
        params["nb_channels"] = 10
    elif a == "34":
        params["model"] = "cstformer"
        params["use_salsalite"] = True
        params["quick_test"] = False
        params["multi_accdoa"] = True
        params["t_pooling_loc"] = "front"
        params["FreqAtten"] = True
        params["ChAtten_ULE"] = True
        params["CMT_block"] = True
        params["f_pool_size"] = [1, 4, 6]
        params["t_pool_size"] = [1, 1, params["feature_label_resolution"]]
        params["nb_fnn_layers"] = 1
        params["fnn_size"] = 256
    elif a == "333":
        params["model"] = "cstformer"
        params["use_ngcc"] = True
        params["quick_test"] = False
        params["multi_accdoa"] = True
        params["t_pooling_loc"] = "front"
        params["FreqAtten"] = True
        params["ChAtten_ULE"] = True
        params["CMT_block"] = True
        params["f_pool_size"] = [1, 2, 2]
        params["t_pool_size"] = [1, 1, params["feature_label_resolution"]]
        params["nb_fnn_layers"] = 1
        params["fnn_size"] = 256
        params["finetune_mode"] = True
        params["raw_chunks"] = True
        params["pretrained_model_weights"] = "models/9_tdoa-3tracks-16channels.h5"
        params["dataset"] = "mic"
        params["n_mics"] = 4
        params["ngcc_channels"] = 32
        params["ngcc_out_channels"] = 16
        params["saved_chunks"] = True
        params["use_mel"] = True
        params["use_mfcc"] = False
        params["predict_tdoa"] = False
        params["lambda"] = 0.0
        params["max_tau"] = 6
        params["tracks"] = 3
        params["fixed_tdoa"] = True
    elif a == "999":
        params["quick_test"] = True
    else:
        raise ValueError(f"Неизвестный ngcc-seld task id: {argv!r}")

    _set_unique_classes(params, str(params["dataset_dir"]))
    _set_nb_channels_mic(params)
    sync_inference_params(params)
    return params


def normalize_keys_for_local_models(params: MutableMapping[str, Any]) -> None:
    """Ключи PascalCase из parameters.py -> ключи, ожидаемые CST/encoder в этом репозитории."""
    if "freqatten" not in params:
        params["freqatten"] = bool(params.get("FreqAtten", False))
    if "chatten_dca" not in params:
        params["chatten_dca"] = bool(params.get("ChAtten_DCA", False))
    if "chatten_ule" not in params:
        params["chatten_ule"] = bool(params.get("ChAtten_ULE", False))
    if "cmt_block" not in params:
        params["cmt_block"] = bool(params.get("CMT_block", False))
    if "linearlayer" not in params:
        params["linearlayer"] = bool(params.get("LinearLayer", False))
    # CMT_Block / layers.py (Spec_attention, Temp_attention) ожидают PascalCase
    if "FreqAtten" not in params and "freqatten" in params:
        params["FreqAtten"] = bool(params["freqatten"])
    if "LinearLayer" not in params and "linearlayer" in params:
        params["LinearLayer"] = bool(params["linearlayer"])
    if "linear_layer" not in params:
        params["linear_layer"] = bool(params.get("linearlayer", False))


def sync_inference_params(params: MutableMapping[str, Any]) -> None:
    """Согласовать unique_classes с classes_list; дополнить ключи после model.conf."""
    cl = params.get("classes_list")
    if isinstance(cl, (list, tuple)) and len(cl) > 0:
        params["unique_classes"] = len(cl)
    normalize_keys_for_local_models(params)


def merge_ngcc_params_with_user(
    base: Mapping[str, Any],
    user: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Объединяет базу (task) и параметры из model.conf. Совпадающие ключи берутся из user.
    Ключ ngcc_seld_task в user игнорируется (только метка).
    """
    out = deepcopy(dict(base))
    skip = {"ngcc_seld_task", "ngcc_task"}
    for k, v in user.items():
        if k in skip:
            continue
        out[k] = deepcopy(v) if isinstance(v, (dict, list)) else v
    sync_inference_params(out)
    return out


def load_params_with_optional_task(
    raw_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    raw_params — уже распарсенные значения из model.conf [params].
    Если задан ngcc_seld_task или ngcc_task, подмешивается база из ngcc-seld.
    """
    task = raw_params.get("ngcc_seld_task", raw_params.get("ngcc_task"))
    dataset_dir = raw_params.get("dataset_dir")
    if task is None:
        user = dict(raw_params)
        user.pop("ngcc_seld_task", None)
        user.pop("ngcc_task", None)
        p = dict(user)
        sync_inference_params(p)
        return p
    user = dict(raw_params)
    user.pop("ngcc_seld_task", None)
    user.pop("ngcc_task", None)
    base = get_ngcc_seld_params(str(task), dataset_dir=dataset_dir)
    return merge_ngcc_params_with_user(base, user)
