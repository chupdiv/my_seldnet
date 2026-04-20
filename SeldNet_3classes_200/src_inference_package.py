"""
Упаковка артефактов для инференса в основном проекте (src/app): model.conf, weights.pth, scaler.joblib.

Распакуйте zip в каталог модели под models/ приложения; имя каталога — произвольное.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import Any, Callable, Dict, Mapping, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# Параметры обучения/путей к данным, которые не нужны в model.conf для инференса.
_EXCLUDE_FROM_PARAMS = frozenset(
    {
        "dataset_dir",
        "feat_label_dir",
        "model_dir",
        "dcase_output_dir",
        "quick_test",
        "mode",
        "nb_epochs",
        "eval_freq",
        "lr",
        "final_lr",
        "batch_size",
        "eval_batch_size",
        "warmup",
        "weight_decay",
        "patience",
        "pretrained_model_weights",
        "finetune_mode",
    }
)


def _to_python_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _model_type_to_src_class(model: str) -> str:
    m = (model or "seldnet").lower()
    if m == "cstformer":
        return "CSTformer"
    # seldnet, myseldnet, ngccmodel — класс-обёртка в specialmodels/seldnet.py
    return "SELDNet"


def _needs_dummy_scaler(params: Mapping[str, Any]) -> bool:
    m = str(params.get("model", "seldnet")).lower()
    if m == "ngccmodel":
        return True
    if bool(params.get("use_ngcc", False)):
        return True
    return False


def scaler_path_from_training_params(params: Mapping[str, Any]) -> str:
    """Путь к файлу весов нормализации, как в cls_feature_class.FeatureClass.get_normalized_wts_file."""
    feat_label_dir = params["feat_label_dir"]
    dataset = params["dataset"]
    raw_chunks = params.get("raw_chunks", False)
    name = dataset if not raw_chunks else f"{dataset}_raw_chunks"
    return os.path.join(feat_label_dir, f"{name}_wts")


def _write_dummy_scaler(path: str, n_features: int = 64) -> None:
    sc = StandardScaler()
    sc.fit(np.zeros((2, max(1, int(n_features))), dtype=np.float32))
    joblib.dump(sc, path)


def build_params_section(
    params: Mapping[str, Any],
    task_id: str,
) -> Dict[str, str]:
    """Ключи и строковые значения для секции [params] (совместимо с ast.literal_eval в src)."""
    out: Dict[str, str] = {}
    out["ngcc_seld_task"] = repr(_to_python_scalar(task_id))

    for key, val in params.items():
        if key in _EXCLUDE_FROM_PARAMS or key == "ngcc_seld_task":
            continue
        try:
            v = _to_python_scalar(val)
        except Exception:
            continue
        if v is None:
            continue
        # Пропускаем не сериализуемые объекты
        if isinstance(v, (dict, list, tuple, str, int, float, bool)) or v is None:
            pass
        else:
            try:
                repr(v)
            except Exception:
                continue
        out[key] = repr(v)
    return out


def write_model_conf(
    path: str,
    *,
    caption: str,
    description: str,
    src_model_class: str,
    params_lines: Mapping[str, str],
) -> None:
    lines = [
        "# Сгенерировано при обучении; для uav_acoustic_classification (src).",
        "# Распакуйте архив в подкаталог models/<имя>/",
        "",
        "[tags]",
        f"caption = {caption}",
        f"description = {description}",
        f"class = {src_model_class}",
        "type = localizer",
        "order = 10",
        "",
        "[files]",
        "model = weights.pth",
        "scaler = scaler.joblib",
        "",
        "[params]",
    ]
    for k in sorted(params_lines.keys()):
        lines.append(f"{k} = {params_lines[k]}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def create_src_inference_zip(
    zip_path: str,
    weights_src_path: str,
    params: Mapping[str, Any],
    task_id: str,
    *,
    unique_name: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Создаёт zip: model.conf, weights.pth, scaler.joblib (+ README.txt).

    weights_src_path — обычно *_model_final.h5 из train_seldnet.
    Возвращает путь к zip.
    """
    log = log_fn or print

    if not os.path.isfile(weights_src_path):
        raise FileNotFoundError(f"Нет файла весов: {weights_src_path}")

    src_class = _model_type_to_src_class(str(params.get("model", "seldnet")))
    params_lines = build_params_section(params, task_id)

    caption = f"SELD {unique_name}"
    description = f"task_id={task_id}, model={params.get('model')}, упаковано для src"

    with tempfile.TemporaryDirectory(prefix="src_pack_") as tmp:
        conf_path = os.path.join(tmp, "model.conf")
        write_model_conf(
            conf_path,
            caption=caption,
            description=description,
            src_model_class=src_class,
            params_lines=params_lines,
        )
        w_dst = os.path.join(tmp, "weights.pth")
        shutil.copy2(weights_src_path, w_dst)

        scaler_dst = os.path.join(tmp, "scaler.joblib")
        if _needs_dummy_scaler(params):
            n_mel = int(params.get("nb_mel_bins", 64))
            _write_dummy_scaler(scaler_dst, n_features=n_mel)
            log("src_inference_package: для NGCC-ветки записан фиктивный scaler.joblib (не используется при инференсе).")
        else:
            sc_path = scaler_path_from_training_params(params)
            if not os.path.isfile(sc_path):
                raise FileNotFoundError(
                    f"Не найден scaler обучения: {sc_path}. Выполните preprocess_features / обучение с нормализацией признаков."
                )
            shutil.copy2(sc_path, scaler_dst)

        readme = os.path.join(tmp, "README.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "Инференс для uav_acoustic_classification (src)\n"
                "-----------------------------------------------\n"
                "1. Распакуйте этот архив в каталог моделей приложения, например:\n"
                "   models/<имя_модели>/\n"
                "2. В каталоге должны лежать: model.conf, weights.pth, scaler.joblib\n"
                "3. Убедитесь, что в основном конфиге приложения указан каталог models.\n"
            )

        zip_dir = os.path.dirname(os.path.abspath(zip_path))
        if zip_dir:
            os.makedirs(zip_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("model.conf", "weights.pth", "scaler.joblib", "README.txt"):
                fp = os.path.join(tmp, name)
                zf.write(fp, arcname=name)

    log(f"src_inference_package: создан архив {zip_path}")
    return zip_path
