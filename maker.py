#!/usr/bin/env python3
"""
maker.py — скрипт для вычисления признаков и обучения нейросети SELD.

Выполняет 3 этапа:
  1. Генерация отражений (generate_gpuRIR_3c.py) — при необходимости.
  2. Расчёт признаков (batch_feature_extraction.py) — при необходимости.
  3. Обучение нейросети (train_seldnet.py) — всегда, когда не отключено параметром --skip-train.

Использование:
    python maker.py --dir SeldNet_3classes_80 --task 6
    python maker.py --dir SeldNet_3classes_80 --task 6 --log output.log
    python maker.py --dir SeldNet_3classes_200 --task 4 --skip-rir --skip-feats --log /var/log/train.log
    python maker.py --dir SeldNet_3classes_80 --task 6 --skip-rir --skip-feats --skip-train
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================================
#  Константы
# ============================================================================

STATE_DIR = Path(".maker_state")

# Ключи параметров, используемые в команде этапа 1 (генерация отражений).
# Команда: generate_gpuRIR_3c.py --fs ... --n-mics ... --frame-step ...
#          --run-sanity-check --lowpass-hz 0 --seed ... --duration ...
RIR_PARAM_KEYS = ["fs", "n_mics", "label_hop_len_s", "seed", "max_audio_len_s", "lowpass_hz"]

# Ключи параметров, влияющих на расчёт признаков (этап 2).
FEATS_PARAM_KEYS = [
    "hop_len_s",
    "nb_mel_bins",
    "mel_fmin_hz",
    "mel_fmax_hz",
    "use_salsalite",
    "raw_chunks",
    "saved_chunks",
    "fmin_doa_salsalite",
    "fmax_doa_salsalite",
    "fmax_spectra_salsalite",
]

# Значение seed по умолчанию, если ключ отсутствует в params.
DEFAULT_SEED = 20260405


# ============================================================================
#  Утилиты: Tee (дублирование вывода в консоль и файл лога)
# ============================================================================

class Tee:
    """Записывает данные в несколько потоков одновременно (stdout + файл)."""

    def __init__(self, *targets):
        self.targets = list(targets)
        self._closed = False

    def write(self, data):
        for t in self.targets:
            try:
                if hasattr(t, "closed") and t.closed:
                    continue
                t.write(data)
            except (ValueError, OSError):
                pass

    def flush(self):
        for t in self.targets:
            try:
                if hasattr(t, "closed") and t.closed:
                    continue
                t.flush()
            except (ValueError, OSError):
                pass

    def close(self):
        self._closed = True
        for t in self.targets:
            try:
                if hasattr(t, "close") and not t.closed:
                    t.close()
            except Exception:
                pass

    def fileno(self):
        """Возвращает fd первого целевого потока (нужно для subprocess)."""
        if self.targets:
            try:
                return self.targets[0].fileno()
            except (AttributeError, OSError):
                pass
        return 1  # fallback на stdout


# ============================================================================
#  Утилиты: управление состоянием (сохранение/загрузка между запусками)
# ============================================================================

def get_state_path(dir_name: str, task_id: str) -> Path:
    """Возвращает путь к файлу состояния для конкретной комбинации dir + task."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{dir_name}_{task_id}.json"


def load_state(state_path: Path) -> dict:
    """Загружает состояние предыдущего запуска из JSON-файла."""
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state_path: Path, state: dict):
    """Сохраняет текущее состояние в JSON-файл."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ============================================================================
#  Утилиты: сериализация параметров (обработка float('inf') и Path)
# ============================================================================

def serialize_params(params: dict) -> dict:
    """Преобразует params в JSON-совместимую форму."""
    safe = {}
    for k, v in params.items():
        if isinstance(v, float):
            if v == float("inf"):
                safe[k] = "__inf__"
            elif v == float("-inf"):
                safe[k] = "__neg_inf__"
            else:
                safe[k] = v
        elif isinstance(v, Path):
            safe[k] = str(v)
        else:
            safe[k] = v
    return safe


def deserialize_params(safe_params: dict) -> dict:
    """Восстанавливает params из JSON-совместимой формы."""
    params = {}
    for k, v in safe_params.items():
        if v == "__inf__":
            params[k] = float("inf")
        elif v == "__neg_inf__":
            params[k] = float("-inf")
        else:
            params[k] = v
    return params


# ============================================================================
#  Утилиты: вычисление хеша каталога датасета
# ============================================================================

def archive_directory(dir_path: str) -> bool:
    """
    Переименовывает существующий каталог, добавляя очередной числовой индекс.

    Примеры:
        foo/           -> foo.1/
        foo.1/         -> foo.2/
        foo/ + foo.1/  -> foo.2/

    Возвращает True, если каталог был переименован, False — если не существовал.
    """
    if not os.path.isdir(dir_path):
        return False

    base = dir_path.rstrip(os.sep)
    # Если уже есть суффикс .N — работаем с ним, иначе начинаем с .1
    idx = 1
    candidate = f"{base}.{idx}"
    while os.path.exists(candidate):
        idx += 1
        candidate = f"{base}.{idx}"

    print(f"  Переименование: {base} -> {candidate}")
    os.rename(base, candidate)
    return True


def compute_dataset_hash(dataset_dir: str) -> str | None:
    """
    Вычисляет хеш содержимого каталога датасета.
    Использует пути файлов, время модификации и размер.
    Возвращает None, если каталог не существует.
    """
    if not os.path.isdir(dataset_dir):
        return None
    hasher = hashlib.md5()
    for root, dirs, files in sorted(os.walk(dataset_dir)):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                stat = os.stat(fpath)
                entry = f"{fpath}|{stat.st_mtime_ns}|{stat.st_size}"
                hasher.update(entry.encode("utf-8"))
            except OSError:
                pass
    return hasher.hexdigest()


# ============================================================================
#  Утилиты: динамический импорт модуля parameters.py
# ============================================================================

def load_module_from_path(module_name: str, file_path: str):
    """Динамически импортирует Python-модуль по пути к файлу."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль из {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
#  Утилиты: сравнение параметров
# ============================================================================

def compare_params(old_params: dict, new_params: dict) -> dict:
    """
    Сравнивает два словаря параметров.
    Возвращает словарь {key: {'old': ..., 'new': ...}} для изменившихся ключей.
    """
    changes = {}
    all_keys = sorted(set(list(old_params.keys()) + list(new_params.keys())))
    for key in all_keys:
        old_val = old_params.get(key)
        new_val = new_params.get(key)
        if old_val != new_val:
            changes[key] = {"old": old_val, "new": new_val}
    return changes


def print_param_changes(changes: dict, indent: int = 4):
    """Выводит форматированный отчёт об изменениях параметров."""
    prefix = " " * indent
    if not changes:
        print(f"{prefix}Изменений параметров нет.")
        return
    print(f"{prefix}Найдено изменённых параметров: {len(changes)}")
    for key in sorted(changes.keys()):
        vals = changes[key]
        old_repr = repr(vals["old"]) if vals["old"] is not None else "(отсутствовал)"
        new_repr = repr(vals["new"]) if vals["new"] is not None else "(отсутствует)"
        print(f"{prefix}  {key}:")
        print(f"{prefix}    было:  {old_repr}")
        print(f"{prefix}    стало: {new_repr}")


# ============================================================================
#  Запуск внешних команд
# ============================================================================

def run_command(cmd: list, description: str) -> int:
    """
    Запускает внешнюю команду и возвращает код завершения.
    Вывод команды транслируется в консоль (и в лог-файл, если включён).
    """
    print(f"\nЗапуск: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    rc = result.returncode
    if rc != 0:
        print(f"ОШИБКА: {description} завершилась с кодом возврата {rc}")
    else:
        print(f"Готово: {description} завершена успешно.")
    return rc


# ============================================================================
#  Основной пайплайн
# ============================================================================

def main() -> int:
    # ----- Разбор аргументов командной строки -----
    parser = argparse.ArgumentParser(
        description="Пайплайн: вычисление признаков и обучение нейросети SELD."
    )
    parser.add_argument(
        "--dir",
        required=True,
        choices=["SeldNet_3classes_80", "SeldNet_3classes_200"],
        help="Конфигурационный каталог (SeldNet_3classes_80 или SeldNet_3classes_200).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Идентификатор задачи (передаётся в get_params).",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Файл для сохранения журнала работы (опционально).",
    )
    parser.add_argument(
        "--skip-rir",
        action="store_true",
        help="Пропустить этап 1: генерацию отражений (gpuRIR).",
    )
    parser.add_argument(
        "--skip-feats",
        action="store_true",
        help="Пропустить этап 2: расчёт признаков (batch_feature_extraction).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Пропустить этап 3: обучение нейросети (train_seldnet).",
    )
    parser.add_argument(
        "--count",
        default=2000,
        type=int,
        help="Количество файлов формируемых при создании отражений.",
    )
    args = parser.parse_args()
    count_files = args.count
    dir_name = args.dir
    task_id = args.task
    log_file = args.log
    skip_rir = args.skip_rir
    skip_feats = args.skip_feats
    skip_train = args.skip_train

    # ----- Настройка логирования (Tee: консоль + файл) -----
    log_fh = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fh = open(log_file, "a", encoding="utf-8")
        log_fh.write(f"\n{'=' * 70}\n")
        log_fh.write(f"[maker] Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write(f"[maker]   --dir   {dir_name}\n")
        log_fh.write(f"[maker]   --task  {task_id}\n")
        log_fh.write(f"[maker]   --log   {log_file}\n")
        log_fh.write(f"{'=' * 70}\n")
        log_fh.flush()

        sys.stdout = Tee(sys.__stdout__, log_fh)
        sys.stderr = Tee(sys.__stderr__, log_fh)

    try:
        print(f"[maker] dir={dir_name}, task={task_id}")
        print(f"[maker] Начало работы: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ================================================================
        #  ЭТАП 0: Загрузка параметров и проверка изменений
        # ================================================================
        print(f"\n{'=' * 70}")
        print("Загрузка параметров и проверка изменений")
        print(f"{'=' * 70}")

        # 1.1  Импортируем parameters.py из <dir> и получаем параметры
        params_module_path = os.path.join(dir_name, "parameters.py")
        if not os.path.isfile(params_module_path):
            raise FileNotFoundError(
                f"Файл параметров не найден: {params_module_path}\n"
                f"Убедитесь, что каталог '{dir_name}' содержит файл parameters.py."
            )

        print(f"\nЗагрузка модуля: {params_module_path}")
        params_module = load_module_from_path("parametres", params_module_path)
        params = params_module.get_params(task_id)
        print(f"Параметры получены (task_id='{task_id}').")

        # 1.2  Загружаем состояние предыдущего запуска с теми же dir + task
        state_path = get_state_path(dir_name, task_id)
        prev_state = load_state(state_path)
        prev_params = deserialize_params(prev_state.get("params", {}))
        prev_dataset_hash = prev_state.get("dataset_hash", None)

        # 1.3  Сравниваем параметры
        if prev_params:
            print("\nСравнение параметров с предыдущим запуском:")
            param_changes = compare_params(prev_params, params)
            print_param_changes(param_changes)
        else:
            print("\nПредыдущий запуск не найден — первое выполнение.")
            param_changes = compare_params({}, params)
            if param_changes:
                print("Все текущие параметры (первый запуск):")
                for key in sorted(param_changes.keys()):
                    val = param_changes[key]["new"]
                    print(f"  {key} = {repr(val)}")

        # Определяем, изменились ли параметры, релевантные для каждого этапа
        rir_params_changed = any(k in param_changes for k in RIR_PARAM_KEYS)
        feats_params_changed = any(k in param_changes for k in RIR_PARAM_KEYS)

        if rir_params_changed:
            changed = [k for k in RIR_PARAM_KEYS if k in param_changes]
            print(f"\nИзменились параметры этапа 1 (генерация отражений): {changed}")

        if feats_params_changed:
            changed = [k for k in FEATS_PARAM_KEYS if k in param_changes]
            print(f"Изменились параметры этапа 2 (расчёт признаков): {changed}")

        # 1.4  Проверяем наличие и изменения файлов датасета
        dataset_dir = params["dataset_dir"]
        current_dataset_hash = compute_dataset_hash(dataset_dir)
        dataset_exists = os.path.isdir(dataset_dir)

        print(f"\nКаталог датасета: {dataset_dir}")
        print(f"  Существует: {'да' if dataset_exists else 'нет'}")

        if dataset_exists:
            file_count = sum(1 for p in Path(dataset_dir).rglob("*") if p.is_file())
            print(f"  Файлов в каталоге: {file_count}")
            print(f"  Хеш содержимого: {current_dataset_hash}")
            if prev_dataset_hash is not None:
                if current_dataset_hash != prev_dataset_hash:
                    print(f"  Датасет ИЗМЕНИЛСЯ с прошлого запуска.")
                    print(f"    предыдущий хеш: {prev_dataset_hash}")
                    dataset_changed = True
                else:
                    print(f"  Датасет не изменялся с прошлого запуска.")
                    dataset_changed = False
            else:
                print(f"  (предыдущий хеш отсутствует)")
                dataset_changed = False
        else:
            print(f"  Каталог не существует.")
            dataset_changed = False

        # ================================================================
        #  ЭТАП 1: Генерация отражений (gpuRIR)
        # ================================================================
        print(f"\n{'=' * 70}")
        print("ЭТАП 1: Генерация отражений (gpuRIR)")
        print(f"{'=' * 70}")

        # Проверяем наличие вложенных каталогов metadata_dev и mic_dev
        metadata_dev_exists = os.path.isdir(os.path.join(dataset_dir, "metadata_dev"))
        mic_dev_exists = os.path.isdir(os.path.join(dataset_dir, "mic_dev"))

        # Условия выполнения этапа 1
        need_rir = (
            rir_params_changed           # изменились параметры, указанные в команде
            or not dataset_exists            # отсутствует папка dataset_dir
            or not metadata_dev_exists       # отсутствует metadata_dev
            or not mic_dev_exists            # отсутствует mic_dev
        )

        if skip_rir:
            print("Пропуск: генерация отражений отключена флагом --skip-rir.")
            rir_executed = False
        elif need_rir:
            # Архивируем существующий каталог датасета, если он есть
            if dataset_exists:
                archived = archive_directory(dataset_dir)
                if archived:
                    print(f"  Существующий каталог переименован (архивирован).")
                    # Обновляем состояние: каталога больше нет
                    dataset_exists = False

            seed_val = params.get("seed", DEFAULT_SEED)
            cmd = [
                sys.executable,
                "generate_gpuRIR_3c.py",
                "--fs", str(params["fs"]),
                "--n-mics", str(params["n_mics"]),
                "--frame-step", str(params["label_hop_len_s"]),
                "--run-sanity-check",
                "--lowpass-hz", str(params.get("lowpass_hz", 0)),
                "--seed", str(seed_val),
                "--duration", str(params["max_audio_len_s"]),
                "--output-dir", str(params["dataset_dir"]),
                "--count", str(count_files),
            ]
            print("Условия выполнения:")
            print(f"  параметры команды изменились: {rir_params_changed}")
            print(f"  dataset_dir существует: {dataset_exists}")
            print(f"  metadata_dev существует: {metadata_dev_exists}")
            print(f"  mic_dev существует: {mic_dev_exists}")

            rc = run_command(cmd, "генерация отражений (gpuRIR)")
            if rc != 0:
                print("\nАВАРИЙНОЕ ЗАВЕРШЕНИЕ: этап 1 завершился с ошибкой.")
                return 1
            rir_executed = True
        else:
            print("Пропуск: условия для выполнения не выполнены.")
            print(f"  параметры команды изменились: {rir_params_changed}")
            print(f"  dataset_dir существует: {dataset_exists}")
            print(f"  metadata_dev существует: {metadata_dev_exists}")
            print(f"  mic_dev существует: {mic_dev_exists}")
            rir_executed = False

        # ================================================================
        #  ЭТАП 2: Расчёт признаков
        # ================================================================
        print(f"\n{'=' * 70}")
        print("ЭТАП 2: Расчёт признаков (batch_feature_extraction)")
        print(f"{'=' * 70}")

        feat_label_dir = params["feat_label_dir"]
        feat_dir_exists = os.path.isdir(feat_label_dir)

        # Условия выполнения этапа 2
        need_feats = (
            rir_executed               # на этапе 2 была выполнена генерация
            or feats_params_changed       # изменились параметры признаков
            or not feat_dir_exists         # отсутствует каталог с признаками
        )

        if skip_feats:
            print("Пропуск: расчёт признаков отключён флагом --skip-feats.")
            feats_executed = False
        elif need_feats:
            cmd = [
                sys.executable,
                os.path.join(dir_name, "batch_feature_extraction.py"),
                task_id,
            ]
            print("Условия выполнения:")
            print(f"  этап 1 был выполнен: {rir_executed}")
            print(f"  параметры признаков изменились: {feats_params_changed}")
            print(f"  feat_label_dir существует: {feat_dir_exists}")
            print(f"  feat_label_dir: {feat_label_dir}")

            rc = run_command(cmd, "расчёт признаков")
            if rc != 0:
                print("\nАВАРИЙНОЕ ЗАВЕРШЕНИЕ: этап 2 завершился с ошибкой.")
                return 1
            feats_executed = True
        else:
            print("Пропуск: условия для выполнения не выполнены.")
            print(f"  этап 1 был выполнен: {rir_executed}")
            print(f"  параметры признаков изменились: {feats_params_changed}")
            print(f"  feat_label_dir существует: {feat_dir_exists}")
            print(f"  feat_label_dir: {feat_label_dir}")
            feats_executed = False

        # ================================================================
        #  ЭТАП 3: Обучение нейросети
        # ================================================================
        print(f"\n{'=' * 70}")
        print("ЭТАП 3: Обучение нейросети (train_seldnet)")
        print(f"{'=' * 70}")

        today_str = datetime.now().strftime("%Y%m%d")
        model_name = f"for_{today_str}"

        cmd = [
            sys.executable,
            os.path.join(dir_name, "train_seldnet.py"),
            task_id,
            model_name,
        ]
        print(f"Имя модели: {model_name}")

        if skip_train:
            print("Пропуск: обучение нейросети отключено флагом --skip-train.")
        else:
            rc = run_command(cmd, "обучение нейросети")
            if rc != 0:
                print("\nАВАРИЙНОЕ ЗАВЕРШЕНИЕ: этап 3 завершился с ошибкой.")
                return 1

        # ================================================================
        #  Сохранение состояния для следующего запуска
        # ================================================================
        print(f"\n{'=' * 70}")
        print("Сохранение состояния")
        print(f"{'=' * 70}")

        # Пересчитываем хеш датасета (после возможной генерации на этапе 2)
        final_dataset_hash = compute_dataset_hash(params["dataset_dir"])
        new_state = {
            "params": serialize_params(params),
            "dataset_hash": final_dataset_hash,
            "last_run": datetime.now().isoformat(),
            "rir_executed": rir_executed,
            "feats_executed": feats_executed,
        }
        save_state(state_path, new_state)
        print(f"Состояние сохранено: {state_path}")
        print(f"  хеш датасета: {final_dataset_hash}")
        print(f"  этап 1 выполнен: {rir_executed}")
        print(f"  этап 2 выполнен: {feats_executed}")

        # ================================================================
        #  Завершение
        # ================================================================
        print(f"\n{'=' * 70}")
        print(f"[maker] Завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")

        return 0

    except FileNotFoundError as e:
        print(f"\n[ОШИБКА] {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n[ФАТАЛЬНАЯ ОШИБКА] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Закрываем файл лога и восстанавливаем stdout/stderr
        if log_fh is not None:
            try:
                log_fh.write(
                    f"\n{'=' * 70}\n"
                    f"[maker] Завершение: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"{'=' * 70}\n"
                )
                log_fh.flush()
            except Exception:
                pass
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            try:
                log_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
