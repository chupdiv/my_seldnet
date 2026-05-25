
from typing import Any, Dict, List, Optional, Sequence, Tuple


from pathlib import Path
import numpy as np
import csv
import json
import soundfile as sf
import scipy.signal as ss

def convert_to_SELDformat(annotations, mic_center, frame_step):
    # Преобразование к виду SELD
    mic_center = np.asarray(mic_center, dtype=np.float32)
    P = annotations[:,:3] - mic_center
    azi = np.atan2(P[:,1], P[:,0])
    ele = np.atan2(P[:,2], np.hypot(P[:,1], P[:,0]))
    azi_deg = np.round(np.rad2deg(azi) % 360).astype(int)
    ele_deg = np.round(np.clip(np.rad2deg(ele), -90., 90.)).astype(int)
    dist_cm = np.round(np.sqrt(np.sum(P ** 2, axis = 1)) * 100).astype(int)
    time_steps = (annotations[:,3]/frame_step).astype(int)
    lbls = annotations[:,4].astype(int)
    zeroes = np.zeros((P.shape[0],), dtype=int)
    new_annots = np.hstack([
        time_steps[:,None],
        lbls[:,None],
        zeroes[:,None],
        azi_deg[:,None],
        ele_deg[:,None], 
        dist_cm[:,None]
    ])
    return new_annots

def filtering(y, fs, lowpass_hz=None):
    nyq = fs / 2.0
    if lowpass_hz is not None and lowpass_hz > 0 and lowpass_hz < nyq:
        sos = ss.butter(2, lowpass_hz, btype='low', fs=fs, output='sos')
        y = ss.sosfiltfilt(sos, y, axis=0)
    y = np.clip(y, -1.0, 1.0)
    return y


def write_annotations(fname, annotations):
    with open(fname, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(annotations)
    return None

def write_soundfile(fname, y, fs):
    sf.write(fname, y, fs, format="WAV", subtype="FLOAT")

def write_plan(task_type, plan, cfg):
    fname = cfg['output_dir'] / f'{task_type}_plan.json'
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=4, default=str)


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """Создать выходные директории (audio_dir, meta_dir и сплиты), если их нет."""
    for d in ("audio_dir", "meta_dir"):
        path = cfg[d]
        if isinstance(path, (str, Path)):
            Path(path).mkdir(parents=True, exist_ok=True)
    # Сплиты
    for split in ("dev-train", "dev-test"):
        (cfg["audio_dir"] / split).mkdir(parents=True, exist_ok=True)
        (cfg["meta_dir"] / split).mkdir(parents=True, exist_ok=True)


def make_output_filenames(num, sample, sample_type, cfg):
    if sample_type == 'test':
        split = "dev-test"
        fold = "fold4"
    else:
        split = "dev-train"
        fold = "fold3"

    sample_classes = '-'.join([lbl for lbl,c in cfg['classes_configs'].items() if sample[c['index']] ])
    wav_fname = cfg['audio_dir'] / split / f'{fold}_{sample_classes}_{num}.wav'
    annot_fname = cfg['meta_dir'] / split / f'{fold}_{sample_classes}_{num}.csv'
    return wav_fname, annot_fname

def run_sanity_check(cfg):
    if cfg.get('run_sanity_check', False) == False:
        return None
    
    """Validate generated dataset structure and basic metadata quality."""
    print("\n=== SANITY CHECK START ===")
    splits = ["dev-train", "dev-test"]
    total_wavs = 0
    total_csvs = 0
    total_rows = 0
    classes_seen = set()
    errors = []

    for split in splits:
        wav_dir = cfg['audio_dir'] / split
        csv_dir = cfg['meta_dir'] / split
        wav_files = sorted(wav_dir.glob("*.wav"))
        csv_files = sorted(csv_dir.glob("*.csv"))
        total_wavs += len(wav_files)
        total_csvs += len(csv_files)

        wav_stems = {p.stem for p in wav_files}
        csv_stems = {p.stem for p in csv_files}
        missing_csv = sorted(wav_stems - csv_stems)
        missing_wav = sorted(csv_stems - wav_stems)
        if missing_csv:
            errors.append(f"[{split}] missing csv for wav: {len(missing_csv)}")
        if missing_wav:
            errors.append(f"[{split}] missing wav for csv: {len(missing_wav)}")

        for wav_path in wav_files:
            info = sf.info(str(wav_path))
            if info.samplerate != cfg['fs']:
                errors.append(f"[{split}] bad fs in {wav_path.name}: {info.samplerate}")
            if info.channels != cfg['n_mics']:
                errors.append(f"[{split}] bad channels in {wav_path.name}: {info.channels}")

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    total_rows += 1
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) != 6:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} expected 6 columns, got {len(parts)}")
                        continue
                    try:
                        frame_i = int(parts[0])
                        class_i = int(parts[1])
                        _track_i = int(parts[2])
                        azi_i = int(parts[3])
                        ele_i = int(parts[4])
                        dist_i = int(parts[5])
                    except ValueError:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} has non-integer fields")
                        continue

                    classes_seen.add(class_i)
                    if frame_i < 0:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} negative frame")
                    if not (0 <= azi_i <= 360):
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} azimuth out of range")
                    if not (-90 <= ele_i <= 90):
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} elevation out of range")
                    if dist_i < 0:
                        errors.append(f"[{split}] {csv_path.name}:{line_idx} negative distance")

    output_dir = cfg['output_dir']
    print(f"Output dir: {output_dir}")
    print(f"WAV files: {total_wavs}, CSV files: {total_csvs}, annotation rows: {total_rows}")
    print(f"Classes seen: {sorted(classes_seen)}")

    if errors:
        print("SANITY CHECK FAILED")
        print(f"Errors: {len(errors)}")
        for err in errors[:30]:
            print(" -", err)
        if len(errors) > 30:
            print(f" ... and {len(errors) - 30} more")
    else:
        print("SANITY CHECK PASSED")
    print("=== SANITY CHECK END ===\n")
