#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Утилита тестирования SELDNet модели.
Формирует отчет с метриками SELD, SED, DOA, Distance и classwise results.

Пример использования:
    python test_seldnet.py --model_path models/best_model.pth --split 6 --argv 6

Аргументы:
    --model_path: Путь к файлу весов модели (.pth)
    --split: Номер сплита для тестирования (например, 6 для dev_split0)
    --argv: Пресет параметров (см. parameters.get_params)
    --dcase_output_dir: Опционально, директория для вывода результатов
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from time import gmtime, strftime
import cls_feature_class
import cls_data_generator
import parameters
import seldnet_model
from model import NGCCModel
from cst_former.CST_former_model import CST_former
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D


def get_model_and_sizes(params, data_gen, device):
    """Загрузка модели на основе параметров."""
    if params['modality'] == 'audio_visual':
        data_in, vid_data_in, data_out = data_gen.get_data_sizes()
    else:
        data_in, data_out = data_gen.get_data_sizes()
        vid_data_in = None

    if params['model'] == 'seldnet':
        model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
    elif params['model'] == 'myseldnet':
        model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
    elif params['model'] == 'ngccmodel':
        model = NGCCModel(data_in, data_out, params, vid_data_in).to(device)
    elif params['model'] == 'cstformer':
        model = CST_former(data_in, data_out, params, vid_data_in).to(device)
    else:
        print('ERROR: Unknown model configuration')
        exit()

    return model, data_in, vid_data_in, data_out


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """Декодирование multi-ACCDOA выходов модели."""
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    dist0[dist0 < 0.] = 0.
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    dist1[dist1 < 0.] = 0.
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 4*nb_classes:7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:]
    dist2[dist2 < 0.] = 0.
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def get_accdoa_labels(accdoa_in, nb_classes):
    """Декодирование single-ACCDOA выходов модели."""
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
    return sed, accdoa_in


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    """Определение схожести локаций для объединения треков."""
    from SELD_evaluation_metrics import distance_between_cartesian_coordinates
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(
                doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]
        ) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device, criterion_tdoa=None):
    """Запуск тестирования и сохранение результатов в DCASE формате."""
    eval_filelist = data_generator.get_filelist()
    model.eval()
    file_cnt = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for values in data_generator.generate():
            if len(values) == 2:  # audio visual
                data, vid_feat = values
                data, vid_feat = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float()
                output = model(data, vid_feat)
            else:
                data = values
                data = torch.tensor(data).to(device).float()
                output = model(data)

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = \
                    get_multi_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                dist_pred0 = reshape_3Dto2D(dist_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # Сохранение результатов в файл
            output_file = os.path.join(dcase_output_folder, eval_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}

            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        flag_0sim1 = determine_similar_location(
                            sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt],
                            doa_pred0[frame_cnt], doa_pred1[frame_cnt],
                            class_cnt, params['thresh_unify'], params['unique_classes']
                        )
                        flag_1sim2 = determine_similar_location(
                            sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt],
                            doa_pred1[frame_cnt], doa_pred2[frame_cnt],
                            class_cnt, params['thresh_unify'], params['unique_classes']
                        )
                        flag_2sim0 = determine_similar_location(
                            sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt],
                            doa_pred2[frame_cnt], doa_pred0[frame_cnt],
                            class_cnt, params['thresh_unify'], params['unique_classes']
                        )

                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([
                                    class_cnt,
                                    doa_pred0[frame_cnt][class_cnt],
                                    doa_pred0[frame_cnt][class_cnt + params['unique_classes']],
                                    doa_pred0[frame_cnt][class_cnt + 2 * params['unique_classes']],
                                    dist_pred0[frame_cnt][class_cnt]
                                ])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([
                                    class_cnt,
                                    doa_pred1[frame_cnt][class_cnt],
                                    doa_pred1[frame_cnt][class_cnt + params['unique_classes']],
                                    doa_pred1[frame_cnt][class_cnt + 2 * params['unique_classes']],
                                    dist_pred1[frame_cnt][class_cnt]
                                ])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([
                                    class_cnt,
                                    doa_pred2[frame_cnt][class_cnt],
                                    doa_pred2[frame_cnt][class_cnt + params['unique_classes']],
                                    doa_pred2[frame_cnt][class_cnt + 2 * params['unique_classes']],
                                    dist_pred2[frame_cnt][class_cnt]
                                ])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5 or sed_pred1[frame_cnt][class_cnt] > 0.5 or sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                doa_pred_new = (
                                    doa_pred0[frame_cnt][class_cnt:class_cnt + 3] * sed_pred0[frame_cnt][class_cnt] +
                                    doa_pred1[frame_cnt][class_cnt:class_cnt + 3] * sed_pred1[frame_cnt][class_cnt] +
                                    doa_pred2[frame_cnt][class_cnt:class_cnt + 3] * sed_pred2[frame_cnt][class_cnt]
                                ) / (sed_pred0[frame_cnt][class_cnt] + sed_pred1[frame_cnt][class_cnt] + sed_pred2[frame_cnt][class_cnt])
                                dist_pred_new = (
                                    dist_pred0[frame_cnt][class_cnt] * sed_pred0[frame_cnt][class_cnt] +
                                    dist_pred1[frame_cnt][class_cnt] * sed_pred1[frame_cnt][class_cnt] +
                                    dist_pred2[frame_cnt][class_cnt] * sed_pred2[frame_cnt][class_cnt]
                                ) / (sed_pred0[frame_cnt][class_cnt] + sed_pred1[frame_cnt][class_cnt] + sed_pred2[frame_cnt][class_cnt])
                                output_dict[frame_cnt].append([
                                    class_cnt,
                                    doa_pred_new[0], doa_pred_new[1], doa_pred_new[2], dist_pred_new
                                ])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5 or sed_pred1[frame_cnt][class_cnt] > 0.5 or sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                doa_pred_new = (
                                    doa_pred0[frame_cnt][class_cnt:class_cnt + 3] +
                                    doa_pred1[frame_cnt][class_cnt:class_cnt + 3] +
                                    doa_pred2[frame_cnt][class_cnt:class_cnt + 3]
                                ) / 3
                                dist_pred_new = (
                                    dist_pred0[frame_cnt][class_cnt] +
                                    dist_pred1[frame_cnt][class_cnt] +
                                    dist_pred2[frame_cnt][class_cnt]
                                ) / 3
                                output_dict[frame_cnt].append([
                                    class_cnt,
                                    doa_pred_new[0], doa_pred_new[1], doa_pred_new[2], dist_pred_new
                                ])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([
                                class_cnt,
                                doa_pred[frame_cnt][class_cnt],
                                doa_pred[frame_cnt][class_cnt + params['unique_classes']],
                                doa_pred[frame_cnt][class_cnt + 2 * params['unique_classes']]
                            ])

            data_gen_test.write_output_format_file(output_file, output_dict)

        print(f"Processed {file_cnt} files")
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='SELDNet Testing Utility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights (.pth file)')
    parser.add_argument('--split', type=int, default=6, help='Test split number (e.g., 6 for dev_split0)')
    parser.add_argument('--argv', type=str, default='6', help='Parameter preset (see parameters.get_params)')
    parser.add_argument('--dcase_output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Установка seed для воспроизводимости
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Загрузка параметров
    params = parameters.get_params(args.argv)

    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Настройка директорий
    if args.dcase_output_dir is None:
        unique_name = f"{args.split}_{strftime('%Y-%m-%d_%H-%M', gmtime())}_dev_split{args.split}"
        if params['multi_accdoa']:
            unique_name += '_multiaccdoa'
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                unique_name += '_mic_salsa'
            else:
                unique_name += '_mic_gcc'
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], f"{unique_name}_{strftime('%Y%m%d%H%M%S', gmtime())}_test")
    else:
        dcase_output_test_folder = args.dcase_output_dir

    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
    print(f"Dumping recording-wise test results in: {dcase_output_test_folder}")

    # Загрузка тестового датасета
    print("Loading unseen test dataset:")
    test_splits = [args.split]
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=test_splits[0], shuffle=False, per_file=True,
    )

    # Создание модели
    model, data_in, vid_data_in, data_out = get_model_and_sizes(params, data_gen_test, device)

    # Загрузка весов
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded model from: {args.model_path}")

    # Инициализация класса для вычисления метрик
    score_obj = ComputeSELDResults(params)

    # Запуск тестирования
    model.eval()
    print("Running inference on test data...")
    test_loss = test_epoch(data_gen_test, model, None, dcase_output_test_folder, params, device)

    # Вычисление метрик
    use_jackknife = True
    test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = \
        score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife)

    # Вывод результатов
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"SELD score (early stopping metric): {test_seld_scr[0]:.2f} [{test_seld_scr[1][0]:.2f}, {test_seld_scr[1][1]:.2f}]")
    print(f"SED metrics: F-score: {100 * test_F[0]:.1f} [{100 * test_F[1][0]:.2f}, {100 * test_F[1][1]:.2f}]")
    print(f"DOA metrics: Angular error: {test_LE[0]:.1f} [{test_LE[1][0]:.2f} , {test_LE[1][1]:.2f}]")
    print(f"Distance metrics: {test_dist_err[0]:.2f} [{test_dist_err[1][0]:.2f} , {test_dist_err[1][1]:.2f}]")
    print(f"Relative Distance metrics: {test_rel_dist_err[0]:.2f} [{test_rel_dist_err[1][0]:.2f} , {test_rel_dist_err[1][1]:.2f}]")

    if params['average'] == 'macro':
        print("\nClasswise results on unseen test data")
        print("Class\tF\tAE\tdist_err\treldist_err\tSELD_score")
        for cls_cnt in range(params['unique_classes']):
            f_val = classwise_test_scr[0][1][cls_cnt]
            f_ci = classwise_test_scr[1][1][cls_cnt]
            ae_val = classwise_test_scr[0][2][cls_cnt]
            ae_ci = classwise_test_scr[1][2][cls_cnt]
            dist_val = classwise_test_scr[0][3][cls_cnt]
            dist_ci = classwise_test_scr[1][3][cls_cnt]
            reldist_val = classwise_test_scr[0][4][cls_cnt]
            reldist_ci = classwise_test_scr[1][4][cls_cnt]
            seld_val = classwise_test_scr[0][6][cls_cnt]
            seld_ci = classwise_test_scr[1][6][cls_cnt]

            print(f"{cls_cnt}\t{f_val:.2f} [{f_ci[0]:.2f}, {f_ci[1]:.2f}]\t"
                  f"{ae_val:.2f} [{ae_ci[0]:.2f}, {ae_ci[1]:.2f}]\t"
                  f"{dist_val:.2f} [{dist_ci[0]:.2f}, {dist_ci[1]:.2f}]\t"
                  f"{reldist_val:.2f} [{reldist_ci[0]:.2f}, {reldist_ci[1]:.2f}]\t"
                  f"{seld_val:.2f} [{seld_ci[0]:.2f}, {seld_ci[1]:.2f}]")

    print("=" * 80)
    print(f"Results saved to: {dcase_output_test_folder}")


if __name__ == "__main__":
    main()
