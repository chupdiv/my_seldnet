Сборшик maker.py получает следующие ключи:

--dir <dir> SeldNet_3classes_80 или  SeldNet_3classes_200
--task <task_id>
-- log <log-file>

Выполняет следующие действия:
Этап 1:
1. Вызывает функцию params = get_params(<task_id>) из <dir>/parametres.py
2. Порверяет, какие параметры params изменились с прошлого запускас теми же аргументами <dir> - <task_id>.
3. Проверяет, наличие файлов в каталоге, указанном в параметре dataset_dir и изменились ли они с прошлого запуска с теми же 2. Порверяет, какие параметры params изменились с прошлого запуска с теми же аргументами <dir> - <task_id>.
 <dir> - <task_id>.
4. Выводит в консоль отчет об изменениях параметров.

Этап 2. Выполняется генерация отражений следующей командой:
python generate_gpuRIR_3c.py --fs params['fs] --n-mics params['n_mics']> --frame-step params['label_hop_len_s] --run-sanity-check --lowpass-hz params['lowpass-hz'] --seed params['seed']--duration params['max_audio_len_s']

Если параметр params['lowpass-hz'] не существует, то --lowpass-hz 0
Команда выполняется только в следующих случаях:
- изменились значения параметров, указанных в команде
- отсутсвует папка params['dataset_dir'] 
- отсутсвуют вложенные каталоги metadata_dev и mic_dev

Этап 3. Выполняется расчет признаков командой 
python <dir>/batch_feature_extraction.py <task_id>

Команда выполняется в следующих случаях:
- На этапе 2 была вызвана команда генерации признаков
- Изменились значения параметров params['hop_len_s'],  params['nb_mel_bins'],    params['mel_fmin_hz'],    params['mel_fmax_hz'],    params['use_salsalite'],    params['raw_chunks'],    params['saved_chunks=False'],    params['fmin_doa_salsalite=50'],    params['fmax_doa_salsalite=2000'],    params['fmax_spectra_salsalite=5000']
- отсутствует каталог, указанный в  params['feat_label_dir']

Этап 4. Выполняется обучение нейросети
для этого 
  python <dir>/train_seldnet.py <task_id> <task_id>-<YYYYMMDD> 
  где <YYYYMMDD>  текущая дата

Программа сохраняет весь консольный вывод в <log-file>, если он передан в качестве параметра




