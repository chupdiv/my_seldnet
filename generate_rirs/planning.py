"""
planning.py
~~~~~~~~

Планирование состава звуковой сцены
"""


from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from scene import sample_scene_acoustics
def classes_choice(
    plan: List[Tuple[list, float]],
    not_empty: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Формирует случайный выбор классов, которые будут представлены в сцене.

    Parameters
    ----------
    plan : list[tuple[list[int], float]]
        Распределение вида ``[([0], 0.7), ([1, 2], 0.5)]``, где:
        - ``list[int]`` — группа классов (внутри группы выбирается один представитель),
        - ``float``     — вероятность того, что группа будет представлена.
    not_empty : bool
        Если ``True`` и ни один класс не выбран — принудительно выбирается один случайный.
    rng : numpy.random.Generator | None

    Returns
    -------
    np.ndarray (n_classes,), bool
        Вектор-маска: ``True`` на позиции выбранных классов.

    Example
    -------
    >>> plan = [([0], 0.9), ([1, 2], 0.5)]
    >>> sample = classes_choice(plan, rng=rng)
    >>> # sample = [ True, False,  True ]  → дрон + hifitts
    """
    if rng is None:
        rng = np.random.default_rng()

    groups, probabilities = list(zip(*plan))

    # Выбор групп, которые будут представлены в сцене
    mask = rng.binomial(n=1, p=probabilities)

    # Выбор одного класса-представителя из каждой группы
    examples = np.array([rng.choice(gr) for gr in groups])
    examples = examples[:, None]

    n_groups = len(groups)
    n_classes = max((idx for gr in groups for idx in gr), default=-1) + 1

    # Матрица распределения по классам
    T = np.zeros((n_groups, n_classes), dtype=bool)
    row_idx = np.repeat(np.arange(n_groups), [len(sub) for sub in examples])
    col_idx = np.concatenate(examples)
    T[row_idx, col_idx] = True
    sample = mask @ T

    # Если ни один класс не попал — принудительно выбираем случайный
    if (not sample.max()) and not_empty:
        i = rng.choice(range(n_classes))
        sample[i] = True

    return sample.astype(bool)


def levels_choice(
    sample: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Сгенерировать относительные уровни громкости для выбранных классов.

    Для выбранных классов (``sample == True``) формируются веса в соответствии с распределением Дирихле
    (равномерное распределение долей), затем масштабируются  на случайный коэффициент из ``[low, high]``.

    Parameters
    ----------
    sample : np.ndarray (n_classes,), bool
        Маска выбранных классов (выход :func:`classes_choice`).
    low : float
        Минимальный общий масштаб (0.0–1.0).
    high : float
        Максимальный общий масштаб (0.0–1.0).
    rng : numpy.random.Generator | None

    Returns
    -------
    np.ndarray (n_classes,), float
        Вектор весов: ``> 0`` для выбранных классов, ``0.0`` для остальных.

    Raises
    ------
    ValueError
        Если параметры вне диапазона [0, 1] или ``high < low``.

    Example
    -------
    >>> sample = np.array([True, False, True])
    >>> weights = levels_choice(sample, low=0.3, high=0.9, rng=rng)
    >>> # weights = [0.55, 0.0, 0.35]  — дрон громче hifitts
    """
    if rng is None:
        rng = np.random.default_rng()

    low = float(low)
    high = float(high)

    if (low < 0.0) or (low > 1.0):
        raise ValueError(
            f"Параметр low должен принадлежать отрезку [0.0, 1.0]. Сейчас low={low}"
        )
    if (high < 0.0) or (high > 1.0):
        raise ValueError(
            f"Параметр high должен принадлежать отрезку [0.0, 1.0]. Сейчас high={high}"
        )
    if high < low:
        raise ValueError(
            f"Параметр high должен быть >= low. Сейчас high={high}, low={low}"
        )

    probs = np.zeros_like(sample, dtype=float)
    n = np.count_nonzero(sample)
    if n != 0:
        probs[sample.astype(bool)] = rng.dirichlet(np.ones(n))

    level = rng.uniform(low=low, high=high)
    return level * probs

def get_dataset(plan, count, cfg, rng=None):
    dataset = []
    for i in range(count):
        sample = classes_choice(
            plan = plan,
            not_empty = True,
            rng = rng,
        ) 
        levels = levels_choice(
            sample = sample,
            low = 0.0,  # cfg['float_wav_peak_target_min']
            high = 1.0, # cfg['float_wav_peak_target_max'] 
            rng = rng,
        )
        scene = sample_scene_acoustics(
            cfg=cfg,
            rng=rng, 
        )
        P = {'sample':sample, 
             'levels':levels,
             'scene':scene,
            }
        
        dataset.append(P)

    return dataset


