# Spatial (T, F) sizes after conv_encoder MaxPool stack — used for ULE patch/fold.

import numpy as np


def _salsa_nb_mel_bins(params):
    """Same count as cls_feature_class.FeatureClass SALSA branch: cutoff_bin - lower_bin."""
    fs = int(params['fs'])
    hop_len = int(fs * float(params['hop_len_s']))
    if params.get('raw_chunks'):
        win_len = hop_len
    else:
        win_len = 2 * hop_len
    nfft = 2 ** (win_len - 1).bit_length()
    lower_bin = int(np.floor(params['fmin_doa_salsalite'] * nfft / float(fs)))
    lower_bin = max(1, lower_bin)
    cutoff_bin = int(np.floor(params['fmax_spectra_salsalite'] * nfft / float(fs)))
    return cutoff_bin - lower_bin


def freq_bins_input(params):
    if params.get('use_salsalite'):
        return _salsa_nb_mel_bins(params)
    return int(params['nb_mel_bins'])


def encoder_output_tf(params):
    """
    Mirrors conv_encoder (encoder.py): temporal and frequency size after all CNN+pool layers.
    """
    t_pooling_loc = params['t_pooling_loc']
    f_pool_size = params['f_pool_size']
    t_pool_size = params['t_pool_size']
    t = int(params['feature_sequence_length'])
    f = freq_bins_input(params)
    n = len(f_pool_size)
    for i in range(n):
        tp = t_pool_size[i] if t_pooling_loc == 'front' else 1
        fp = f_pool_size[i]
        t = t // tp if tp else t
        f = f // fp if fp else f
    return t, f


def choose_ule_patch_sizes(t_enc, f_enc, prefer_t=10, prefer_f=4):
    """
    Pick divisors of T_enc and F_enc close to DCASE defaults (10, 4).
    Ensures unfold/fold cover the full map without padding.
    """
    if t_enc < 1 or f_enc < 1:
        raise ValueError(
            'encoder_output is too small (T={}, F={}); relax pooling or use longer input'.format(
                t_enc, f_enc
            )
        )

    def _best_divisor(n, prefer):
        best = 1
        best_dist = abs(1 - prefer)
        for d in range(1, n + 1):
            if n % d != 0:
                continue
            dist = abs(d - prefer)
            if dist < best_dist or (dist == best_dist and d > best):
                best = d
                best_dist = dist
        return best

    pt = _best_divisor(t_enc, prefer_t)
    pf = _best_divisor(f_enc, prefer_f)
    return pt, pf


def ule_mha_num_heads(embed_dim, max_heads):
    """MultiheadAttention requires embed_dim % num_heads == 0."""
    max_heads = min(max_heads, embed_dim)
    for h in range(max_heads, 0, -1):
        if embed_dim % h == 0:
            return h
    return 1
