"""
NGCC model implementation for SELD inference.
Extracted from SeldNet_3classes_80/ngcc/
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft
import torchaudio


def get_pad(size, kernel_size=3, stride=1, dilation=1):
    """Вычисляет padding для сохранения размера при свёртке."""
    import collections.abc as collections
    
    def _calc_pad(size, kernel_size=3, stride=1, dilation=1):
        pad = (((size + stride - 1) // stride - 1) * stride + kernel_size - size) * dilation
        return pad // 2, pad - pad // 2
    
    def _get_compressed(item, index):
        if isinstance(item, collections.Sequence):
            return item[index]
        return item
    
    len_size = 1
    if isinstance(size, collections.Sequence):
        len_size = len(size)
    pad = ()
    for i in range(len_size):
        pad = _calc_pad(
            size=_get_compressed(size, i),
            kernel_size=_get_compressed(kernel_size, i),
            stride=_get_compressed(stride, i),
            dilation=_get_compressed(dilation, i)
        ) + pad
    return pad


def next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band, t_right):
    y_right = torch.sin(2*np.pi*band*t_right)/(2*np.pi*band*t_right)
    y_left = flip(y_right, 0)
    y = torch.cat([y_left, torch.ones(1).to(y_right.device), y_right])
    return y


class SincConv_fast(nn.Module):
    """Sinc-based convolution"""
    
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only support one input channel (here, in_channels = {in_channels}).")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2*np.pi*n_lin/self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*np.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_/2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, bias=None, groups=1)


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "tanh":
        return nn.Tanh()
    if act_type == "sigmoid":
        return nn.Sigmoid()
    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if act_type == "elu":
        return nn.ELU()
    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)
    if act_type == "linear":
        return nn.LeakyReLU(1)
    raise ValueError(f"Unknown activation type: {act_type}")


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SincNet(nn.Module):
    def __init__(self, options):
        super(SincNet, self).__init__()

        self.cnn_N_filt = options['cnn_N_filt']
        self.cnn_len_filt = options['cnn_len_filt']
        self.cnn_max_pool_len = options['cnn_max_pool_len']
        self.cnn_act = options['cnn_act']
        self.cnn_drop = options['cnn_drop']
        self.cnn_use_laynorm = options['cnn_use_laynorm']
        self.cnn_use_batchnorm = options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp = options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp = options['cnn_use_batchnorm_inp']
        self.input_dim = int(options['input_dim'])
        self.fs = options['fs']
        self.N_cnn_lay = len(options['cnn_N_filt'])
        
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        self.use_sinc = options['use_sinc']

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):
            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
            self.act.append(act_fun(self.cnn_act[i]))
            self.bn.append(nn.BatchNorm1d(N_filt, momentum=0.05))

            if i == 0:
                if self.use_sinc:
                    self.conv.append(SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))
                else:
                    self.conv.append(nn.Conv1d(1, self.cnn_N_filt[i], self.cnn_len_filt[i]))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[-1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0(x)

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0(x)

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):
            s = x.shape[2]
            padding = get_pad(size=s, kernel_size=self.cnn_len_filt[i], stride=1, dilation=1)
            x = F.pad(x, pad=padding, mode='circular')

            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))
                else:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        return x


class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt='phat', epsilon=0.001, beta=None):
        super().__init__()
        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y):
        n = x.shape[-1] + y.shape[-1]

        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        if self.filt == 'phat':
            phi = 1 / (torch.abs(Gxy) + self.epsilon)
        elif self.filt == 'roth':
            phi = 1 / (X * torch.conj(X) + self.epsilon)
        elif self.filt == 'scot':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            phi = 1 / (torch.sqrt(Gxx * Gyy) + self.epsilon)
        elif self.filt == 'ht':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            gamma = Gxy / torch.sqrt(Gxx * Gxy)
            phi = torch.abs(gamma)**2 / (torch.abs(Gxy) * (1 - gamma)**2 + self.epsilon)
        elif self.filt == 'cc':
            phi = 1.0
        else:
            raise ValueError('Unsupported filter function')

        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(Gxy * torch.pow(phi, self.beta[i]), n))
            cc = torch.cat(cc, dim=1)
        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, :max_shift+1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat((cc[:, :, -max_shift:], cc[:, :, :max_shift+1]), dim=-1)
        elif self.dim == 4:
            cc = torch.cat((cc[:, :, :, -max_shift:], cc[:, :, :, :max_shift+1]), dim=-1)

        return cc


class NGCC_model(nn.Module):
    def __init__(self, max_tau=64, n_mel_bins=64, use_sinc=True, sig_len=960, 
                 num_channels=128, num_out_channels=8, fs=24000, normalize_input=True, 
                 normalize_output=False, pool_len=5, use_mel=True, use_mfcc=False, 
                 tracks=5, predict_tdoa=False, fixed_tdoa=False):
        super().__init__()

        self.max_tau = max_tau
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.pool_len = pool_len
        self.n_mel_bins = n_mel_bins
        self.use_mel = use_mel
        self.use_mfcc = use_mfcc
        self.tracks = tracks
        self.predict_tdoa = predict_tdoa
        self.fixed_tdoa = fixed_tdoa 

        sincnet_params = {
            'input_dim': sig_len,
            'fs': fs,
            'cnn_N_filt': [num_channels, num_channels, num_channels, num_channels],
            'cnn_len_filt': [sig_len-1, 11, 9, 7],
            'cnn_max_pool_len': [1, 1, 1, 1],
            'cnn_use_laynorm_inp': False,
            'cnn_use_batchnorm_inp': False,
            'cnn_use_laynorm': [False, False, False, False],
            'cnn_use_batchnorm': [True, True, True, True],
            'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
            'cnn_drop': [0.0, 0.0, 0.0, 0.0],
            'use_sinc': use_sinc,
        } 

        self.backbone = SincNet(sincnet_params)
        self.pool = torch.nn.AvgPool2d((pool_len, 1))
        self.mlp_kernels = [11, 9, 7]
        self.channels = [num_channels, num_channels, num_channels, num_channels]
        self.final_kernel = 3

        self.gcc = GCC(max_tau=self.max_tau, dim=4, filt='phat')

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size=k),
                nn.BatchNorm1d(self.channels[i+1]),
                nn.LeakyReLU(0.2)
            ) for i, k in enumerate(self.mlp_kernels)
        ])

        self.final_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel),
            nn.BatchNorm1d(num_out_channels),
            nn.LeakyReLU(0.2)
        )

        if self.predict_tdoa:
            self.tdoa_conv = nn.Conv1d(num_out_channels, tracks, kernel_size=self.final_kernel)

        self.spec_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel, stride=self.final_kernel),
            nn.BatchNorm1d(num_out_channels),
            nn.GELU()
        )
        
        self.cc_proj = nn.Sequential(
            nn.Linear(max_tau*2+1, self.n_mel_bins // 2),
            nn.LayerNorm(self.n_mel_bins // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.n_mel_bins // 2, self.n_mel_bins)
        )

        if self.use_mel:
            self.nfft = next_greater_power_of_2(2 * sig_len)
            self.spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.nfft, win_length=2*sig_len, hop_length=sig_len, normalized=True
            )
            self.mel_transform = torchaudio.transforms.MelScale(
                n_mels=self.n_mel_bins, sample_rate=fs, n_stft=self.nfft//2+1, norm='slaney'
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
            if self.use_mfcc:
                melkwargs = {
                    "n_fft": self.nfft, "win_length": 2*sig_len, "power": 1,
                    "hop_length": sig_len, "n_mels": 80, "f_min": 20, "f_max": 7000
                }
                self.mfcc = torchaudio.transforms.MFCC(
                    sample_rate=fs, n_mfcc=n_mel_bins, log_mels=True, melkwargs=melkwargs
                )
        else:
            in_size = sig_len // self.final_kernel
            self.proj = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_size, in_size // 2),
                nn.LayerNorm(in_size // 2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_size // 2, self.n_mel_bins)
            )

    def forward(self, audio):
        if self.normalize_input:
            audio /= audio.std(dim=-1, keepdims=True)

        with torch.set_grad_enabled(not self.fixed_tdoa):
            B, M, T, L = audio.shape
            x = audio.reshape(-1, 1, T*L)
            x = self.backbone(x)

            _, C, _ = x.shape
            L_spec = int(L // self.final_kernel)
            x_cc = x.reshape(B, M, C, T*L)
            x_cc = x.reshape(B, M, C, T, L).permute(0, 1, 3, 2, 4)

        if not self.use_mel:
            x_spec = self.spec_conv(x)
            _, C_spec, _ = x_spec.shape
            x_spec = x_spec.reshape(B, M, C_spec, T*L_spec)
            x_spec = x_spec.reshape(B, M, C_spec, T, L_spec).permute(0, 1, 3, 2, 4)

        with torch.set_grad_enabled(not self.fixed_tdoa):
            cc = [] 
            for m1 in range(0, M):
                for m2 in range(m1+1, M):
                    y1 = x_cc[:, m1, :, :, :]
                    y2 = x_cc[:, m2, :, :, :]
                    cc1 = self.gcc(y1, y2)
                    cc.append(cc1)

            cc = torch.stack(cc, dim=-1)
            cc = cc.permute(0, 4, 1, 2, 3)

            B, N, _, C, tau = cc.shape
            cc = cc.reshape(-1, C, tau)
            for k, layer in enumerate(self.mlp):
                s = cc.shape[2]
                padding = get_pad(size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
                cc = F.pad(cc, pad=padding, mode='constant')
                cc = layer(cc)

            s = cc.shape[2]
            padding = get_pad(size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
            cc = F.pad(cc, pad=padding, mode='constant')
            cc = self.final_conv(cc)

            if self.predict_tdoa:
                s = cc.shape[2]
                padding = get_pad(size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
                cc_out = F.pad(cc, pad=padding, mode='constant')
                cc_out = self.tdoa_conv(cc_out)
                _, C, tau = cc_out.shape
                cc_out = cc_out.reshape(B, N, T, self.tracks, tau)
                cc_out = cc_out.permute(0, 2, 4, 3, 1)

        _, C, tau = cc.shape
        cc = cc.reshape(B, N, T, C, tau)
        cc = cc.permute(0, 1, 3, 2, 4)
        cc = cc.reshape(B, N * C, T, tau)
        cc = self.cc_proj(cc)

        if self.normalize_output:
            cc /= cc.std(dim=-1, keepdims=True)

        if self.use_mel:
            B, M, T, L = audio.shape
            audio_in = audio.reshape(B, M, T*L)
            if self.use_mfcc:
                mel_spectra = self.mfcc(audio_in)[:, :, :, :T]
            else:
                mag_spectra = self.spec_transform(audio_in)[:, :, :, :T]
                mel_spectra = self.mel_transform(mag_spectra)
                mel_spectra = self.to_db(mel_spectra)
            
            mel_spectra = mel_spectra.permute(0, 1, 3, 2)
            feat = torch.cat((mel_spectra, cc), dim=1)
        else:
            x_spec = x_spec.permute(0, 1, 3, 2, 4)
            x_spec = x_spec.reshape(B, M * C_spec, T, L_spec)
            mel_spectra = self.proj(x_spec)
            feat = torch.cat((mel_spectra, cc), dim=1)

        feat = self.pool(feat)

        if self.predict_tdoa:
            return feat, cc_out
        else:
            return feat
