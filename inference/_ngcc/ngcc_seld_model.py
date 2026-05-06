import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NGCCPHAT


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class NGCCModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.ngcc_channels = params["ngcc_channels"]
        self.ngcc_out_channels = params["ngcc_out_channels"]
        self.mel_bins = params["nb_mel_bins"]
        self.fs = params["fs"]
        self.sig_len = int(self.fs * params["hop_len_s"])
        self.predict_tdoa = params["predict_tdoa"]
        if params["use_mel"]:
            self.in_channels = int(self.ngcc_out_channels * params["n_mics"] * (params["n_mics"] - 1) / 2 + params["n_mics"])
        else:
            self.in_channels = int(self.ngcc_out_channels * params["n_mics"] * (1 + (params["n_mics"] - 1) / 2))

        self.ngcc = NGCCPHAT(
            max_tau=params["max_tau"],
            n_mel_bins=self.mel_bins,
            use_sinc=True,
            sig_len=self.sig_len,
            num_channels=self.ngcc_channels,
            num_out_channels=self.ngcc_out_channels,
            fs=self.fs,
            normalize_input=False,
            normalize_output=False,
            pool_len=1,
            use_mel=params["use_mel"],
            use_mfcc=params["use_mfcc"],
            predict_tdoa=params["predict_tdoa"],
            tracks=params["tracks"],
            fixed_tdoa=params["fixed_tdoa"],
        )

        self.params = params
        self.conv_block_list = nn.ModuleList()
        for conv_cnt in range(len(params["f_pool_size"])):
            self.conv_block_list.append(ConvBlock(in_channels=params["nb_cnn2d_filt"] if conv_cnt else self.in_channels, out_channels=params["nb_cnn2d_filt"]))
            self.conv_block_list.append(nn.MaxPool2d((params["t_pool_size"][conv_cnt], params["f_pool_size"][conv_cnt])))
            self.conv_block_list.append(nn.Dropout2d(p=params["dropout_rate"]))

        self.gru_input_dim = params["nb_cnn2d_filt"] * int(np.floor(self.mel_bins / np.prod(params["f_pool_size"])))
        self.gru = torch.nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params["rnn_size"],
            num_layers=params["nb_rnn_layers"],
            batch_first=True,
            dropout=params["dropout_rate"],
            bidirectional=True,
        )
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params["nb_self_attn_layers"]):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=self.params["rnn_size"],
                    num_heads=self.params["nb_heads"],
                    dropout=self.params["dropout_rate"],
                    batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(self.params["rnn_size"]))

        self.fnn_list = torch.nn.ModuleList()
        if params["nb_fnn_layers"]:
            for fc_cnt in range(params["nb_fnn_layers"]):
                self.fnn_list.append(nn.Linear(params["fnn_size"] if fc_cnt else self.params["rnn_size"], params["fnn_size"], bias=True))
        self.fnn_list.append(nn.Linear(params["fnn_size"] if params["nb_fnn_layers"] else self.params["rnn_size"], out_shape[-1], bias=True))

    def forward(self, x, vid_feat=None):
        if self.predict_tdoa:
            x, tdoa = self.ngcc(x)
        else:
            x = self.ngcc(x)
            tdoa = None

        for layer in self.conv_block_list:
            x = layer(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]

        for idx in range(len(self.mhsa_block_list)):
            x_in = x
            x, _ = self.mhsa_block_list[idx](x_in, x_in, x_in)
            x = self.layer_norm_list[idx](x + x_in)

        for fnn in self.fnn_list[:-1]:
            x = fnn(x)
        doa = self.fnn_list[-1](x)
        if self.predict_tdoa:
            return doa, tdoa.mean(dim=1, keepdims=True)
        return doa

