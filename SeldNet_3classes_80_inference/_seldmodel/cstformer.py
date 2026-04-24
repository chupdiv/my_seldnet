"""
Модули для модели CSTFormer.
Код скопирован из SeldNet_3classes_80/cst_former/ для обеспечения автономности.
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


# =============================================================================
# cst_geometry.py
# =============================================================================

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


# =============================================================================
# layers.py - Convolutional Blocks и другие слои
# =============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def make_pairs(x):
    """make the int -> tuple"""
    return x if isinstance(x, tuple) else (x, x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class ConvBlockTwo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_init):
        identity = x_init.clone()
        x = F.relu(self.bn1(self.conv1(x_init)))
        x = F.relu(self.bn2(self.conv2(x)) + identity)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_MSCAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_MSCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.se2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c, 1, 1)
        y1 = self.se1(y1).view(b, c, 1, 1)
        y2 = self.se2(x)
        y = y1.expand_as(y2) + y2
        y = self.activation(y)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 *, reduction=16, MSCAM=False):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if not MSCAM:
            self.se = SELayer(out_channels, reduction)
        else:
            self.se = SE_MSCAM(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class LocalPerceptionUint(torch.nn.Module):
    def __init__(self, dim, act=False):
        super(LocalPerceptionUint, self).__init__()
        self.act = act
        self.conv_3x3_dw = ConvDW3x3(dim)
        if self.act:
            self.actation = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.act:
            out = self.actation(self.conv_3x3_dw(x))
            return out
        else:
            out = self.conv_3x3_dw(x)
            return out


class InvertedResidualFeedForward(torch.nn.Module):
    def __init__(self, dim, dim_ratio=4.):
        super(InvertedResidualFeedForward, self).__init__()
        output_dim = int(dim_ratio * dim)
        self.conv1x1_gelu_bn = ConvGeluBN(
            in_channel=dim,
            out_channel=output_dim,
            kernel_size=1,
            stride_size=1,
            padding=0
        )
        self.conv3x3_dw = ConvDW3x3(dim=output_dim)
        self.act = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(output_dim)
        )
        self.conv1x1_pw = nn.Sequential(
            nn.Conv2d(output_dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv1x1_gelu_bn(x)
        out = x + self.act(self.conv3x3_dw(x))
        out = self.conv1x1_pw(out)
        return out


class ConvDW3x3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=make_pairs(kernel_size),
            padding=make_pairs(1),
            groups=dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvGeluBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
        """build the conv3x3 + gelu + bn module"""
        super(ConvGeluBN, self).__init__()
        self.kernel_size = make_pairs(kernel_size)
        self.stride_size = make_pairs(stride_size)
        self.padding_size = make_pairs(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv3x3_gelu_bn = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.kernel_size,
                      stride=self.stride_size,
                      padding=self.padding_size),
            nn.GELU(),
            nn.BatchNorm2d(self.out_channel)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv3x3_gelu_bn(x)
        return x


class FC_layer(torch.nn.Module):
    """
    Fully Connected layer for baseline

    Args:
        out_shape (int): output shape for SLED
                         ex. 39 for single-ACCDOA, 117 for multi-ACCDOA
        temp_embed_dim (int): the input size
        params : parameters from parameter.py
    """
    def __init__(self, out_shape, temp_embed_dim, params):
        super().__init__()

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(params['fnn_size'] if fc_cnt else temp_embed_dim, params['fnn_size'], bias=True))
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else temp_embed_dim, out_shape[-1],
                      bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)

        doa = self.fnn_list[-1](x)

        doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 3)
        doa = doa.reshape((doa.size(0), doa.size(1), -1))
        return doa


# =============================================================================
# encoder.py
# =============================================================================

class conv_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params["t_pooling_loc"]
        assert (len(params['f_pool_size']))

        self.conv_block_list = nn.ModuleList()

        if self.params['ChAtten_DCA']:
            in_channels = 1
        else:
            in_channels = params['nb_channels']

        for conv_cnt in range(len(params['f_pool_size'])):
            self.conv_block_list.append(nn.Sequential(
                ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_channels,
                          out_channels=params['nb_cnn2d_filt']),
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt])),
                nn.Dropout2d(p=params['dropout_rate']),
            ))

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        return x


class resnet_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ratio = [1, 2, 4, 8]
        assert (len(params['f_pool_size']))

        self.res_block_list = nn.ModuleList()
        self.first_conv = ConvBlock(in_channels=in_feat_shape[1],
                                    out_channels=params['nb_resnet_filt'])

        self.last_conv = ConvBlock(in_channels=params['nb_resnet_filt'] * self.ratio[-1],
                                   out_channels=params['nb_cnn2d_filt'])

        for conv_cnt in range(len(params['f_pool_size'])):
            self.res_block_list.append(nn.Sequential(
                ResidualBlock(in_channels=params['nb_resnet_filt'],
                              out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt]) if not conv_cnt else
                ConvBlockTwo(in_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt - 1],
                             out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt]),
                ResidualBlock(in_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt],
                              out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt]),
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt]))
            ))

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.first_conv(x)
        for conv_cnt in range(len(self.res_block_list)):
            x = self.res_block_list[conv_cnt](x)
        x = self.last_conv(x)
        return x


class senet_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.baseline = params['baseline']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ratio = [1, 2, 4, 8]
        assert (len(params['f_pool_size']))

        self.res_block_list = nn.ModuleList()
        self.first_conv = ConvBlock(in_channels=in_feat_shape[1],
                                    out_channels=params['nb_resnet_filt'])

        if not self.baseline:
            self.last_conv = ConvBlock(in_channels=params['nb_resnet_filt'] * self.ratio[-1],
                                       out_channels=params['nb_cnn2d_filt'])

        for conv_cnt in range(len(params['f_pool_size'])):
            self.res_block_list.append(nn.Sequential(
                SEBasicBlock(in_channels=params['nb_resnet_filt'],
                             out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt],
                             MSCAM=params['MSCAM']) if not conv_cnt else
                ConvBlockTwo(in_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt - 1],
                             out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt]),
                SEBasicBlock(in_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt],
                             out_channels=params['nb_resnet_filt'] * self.ratio[conv_cnt],
                             MSCAM=params['MSCAM']),
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt]))
            ))

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.first_conv(x)
        for conv_cnt in range(len(self.res_block_list)):
            x = self.res_block_list[conv_cnt](x)
        if not self.baseline:
            x = self.last_conv(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()

        if params['encoder'] == 'ResNet':
            self.encoder = resnet_encoder(in_feat_shape, params)
        elif params['encoder'] == 'conv':
            self.encoder = conv_encoder(in_feat_shape, params)
        elif params['encoder'] == 'SENet':
            self.encoder = senet_encoder(in_feat_shape, params)

    def forward(self, x):
        x = self.encoder(x)
        return x


# =============================================================================
# Spec_attention и Temp_attention (из layers.py)
# =============================================================================

class Spec_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.dropout_rate = params['dropout_rate']
        self.linear_layer = params['linear_layer']
        self.temp_embed_dim = temp_embed_dim
        self.sp_attn_embed_dim = params['nb_cnn2d_filt']
        self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
                                             dropout=params['dropout_rate'], batch_first=True)
        self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if params.get('LinearLayer', False):
            self.sp_linear = nn.Linear(self.sp_attn_embed_dim, self.sp_attn_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, C, T, F):
        x_init = x
        x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', c=C, f=F).contiguous()
        xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
        xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
        if self.linear_layer:
            xs = self.activation(self.sp_linear(xs))
        xs = xs + x_init
        if self.dropout_rate:
            xs = self.drop_out(xs)
        x_out = self.sp_layer_norm(xs)
        return x_out


class Temp_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.dropout_rate = params['dropout_rate']
        self.linear_layer = params['linear_layer']
        self.temp_embed_dim = temp_embed_dim
        self.embed_dim_4_freq_attn = params['nb_cnn2d_filt']
        self.temp_mhsa = nn.MultiheadAttention(
            embed_dim=self.embed_dim_4_freq_attn if params.get('FreqAtten', False) else self.temp_embed_dim,
            num_heads=params['nb_heads'],
            dropout=params['dropout_rate'], batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if params.get('LinearLayer', False):
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, C, T, F):
        x_init = x
        xt = rearrange(x_init, ' b t (f c) -> (b f) t c', c=C).contiguous()
        xt, _ = self.temp_mhsa(xt, xt, xt)
        xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
        if self.linear_layer:
            xt = self.activation(self.temp_linear(xt))
        xt = xt + x_init
        if self.dropout_rate:
            xt = self.drop_out(xt)
        x_out = self.temp_layer_norm(xt)
        return x_out


# =============================================================================
# CST_encoder.py
# =============================================================================

class CST_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.ChAtten_dca = params['ChAtten_DCA']
        self.ChAtten_ule = params['ChAtten_ULE']
        self.FreqAtten = params.get('FreqAtten', False)
        self.linear_layer = params.get('linear_layer', False)
        self.dropout_rate = params['dropout_rate']
        self.temp_embed_dim = temp_embed_dim

        # Channel attention w. Divided Channel Attention (DCA)
        if self.ChAtten_dca:
            self.ch_attn_embed_dim = params['nb_cnn2d_filt']
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=params['nb_heads'],
                                                 dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Channel attention w. Unfolded Local Embedding (ULE)
        if self.ChAtten_ule:
            if params['t_pooling_loc'] == 'end':
                self.patch_size_t = int(params.get('ule_patch_t', 25))
                self.patch_size_f = int(params.get('ule_patch_f', 4))
                f_dim = float(freq_bins_input(params))
                self.freq_dim = int(f_dim / np.prod(params['f_pool_size']))
                self.temp_dim = int(params.get('ule_temp_dim', 250))
            else:
                t_enc, f_enc = encoder_output_tf(params)
                prefer_t = int(params.get('ule_prefer_patch_t', 10))
                prefer_f = int(params.get('ule_prefer_patch_f', 4))
                self.patch_size_t, self.patch_size_f = choose_ule_patch_sizes(
                    t_enc, f_enc, prefer_t=prefer_t, prefer_f=prefer_f
                )
                self.temp_dim = t_enc
                self.freq_dim = f_enc
            self.patch_size = (self.patch_size_t, self.patch_size_f)
            self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
            self.fold = nn.Fold(
                output_size=(self.temp_dim, self.freq_dim),
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.ch_attn_embed_dim = int(self.patch_size_t * self.patch_size_f)
            ule_heads = ule_mha_num_heads(
                self.ch_attn_embed_dim,
                10 if params['t_pooling_loc'] == 'end' else params['nb_heads'],
            )
            self.ch_mhsa = nn.MultiheadAttention(
                embed_dim=self.ch_attn_embed_dim,
                num_heads=ule_heads,
                dropout=self.dropout_rate,
                batch_first=True,
            )
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Spectral attention
        if self.FreqAtten:
            self.sp_attn_embed_dim = params['nb_cnn2d_filt']
            self.embed_dim_4_freq_attn = params['nb_cnn2d_filt']
            self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
                                                 dropout=self.dropout_rate, batch_first=True)
            self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.sp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # temporal attention
        self.embed_dim = temp_embed_dim
        self.temp_mhsa = nn.MultiheadAttention(
            embed_dim=self.embed_dim_4_freq_attn if params.get('FreqAtten', False) else self.embed_dim,
            num_heads=params['nb_heads'],
            dropout=self.dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.linear_layer:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, M, C, T, F):
        if self.ChAtten_ule:  # CST-attention(ULE)
            B = x.size(0)
            x_init = x.clone()

            x_unfold_in = rearrange(x_init, ' b t (f c) -> b c t f', c=C, t=T, f=F).contiguous()
            x_unfold = self.unfold(x_unfold_in)
            x_unfold = rearrange(x_unfold, 'b (c u) tf -> (b tf) c u', c=C).contiguous()

            xc, _ = self.ch_mhsa(x_unfold, x_unfold, x_unfold)

            xc = rearrange(xc, '(b tf) c u -> b (c u) tf', b=B).contiguous()
            xc = self.fold(xc)
            xc = rearrange(xc, 'b c t f -> b t (f c)').contiguous()

            if self.linear_layer:
                xc = self.activation(self.ch_linear(xc))
            xc = xc + x_init
            if self.dropout_rate:
                xc = self.drop_out(xc)
            xc = self.ch_layer_norm(xc)

            xs = rearrange(xc, ' b t (f c) -> (b t) f c', f=F).contiguous()
            xs, _ = self.sp_mhsa(xs, xs, xs)
            xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + xc
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            xt = rearrange(xs, ' b t (f c) -> (b f) t c', f=F).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        elif self.ChAtten_dca:  # CST-attention (DCA)
            x_init = x.clone()
            xc = rearrange(x_init, 'b t (m f c)-> (b t f) m c', c=C, f=F).contiguous()

            xc, _ = self.ch_mhsa(xc, xc, xc)
            xc = rearrange(xc, ' (b t f) m c -> b t (f m c)', t=T, f=F).contiguous()
            if self.linear_layer:
                xc = self.activation(self.ch_linear(xc))
            xc = xc + x_init
            if self.dropout_rate:
                xc = self.drop_out(xc)
            xc = self.ch_layer_norm(xc)

            xs = rearrange(xc, ' b t (f m c) -> (b t m) f c', c=C, t=T, f=F).contiguous()
            xs, _ = self.sp_mhsa(xs, xs, xs)
            xs = rearrange(xs, ' (b t m) f c -> b t (f m c)', m=M, t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + xc
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            xt = rearrange(xs, ' b t (f m c) -> (b f m) t c', m=M, f=F).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f m) t c -> b t (f m c)', m=M, f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        elif self.FreqAtten:  # DST-attention
            x_init = x.clone()
            x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', f=F).contiguous()
            xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
            xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + x_init
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            xt = rearrange(xs, ' b t (f c) -> (b f) t c', c=C).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        else:  # Basic Temporal Attention
            x_attn_in = x
            x, _ = self.temp_mhsa(x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.temp_layer_norm(x)

        return x


class CST_encoder(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.freq_atten = params.get('FreqAtten', False)
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ch_atten_ule = params['ChAtten_ULE']
        self.nb_ch = params['nb_channels']
        n_layers = params['nb_self_attn_layers']

        self.block_list = nn.ModuleList([CST_attention(
            temp_embed_dim=temp_embed_dim,
            params=params
        ) for _ in range(n_layers)]
        )

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch

        if self.ch_atten_dca:
            B = B // M
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B, m=M).contiguous()

        if self.ch_atten_ule or self.freq_atten:
            x = rearrange(x, 'b c t f -> b t (f c)').contiguous()

        for block in self.block_list:
            x = block(x, M, C, T, F)

        return x


# =============================================================================
# CMT_Block.py
# =============================================================================

class CMT_Layers(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.):
        super().__init__()
        self.cmt_split = params.get('CMT_split', False)
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_ule = params['ChAtten_ULE']
        self.temp_embed_dim = temp_embed_dim
        self.ffn_ratio = ffn_ratio
        self.dim = params['nb_cnn2d_filt']
        self.channels = params['nb_channels']

        self.norm1 = nn.LayerNorm(self.dim)
        self.LPU = LocalPerceptionUint(self.dim)
        self.IRFFN = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        if not self.cmt_split:
            self.cst_attention = CST_attention(temp_embed_dim=self.temp_embed_dim, params=params)
        elif self.cmt_split:
            self.spectral_atten = Spec_attention(temp_embed_dim=self.temp_embed_dim, params=params)
            self.temporal_atten = Temp_attention(temp_embed_dim=self.temp_embed_dim, params=params)
            self.norm2 = nn.LayerNorm(self.dim)
            self.LPU2 = LocalPerceptionUint(self.dim)
            self.IRFFN2 = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if not self.cmt_split:
            if not self.ch_attn_dca:
                lpu = self.LPU(x)
                x = x + lpu

                B, C, T, F = x.size()
                x = rearrange(x, 'b c t f -> b t (f c)')
                if not self.ch_attn_dca:
                    x = self.cst_attention(x, self.channels, C, T, F)
                else:
                    x = self.cst_attention(x, C, T, F)

                x_2 = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
                norm1 = self.norm1(x_2)
                norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn = self.IRFFN(norm1)
                x = x_res + self.drop_path(ffn)

            if self.ch_attn_dca:
                B, C, T, F = x.size()
                M = self.channels

                lpu = self.LPU(x)
                x = x + lpu

                x = rearrange(x, '(b m) c t f -> b t (m f c)', m=M).contiguous()
                x = self.cst_attention(x, M, C, T, F)

                x_2 = rearrange(x, 'b t (m f c) -> b (t m f) c', m=M, f=F).contiguous()
                x_res = rearrange(x, 'b t (m f c) -> (b m) c t f', m=M, f=F).contiguous()
                norm1 = self.norm1(x_2)
                norm1 = rearrange(norm1, 'b (t m f) c -> (b m) c t f', f=F, c=C, t=T).contiguous()
                ffn = self.IRFFN(norm1)
                x = x_res + self.drop_path(ffn)

        else:
            if not self.ch_attn_dca and not self.ch_attn_ule:
                lpu = self.LPU(x)
                x = x + lpu

                B, C, T, F = x.size()
                x = rearrange(x, 'b c t f -> b t (f c)').contiguous()
                x = self.spectral_atten(x, C, T, F)

                x_s = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
                norm1 = self.norm1(x_s)
                norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn_s = self.IRFFN(norm1)
                xs = x_res + self.drop_path(ffn_s)

                lpu2 = self.LPU2(xs)
                xs = xs + lpu2

                B, C, T, F = xs.size()
                x2 = rearrange(xs, 'b c t f -> b t (f c)').contiguous()
                x2 = self.temporal_atten(x2, C, T, F)

                x_t = rearrange(x2, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res_t = rearrange(x2, 'b t (f c) -> b c t f', f=F).contiguous()
                norm2 = self.norm2(x_t)
                norm2 = rearrange(norm2, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn_t = self.IRFFN2(norm2)
                x = x_res_t + self.drop_path(ffn_t)
            else:
                print("CST attention with split cmt block is not implemented yet.")
                raise ()
        return x


class CMT_block(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        self.num_layers = params['nb_self_attn_layers']
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ffn_ratio = ffn_ratio
        self.nb_ch = params['nb_channels']

        self.block_list = nn.ModuleList([CMT_Layers(
            params=params,
            temp_embed_dim=self.temp_embed_dim,
            ffn_ratio=self.ffn_ratio,
            drop_path_rate=drop_path_rate
        ) for i in range(self.num_layers)]
        )

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch

        for block in self.block_list:
            x = block(x)

        if self.ch_atten_dca:
            B = B // M
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B, m=M).contiguous()
        else:
            x = rearrange(x, 'b c t f -> b t (f c)', c=C, t=T, f=F).contiguous()

        return x


# =============================================================================
# CSTFormer_model.py - основной класс CSTFormer
# =============================================================================

class CSTFormer(torch.nn.Module):
    """
    CSTFormer : Channel-Spectral-Temporal Transformer for SELD task
    """
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_unfold = params['ChAtten_ULE']
        self.cmt_block_flag = params.get('CMT_block', False)
        self.encoder = Encoder(in_feat_shape, params)
        self.mel_bins = params['nb_mel_bins']
        self.fs = params['fs']
        self.sig_len = int(self.fs * params['hop_len_s'])
        self.predict_tdoa = params.get('predict_tdoa', False)
        self.use_ngcc = params.get('use_ngcc', False)

        if params.get('use_ngcc', False):
            self.ngcc_channels = params['ngcc_channels']
            self.ngcc_out_channels = params['ngcc_out_channels']

        self.input_nb_ch = params['nb_channels']

        if params.get('use_ngcc', False):
            # NGCC module would be imported from ngcc.model if needed
            # For now, we skip it as it requires external dependency
            self.ngcc = None

        if params.get('use_salsalite', False):
            bins = 382
        else:
            bins = params['nb_mel_bins']
        self.conv_block_freq_dim = int(np.floor(bins / np.prod(params['f_pool_size'])))
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt'] * self.input_nb_ch if self.ch_attn_dca \
            else self.conv_block_freq_dim * params['nb_cnn2d_filt']

        ## Attention Layer
        if not self.cmt_block_flag:
            self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        else:
            self.attention_stage = CMT_block(params, self.temp_embed_dim)

        if self.t_pooling_loc == 'end':
            if not params["f_pool_size"] == [1, 1, 1]:
                self.t_pooling = nn.MaxPool2d((5, 1))
            else:
                self.t_pooling = nn.MaxPool2d((5, 4))

        ## Fully Connected Layer
        self.fc_layer = FC_layer(out_shape, self.temp_embed_dim, params)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, video=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        B, M, T, F = x.size()

        if self.use_ngcc and self.ngcc is not None:
            if self.predict_tdoa:
                x, tdoa = self.ngcc(x)
            else:
                x = self.ngcc(x)

        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous()
        x = self.encoder(x)
        x = self.attention_stage(x)

        if self.t_pooling_loc == 'end':
            x = self.t_pooling(x)

        doa = self.fc_layer(x)

        if self.predict_tdoa:
            return doa, tdoa
        else:
            return doa
