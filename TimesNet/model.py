import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from transformers import PreTrainedModel, PretrainedConfig

from .layers.Embed import DataEmbedding
from .layers.Conv_Blocks import Inception_Block_V1
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class TimesNetForTimeSeriesClassificationOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None



def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(2).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNetConfig(PretrainedConfig):
    model_type = "timesnet" 

    def __init__(
        self,
        seq_len=96,
        d_model=64,
        d_ff=128,
        top_k=2,
        num_kernels=3,
        e_layers=2,
        enc_in=7,
        freq='h',
        embed='timeF',
        dropout=0.05,
        num_class=10,
        **kwargs
    ):
        super().__init__(**kwargs) 
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.e_layers = e_layers
        self.enc_in = enc_in
        self.freq = freq
        self.embed = embed
        self.dropout = dropout
        self.num_class = num_class
        for k, v in kwargs.items():
            setattr(self, k, v)


class TimesNetForTimeSeriesClassification(PreTrainedModel):
    """
    A TimesNet model that only supports classification.
    """
    config_class = TimesNetConfig

    def __init__(self, configs):
        super().__init__(configs)
        self.seq_len = configs.seq_len
        self.layer = configs.e_layers
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(self.layer)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        # For classification
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, past_values, target_values=None, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # embedding
        enc_out = self.enc_embedding(past_values, None)  # [B, T, C]
        # TimesNet stack
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        logits = output.reshape(output.shape[0], -1)
        logits = self.projection(logits)

        loss = None
        if target_values is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, target_values)

        return TimesNetForTimeSeriesClassificationOutput(
            loss=loss,
            logits=logits,
        )
