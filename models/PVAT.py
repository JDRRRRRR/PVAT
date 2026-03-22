import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

from torch import distributed

import layers.Transformer_EncDec as Transformer_EncDec

class PatchEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.linear_patch = nn.Linear(patch_len, d_model, bias=False)
        self.cross_domain_carrier = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        carrier = self.cross_domain_carrier.repeat((x.shape[0], 1, 1, 1))  # [bs, n_vars, 1, d_model]
        #  [bs, n_vars, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        #  [bs * n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # [bs * n_vars, patch_num, d_model]
        x = self.linear_patch(x) + self.position_embedding(x)
        #  [bs, n_vars, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # [bs, n_vars, patch_num + 1, d_model]
        x = torch.cat([x, carrier], dim=2)
        #  [bs * n_vars, patch_num + 1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        return self.dropout(x), n_vars

class AuxiliaryEncoder(nn.Module):

    def __init__(self,
                 seq_len=96,
                 n_vars=7,
                 d_model=128,
                 dropout=0.1,
                 factor=3,
                 n_heads=8,
                 d_ff=128,
                 e_layers=1,
                 activation='gelu'):
        super().__init__()
        self.seq_len = seq_len

        # Input: [bs, seq_len, n_vars] -> Output: [bs, n_vars, d_model]
        self.linear_var = DataEmbedding_inverted(c_in=seq_len, d_model=d_model, dropout=dropout)

        self.variable_embedding_table = nn.Embedding(n_vars, d_model)

        self.variate_attention = Transformer_EncDec.Encoder(
            [
                Transformer_EncDec.EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x_enc):
        # bs, seq_len, n_vars] -> [bs, n_vars, d_model]
        enc_out = self.linear_var(x_enc, None)
        pos_indices = torch.arange(enc_out.shape[1], device=enc_out.device)
        enc_out = enc_out + self.variable_embedding_table(pos_indices).unsqueeze(0)
        enc_out, attns = self.variate_attention(enc_out)

        return enc_out

class ForecastHead(nn.Module):
    """
    Flatten and project decoder output to Target Forecast Series.

    Flattens the patch dimension and projects to the target prediction length.

    Args:
        nf (int): Number of features (d_model * patch_num)
        target_window (int): Prediction horizon length
        head_dropout (float): Dropout rate
    """

    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x[:, :, :, :-1])  # [bs, n_vars, d_model * patch_num]

        x = self.linear(x)  # [bs, n_vars, pred_len]

        return self.dropout(x)
class TargetDecoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x
    
class TargetDecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # Feed-forward network
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape

        # Self-attention on patches
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_carrier_ori = x[:, -1, :].unsqueeze(1)  # [bs * n_vars, 1, d_model]
        x_carrier = torch.reshape(x_carrier_ori, (B, -1, D))  # [bs, n_vars, d_model]

        x_carrier_attn = self.dropout(self.cross_attention(
            x_carrier, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        x_carrier_attn = torch.reshape(x_carrier_attn,
                                   (x_carrier_attn.shape[0] * x_carrier_attn.shape[1], x_carrier_attn.shape[2])).unsqueeze(1)
        x_carrier = x_carrier_ori + x_carrier_attn
        x_carrier = self.norm2(x_carrier)

        y = x = torch.cat([x[:, :-1, :], x_carrier], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


# =============================================================================
# PVAT Model
# =============================================================================
class Model(nn.Module):

    def __init__(self,
                 task_name='long_term_forecast',
                 features='M',
                 seq_len=96,
                 pred_len=96,
                 use_norm=1,
                 patch_len=48,
                 n_vars=7,
                 d_model=128,
                 dropout=0.1,
                 factor=3,
                 n_heads=2,
                 en_d_ff=512,
                 de_d_ff=512,
                 en_layers=1,
                 de_layers=2,
                 activation='gelu'):
        super().__init__()

        self.task_name = task_name
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.patch_len = patch_len

        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")
        self.patch_num = seq_len // patch_len

        self.n_vars = 1 if features == 'MS' else n_vars

        self.patch_embedding = PatchEmbedding(self.n_vars, d_model, self.patch_len, dropout)

        self.auxiliary_encoder = AuxiliaryEncoder(seq_len, n_vars, d_model, dropout, factor, n_heads, en_d_ff, en_layers)

        self.target_decoder = TargetDecoder(
            [
                TargetDecoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    de_d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(de_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.head_nf = d_model * self.patch_num
        self.forecast_head = ForecastHead(self.head_nf, pred_len, head_dropout=dropout)

    def forecast(self, x_enc):

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        patch_embed, n_vars = self.patch_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))

        aux_encoding = self.auxiliary_encoder(x_enc[:, :, :-1])

        # Target Decoder
        out = self.target_decoder(patch_embed, aux_encoding)
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        out = out.permute(0, 1, 3, 2)

        # Forecast Head -> Target Forecast Series
        out = self.forecast_head(out)
        out = out.permute(0, 2, 1)

        # Denormalization
        if self.use_norm:
            out = out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            out = out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return out

    def forecast_multi(self, x_enc):
        # Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        patch_embed, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))

        aux_encoding = self.auxiliary_encoder(x_enc)

        out = self.target_decoder(patch_embed, aux_encoding)
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        out = out.permute(0, 1, 3, 2)

        # Forecast Head -> Target Forecast Series
        out = self.forecast_head(out)
        out = out.permute(0, 2, 1)

        # Denormalization
        if self.use_norm:
            out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.features == 'M':
            dec_out = self.forecast_multi(x_enc)
            return dec_out[:, -self.pred_len:, :]
        else:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]

