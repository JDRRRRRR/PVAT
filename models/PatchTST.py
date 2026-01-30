"""
PatchTST Model Implementation

PatchTST (Patch Time Series Transformer) is a patch-based Transformer architecture
for time series forecasting. It divides time series into patches and applies
Transformer attention on the patch level, reducing computational complexity.

Key Features:
    - Patch-based processing: Divides input into non-overlapping patches
    - Channel-independent: Processes each variable independently
    - Efficient attention: Operates on patches instead of full sequences
    - Non-stationary normalization: Handles non-stationary time series

Paper Reference:
    A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
    https://arxiv.org/pdf/2211.14730.pdf

Author: PatchTST Team
"""

# Note: This model does not support MS (multivariate-to-single) task, only S (univariate) or M (multivariate)

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# =============================================================================
# Utility Modules
# =============================================================================
class Transpose(nn.Module):
    """
    Transpose tensor dimensions.

    Utility module for transposing tensor dimensions, with optional
    contiguous memory layout.

    Args:
        *dims: Dimensions to transpose
        contiguous (bool): Whether to make tensor contiguous after transpose
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        """Transpose and optionally make contiguous."""
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    Flatten and project patch embeddings to prediction horizon.

    Flattens the patch dimension and projects to the target prediction length.

    Args:
        n_vars (int): Number of variables
        nf (int): Number of features (d_model * patch_num)
        target_window (int): Prediction horizon length
        head_dropout (float): Dropout rate
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Project patch embeddings to prediction horizon.

        Args:
            x (Tensor): Patch embeddings [bs, n_vars, d_model, patch_num]

        Returns:
            Tensor: Predictions [bs, n_vars, target_window]
        """
        # Flatten: [bs, n_vars, d_model, patch_num] -> [bs, n_vars, d_model * patch_num]
        x = self.flatten(x)

        # Project to target window
        x = self.linear(x)

        return self.dropout(x)


# =============================================================================
# PatchTST Model
# =============================================================================
class Model(nn.Module):
    """
    PatchTST: Patch Time Series Transformer.

    Divides time series into patches and applies Transformer attention
    on the patch level for efficient long-term forecasting.

    Architecture:
        1. Patch Embedding: Divide input into patches and embed
        2. Encoder: Apply Transformer attention on patches
        3. Prediction Head: Project to prediction horizon
        4. Denormalization: Reverse normalization

    Supported Tasks:
        - long_term_forecast: Long-term time series forecasting
        - short_term_forecast: Short-term time series forecasting
        - imputation: Missing value imputation
        - anomaly_detection: Anomaly detection in time series
        - classification: Time series classification

    Args:
        task_name (str): Task type (default: 'long_term_forecast')
        seq_len (int): Input sequence length
        pred_len (int): Prediction horizon length
        patch_len (int): Length of each patch
        d_model (int): Model embedding dimension
        dropout (float): Dropout rate
        factor (int): Attention factor for sparse attention
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward network dimension
        activation (str): Activation function ('gelu' or 'relu')
        e_layers (int): Number of encoder layers
        enc_in (int): Number of input channels (variables)
        num_class (int): Number of classes for classification task
        stride (int): Stride for patch extraction
    """

    def __init__(self, task_name='long_term_forecast',
                 seq_len=96,
                 pred_len=96,
                 patch_len=16,
                 d_model=512,
                 dropout=0.1,
                 factor=3,
                 n_heads=2,
                 d_ff=2048,
                 activation='gelu',
                 e_layers=1,
                 enc_in=7,
                 num_class=10,
                 stride=8):
        """
        Initialize PatchTST model.

        Args:
            patch_len (int): Length of each patch
            stride (int): Stride for patch extraction
        """
        super().__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride

        # =====================================================================
        # Patch Embedding
        # =====================================================================
        # Divide input into patches and embed each patch
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )

        # =====================================================================
        # Encoder
        # =====================================================================
        # Stack of encoder layers with full attention on patches
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Full attention layer
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.BatchNorm1d(d_model)
        )

        # =====================================================================
        # Prediction Head
        # =====================================================================
        # Calculate number of patches after patching
        # Formula: (seq_len - patch_len) / stride + 2 (accounting for padding)
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Forecasting: project to prediction horizon
            self.head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # Imputation/Anomaly: project to sequence length
            self.head = FlattenHead(enc_in, self.head_nf, seq_len, head_dropout=dropout)

        elif self.task_name == 'classification':
            # Classification head
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(self.head_nf * enc_in, num_class)

    # =========================================================================
    # Task-specific Methods
    # =========================================================================
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task: predict future values.

        Args:
            x_enc (Tensor): Encoder input [batch, seq_len, n_vars]
            x_mark_enc (Tensor): Encoder time features (unused)
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)

        Returns:
            Tensor: Predictions [batch, pred_len, n_vars]
        """
        # =====================================================================
        # Normalization (Non-stationary Transformer)
        # =====================================================================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # =====================================================================
        # Patch Embedding
        # =====================================================================
        # Permute to [batch, n_vars, seq_len] for patching
        x_enc = x_enc.permute(0, 2, 1)

        # Apply patch embedding: [batch * n_vars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # =====================================================================
        # Encoder
        # =====================================================================
        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out)

        # Reshape back to [batch, n_vars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # Permute to [batch, n_vars, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # =====================================================================
        # Prediction Head
        # =====================================================================
        # Project to prediction horizon: [batch, n_vars, pred_len]
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # =====================================================================
        # Denormalization
        # =====================================================================
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation task: fill missing values.

        Args:
            x_enc (Tensor): Input with missing values [batch, seq_len, n_vars]
            x_mark_enc (Tensor): Time features (unused)
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)
            mask (Tensor): Missing value mask

        Returns:
            Tensor: Imputed values [batch, seq_len, n_vars]
        """
        # =====================================================================
        # Normalization (using only observed values)
        # =====================================================================
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # =====================================================================
        # Patch Embedding and Encoding
        # =====================================================================
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # =====================================================================
        # Prediction Head and Denormalization
        # =====================================================================
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task: identify anomalies.

        Args:
            x_enc (Tensor): Input time series [batch, seq_len, n_vars]

        Returns:
            Tensor: Anomaly scores [batch, seq_len, n_vars]
        """
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patch embedding and encoding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction head and denormalization
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Classification task: classify time series.

        Args:
            x_enc (Tensor): Input time series [batch, seq_len, n_vars]
            x_mark_enc (Tensor): Time features (unused)

        Returns:
            Tensor: Class logits [batch, num_class]
        """
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patch embedding and encoding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Classification head
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output

    # =========================================================================
    # Main Forward Method
    # =========================================================================
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass: route to appropriate task-specific method.

        Args:
            x_enc (Tensor): Encoder input
            x_mark_enc (Tensor): Encoder time features
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)
            mask (Tensor, optional): Mask for imputation task

        Returns:
            Tensor: Task-specific output
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out

        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out

        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out

        return None
