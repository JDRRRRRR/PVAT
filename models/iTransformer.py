"""
iTransformer Model Implementation

iTransformer (Inverted Transformer) is a novel architecture that applies Transformer
attention across variables instead of time steps. This inverted perspective is more
suitable for multivariate time series forecasting.

Key Innovation:
    - Inverted attention: Applies attention across variables (columns) instead of time (rows)
    - Variable-wise dependencies: Captures correlations between different variables
    - Non-stationary normalization: Handles non-stationary time series effectively

Paper Reference:
    iTransformer: Inverted Transformers for Time Series Forecasting
    https://arxiv.org/abs/2310.06625

Author: iTransformer Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


# =============================================================================
# iTransformer Model
# =============================================================================
class Model(nn.Module):
    """
    iTransformer: Inverted Transformers for Time Series Forecasting.

    Applies Transformer attention across variables instead of time steps,
    capturing variable-wise dependencies and correlations.

    Architecture:
        1. Inverted Embedding: Transpose to [batch, n_vars, d_model]
        2. Encoder: Apply Transformer attention across variables
        3. Projection: Project to prediction horizon
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
        d_model (int): Model embedding dimension
        dropout (float): Dropout rate
        factor (int): Attention factor for sparse attention
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward network dimension
        activation (str): Activation function ('gelu' or 'relu')
        e_layers (int): Number of encoder layers
        enc_in (int): Number of input channels (variables)
        num_class (int): Number of classes for classification task
    """

    def __init__(self,
                 task_name='long_term_forecast',
                 seq_len=96,
                 pred_len=96,
                 d_model=128,
                 dropout=0.1,
                 factor=3,
                 n_heads=8,
                 d_ff=128,
                 activation='gelu',
                 e_layers=2,
                 enc_in=7,
                 num_class=10):
        super(Model, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len

        # =====================================================================
        # Inverted Embedding
        # =====================================================================
        # Treats sequence length as input dimension for inverted processing
        # Input: [batch, seq_len, n_vars] -> Output: [batch, n_vars, d_model]
        self.enc_embedding = DataEmbedding_inverted(
            c_in=seq_len, d_model=d_model, dropout=dropout
        )

        # =====================================================================
        # Encoder
        # =====================================================================
        # Stack of encoder layers with full attention across variables
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
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # =====================================================================
        # Task-specific Projections
        # =====================================================================
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Project from d_model to prediction horizon
            self.projection = nn.Linear(d_model, pred_len, bias=True)

        if self.task_name == 'imputation':
            # Project from d_model to sequence length
            self.projection = nn.Linear(d_model, seq_len, bias=True)

        if self.task_name == 'anomaly_detection':
            # Project from d_model to sequence length
            self.projection = nn.Linear(d_model, seq_len, bias=True)

        if self.task_name == 'classification':
            # Classification head
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            # Flatten and project to class logits
            self.projection = nn.Linear(d_model * enc_in, num_class)

    # =========================================================================
    # Task-specific Methods
    # =========================================================================
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task: predict future values.

        Algorithm:
            1. Normalize input using mean and standard deviation
            2. Apply inverted embedding
            3. Encode with Transformer
            4. Project to prediction horizon
            5. Denormalize output

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
        # Compute mean and standard deviation for normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means

        # Compute standard deviation with small epsilon for numerical stability
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Get number of variables
        _, _, N = x_enc.shape

        # =====================================================================
        # Encoding
        # =====================================================================
        # Apply inverted embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # =====================================================================
        # Projection and Denormalization
        # =====================================================================
        # Project to prediction horizon: [batch, n_vars, pred_len]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # Denormalize: reverse the normalization applied earlier
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
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embed and encode
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to sequence length
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # Denormalize
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

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

        _, L, N = x_enc.shape

        # Embed and encode
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to sequence length
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # Denormalize
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

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
        # Embed and encode
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Apply activation and dropout
        output = self.act(enc_out)
        output = self.dropout(output)

        # Flatten: [batch, n_vars, d_model] -> [batch, n_vars * d_model]
        output = output.reshape(output.shape[0], -1)

        # Project to class logits
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
        # Route to appropriate task handler
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Forecasting: return only prediction horizon
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

        if self.task_name == 'imputation':
            # Imputation: return full sequence
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out

        if self.task_name == 'anomaly_detection':
            # Anomaly detection: return anomaly scores
            dec_out = self.anomaly_detection(x_enc)
            return dec_out

        if self.task_name == 'classification':
            # Classification: return class logits
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out

        # Unknown task
        return None
