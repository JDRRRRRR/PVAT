"""
Autoformer Model Implementation

Autoformer is a decomposition-based Transformer architecture specifically designed
for long-term time series forecasting. It achieves O(L log L) complexity through
the AutoCorrelation mechanism and incorporates series decomposition for improved
forecasting performance.

Key Innovations:
    - Series Decomposition: Separates trend and seasonal components
    - AutoCorrelation: Efficient attention mechanism using FFT
    - Trend Extrapolation: Explicitly models trend component
    - Multi-task Support: Forecasting, imputation, anomaly detection, classification

Paper Reference:
    Autoformer: Decomposition Transformers with Auto-Correlation for Long-term Series Forecasting
    https://arxiv.org/abs/2106.13008

Author: Autoformer Team
"""

import torch
import torch.nn as nn
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Embed import DataEmbedding_wo_pos


# =============================================================================
# Autoformer Model
# =============================================================================
class Model(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation.

    Autoformer is the first method to achieve series-wise connection with
    inherent O(L log L) complexity. It combines:
    1. Series decomposition for trend-seasonal separation
    2. AutoCorrelation mechanism for efficient attention
    3. Trend extrapolation for explicit trend modeling

    Architecture:
        1. Series Decomposition: Split input into trend and seasonal
        2. Encoder: Process seasonal component with AutoCorrelation
        3. Decoder: Combine seasonal predictions with trend extrapolation
        4. Reconstruction: Add trend and seasonal components

    Supported Tasks:
        - long_term_forecast: Long-term time series forecasting
        - short_term_forecast: Short-term time series forecasting
        - imputation: Missing value imputation
        - anomaly_detection: Anomaly detection in time series
        - classification: Time series classification

    Args:
        task_name (str): Task type (default: 'long_term_forecast')
        seq_len (int): Input sequence length
        label_len (int): Decoder start token length (overlap with encoder)
        pred_len (int): Prediction horizon length
        output_attention (bool): Whether to output attention weights
        moving_avg (int): Window size for moving average decomposition
        enc_in (int): Number of encoder input channels
        dec_in (int): Number of decoder input channels
        d_model (int): Model embedding dimension
        embed (str): Embedding type ('timeF' for time features)
        freq (str): Data frequency ('h' for hourly, etc.)
        dropout (float): Dropout rate
        factor (int): Attention factor for AutoCorrelation
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward network dimension
        activation (str): Activation function ('gelu' or 'relu')
        e_layers (int): Number of encoder layers
        d_layers (int): Number of decoder layers
        c_out (int): Number of output channels
        num_class (int): Number of classes for classification task
    """

    def __init__(self,
                 task_name='long_term_forecast',
                 seq_len=96,
                 label_len=48,
                 pred_len=96,
                 output_attention=False,
                 moving_avg=25,
                 enc_in=7,
                 dec_in=7,
                 d_model=512,
                 embed='timeF',
                 freq='h',
                 dropout=0.1,
                 factor=3,
                 n_heads=8,
                 d_ff=2048,
                 activation='gelu',
                 e_layers=2,
                 d_layers=1,
                 c_out=7,
                 num_class=10):
        super(Model, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # =====================================================================
        # Series Decomposition
        # =====================================================================
        # Decompose input into trend and seasonal components
        # Uses moving average with specified kernel size
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # =====================================================================
        # Embeddings
        # =====================================================================
        # Encoder embedding: embed encoder input (without positional encoding)
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)

        # Decoder embedding: embed decoder input (without positional encoding)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # =====================================================================
        # Encoder
        # =====================================================================
        # Stack of encoder layers with AutoCorrelation attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # AutoCorrelation layer for attention
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,  # No masking for encoder
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # =====================================================================
        # Decoder
        # =====================================================================
        # Decoder configuration depends on task type
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Forecasting decoder with trend extrapolation
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        # Self-attention on decoder input (with masking)
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                mask_flag=True,  # Causal masking for decoder
                                factor=factor,
                                attention_dropout=dropout,
                                output_attention=False
                            ),
                            d_model, n_heads
                        ),
                        # Cross-attention between decoder and encoder
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                mask_flag=False,  # No masking for cross-attention
                                factor=factor,
                                attention_dropout=dropout,
                                output_attention=False
                            ),
                            d_model, n_heads
                        ),
                        d_model,
                        c_out,
                        d_ff,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation,
                    ) for _ in range(d_layers)
                ],
                norm_layer=my_Layernorm(d_model),
                projection=nn.Linear(d_model, c_out, bias=True)
            )

        # =====================================================================
        # Task-specific Projections
        # =====================================================================
        if self.task_name == 'imputation':
            # Projection for imputation task
            self.projection = nn.Linear(d_model, c_out, bias=True)

        if self.task_name == 'anomaly_detection':
            # Projection for anomaly detection task
            self.projection = nn.Linear(d_model, c_out, bias=True)

        if self.task_name == 'classification':
            # Classification head
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(dropout)
            # Flatten and project to number of classes
            self.projection = nn.Linear(d_model * seq_len, num_class)

    # =========================================================================
    # Task-specific Forward Methods
    # =========================================================================
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task: predict future values.

        Algorithm:
            1. Decompose encoder input into trend and seasonal
            2. Encode seasonal component
            3. Extrapolate trend component
            4. Decode seasonal component with trend context
            5. Combine trend and seasonal predictions

        Args:
            x_enc (Tensor): Encoder input [batch, seq_len, enc_in]
            x_mark_enc (Tensor): Encoder time features [batch, seq_len, n_features]
            x_dec (Tensor): Decoder input [batch, label_len + pred_len, dec_in]
            x_mark_dec (Tensor): Decoder time features [batch, label_len + pred_len, n_features]

        Returns:
            Tensor: Predictions [batch, pred_len, c_out]
        """
        # =====================================================================
        # Series Decomposition
        # =====================================================================
        # Compute mean of encoder input for trend extrapolation
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # Create zero tensor for seasonal component padding
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        # Decompose encoder input into seasonal and trend
        seasonal_init, trend_init = self.decomp(x_enc)

        # =====================================================================
        # Prepare Decoder Input
        # =====================================================================
        # Trend: concatenate last label_len steps with mean extrapolation
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

        # Seasonal: concatenate last label_len steps with zeros for prediction
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # =====================================================================
        # Encoder
        # =====================================================================
        # Embed encoder input
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # =====================================================================
        # Decoder
        # =====================================================================
        # Embed decoder input (seasonal component)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        # Apply decoder layers with trend context
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out,
            x_mask=None,
            cross_mask=None,
            trend=trend_init
        )

        # =====================================================================
        # Reconstruction
        # =====================================================================
        # Combine trend and seasonal predictions
        dec_out = trend_part + seasonal_part

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Imputation task: fill missing values.

        Algorithm:
            1. Encode input with missing values
            2. Project encoder output to reconstruct missing values

        Args:
            x_enc (Tensor): Encoder input with missing values [batch, seq_len, enc_in]
            x_mark_enc (Tensor): Encoder time features [batch, seq_len, n_features]
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)
            mask (Tensor): Missing value mask

        Returns:
            Tensor: Imputed values [batch, seq_len, c_out]
        """
        # Embed encoder input
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to output dimension
        dec_out = self.projection(enc_out)

        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task: identify anomalies in time series.

        Algorithm:
            1. Encode input
            2. Project encoder output to anomaly scores

        Args:
            x_enc (Tensor): Input time series [batch, seq_len, enc_in]

        Returns:
            Tensor: Anomaly scores [batch, seq_len, c_out]
        """
        # Embed encoder input (no time features for anomaly detection)
        enc_out = self.enc_embedding(x_enc, None)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to output dimension
        dec_out = self.projection(enc_out)

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Classification task: classify time series.

        Algorithm:
            1. Encode input
            2. Apply activation and dropout
            3. Flatten and project to class logits

        Args:
            x_enc (Tensor): Input time series [batch, seq_len, enc_in]
            x_mark_enc (Tensor): Time features [batch, seq_len, n_features]

        Returns:
            Tensor: Class logits [batch, num_class]
        """
        # Embed encoder input (no time features for classification)
        enc_out = self.enc_embedding(x_enc, None)

        # Apply encoder layers
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Apply activation function
        output = self.act(enc_out)

        # Apply dropout
        output = self.dropout(output)

        # Apply time feature weighting if provided
        output = output * x_mark_enc.unsqueeze(-1) if x_mark_enc is not None else output

        # Flatten: [batch, seq_len, d_model] -> [batch, seq_len * d_model]
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
            x_dec (Tensor): Decoder input
            x_mark_dec (Tensor): Decoder time features
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
