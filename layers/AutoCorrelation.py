"""
AutoCorrelation Mechanism for Time Series

This module implements the AutoCorrelation mechanism, a novel attention mechanism
specifically designed for time series forecasting. It discovers period-based
dependencies through FFT-based autocorrelation and aggregates information based
on time delays.
 
Key Innovation:
    - Period-based dependency discovery using FFT
    - Time delay aggregation for capturing periodic patterns
    - Efficient computation with O(L log L) complexity

Paper Reference:
    Autoformer: Decomposition Transformers with Auto-Correlation for Long-term Series Forecasting

Author: Autoformer Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


# =============================================================================
# AutoCorrelation Mechanism
# =============================================================================
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism for time series attention.

    This mechanism replaces traditional self-attention with autocorrelation-based
    attention, which is more suitable for time series data. It operates in two phases:

    Phase 1: Period-based Dependencies Discovery
        - Compute FFT of queries and keys
        - Multiply in frequency domain to find correlations
        - Inverse FFT to get autocorrelation in time domain
        - Identify top-k periodic patterns

    Phase 2: Time Delay Aggregation
        - For each top-k period, shift values by the delay
        - Weight each shifted pattern by its correlation strength
        - Aggregate weighted patterns to produce output

    This approach captures periodic patterns more efficiently than standard attention.

    Args:
        mask_flag (bool): Whether to apply attention mask (default: True)
        factor (int): Factor for determining top-k (top_k = factor * log(length))
        scale (float, optional): Scaling factor for attention
        attention_dropout (float): Dropout rate for attention weights
        output_attention (bool): Whether to return attention weights
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        Time delay aggregation for training phase (batch-normalized version).

        This is an optimized version that computes aggregation more efficiently
        during training by averaging correlations across heads and channels first.

        Algorithm:
            1. Compute mean correlation across heads and channels
            2. Find top-k delays with highest correlations
            3. For each top-k delay:
               - Shift values by the delay amount
               - Weight by normalized correlation
               - Accumulate weighted shifted values

        Args:
            values (Tensor): Value tensor [batch, n_heads, n_channels, length]
            corr (Tensor): Autocorrelation tensor [batch, n_heads, n_channels, length]

        Returns:
            Tensor: Aggregated values [batch, n_heads, n_channels, length]
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # Find top-k delays
        # Compute mean correlation across heads and channels
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [batch, length]

        # Find top-k indices across all delays
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        # Extract correlation weights for top-k delays
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [batch, top_k]

        # Normalize weights using softmax
        tmp_corr = torch.softmax(weights, dim=-1)

        # Aggregate values with time delays
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            # Shift values by delay amount (circular shift)
            pattern = torch.roll(tmp_values, -int(index[i]), -1)

            # Weight and accumulate
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))

        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        Time delay aggregation for inference phase.

        This version computes per-sample correlations for more accurate inference.
        It uses gather operations to efficiently extract shifted values.

        Algorithm:
            1. For each sample, find top-k delays with highest correlations
            2. For each top-k delay:
               - Create index tensor for gathering shifted values
               - Gather values at shifted positions
               - Weight by normalized correlation
               - Accumulate weighted values

        Args:
            values (Tensor): Value tensor [batch, n_heads, n_channels, length]
            corr (Tensor): Autocorrelation tensor [batch, n_heads, n_channels, length]

        Returns:
            Tensor: Aggregated values [batch, n_heads, n_channels, length]
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # Create index tensor for gathering (use same device as input)
        init_index = torch.arange(length, device=values.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)

        # Find top-k delays
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [batch, length]
        weights, delay = torch.topk(mean_value, top_k, dim=-1)  # [batch, top_k]

        # Normalize weights using softmax
        tmp_corr = torch.softmax(weights, dim=-1)

        # Duplicate values to handle circular shifts via gather
        tmp_values = values.repeat(1, 1, 1, 2)  # [batch, head, channel, 2*length]
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            # Create gather indices for shifted positions
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)

            # Gather shifted values
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)

            # Weight and accumulate
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))

        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard time delay aggregation (full version).

        This version computes per-head and per-channel correlations for maximum accuracy.
        It's more computationally expensive but provides the most accurate results.

        Args:
            values (Tensor): Value tensor [batch, n_heads, n_channels, length]
            corr (Tensor): Autocorrelation tensor [batch, n_heads, n_channels, length]

        Returns:
            Tensor: Aggregated values [batch, n_heads, n_channels, length]
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # Create index tensor for gathering (use same device as input)
        init_index = torch.arange(length, device=values.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)

        # Find top-k delays per head and channel
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)  # [batch, head, channel, top_k]

        # Normalize weights using softmax
        tmp_corr = torch.softmax(weights, dim=-1)

        # Duplicate values to handle circular shifts via gather
        tmp_values = values.repeat(1, 1, 1, 2)  # [batch, head, channel, 2*length]
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            # Create gather indices for shifted positions
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)

            # Gather shifted values
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)

            # Weight and accumulate
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))

        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        """
        Compute autocorrelation-based attention.

        Args:
            queries (Tensor): Query tensor [batch, length, n_heads, d_keys]
            keys (Tensor): Key tensor [batch, length, n_heads, d_keys]
            values (Tensor): Value tensor [batch, length, n_heads, d_values]
            attn_mask (Tensor, optional): Attention mask

        Returns:
            tuple: (output, attention_weights)
                - output: Attention output [batch, length, n_heads, d_values]
                - attention_weights: Autocorrelation weights (if output_attention=True)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Handle length mismatch between queries and keys/values
        if L > S:
            # Pad keys and values with zeros
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            # Truncate keys and values to match query length
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # =====================================================================
        # Phase 1: Period-based Dependencies Discovery
        # =====================================================================
        # Compute FFT of queries and keys
        # Permute to [batch, n_heads, d_keys, length] for FFT computation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        # Multiply in frequency domain: Q * conj(K)
        # This computes autocorrelation in frequency domain
        res = q_fft * torch.conj(k_fft)

        # Inverse FFT to get autocorrelation in time domain
        corr = torch.fft.irfft(res, dim=-1)

        # =====================================================================
        # Phase 2: Time Delay Aggregation
        # =====================================================================
        # Choose aggregation method based on training/inference mode
        if self.training:
            # Training: use batch-normalized version for efficiency
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            # Inference: use per-sample version for accuracy
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        # Return output and optionally attention weights
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


# =============================================================================
# AutoCorrelation Layer (Multi-Head Wrapper)
# =============================================================================
class AutoCorrelationLayer(nn.Module):
    """
    Multi-head AutoCorrelation layer wrapper.

    This layer wraps the AutoCorrelation mechanism with multi-head projection,
    similar to multi-head attention in standard Transformers.

    Architecture:
        1. Project input to multiple heads
        2. Apply AutoCorrelation mechanism
        3. Project output back to original dimension

    Args:
        correlation (AutoCorrelation): AutoCorrelation mechanism instance
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_keys (int, optional): Dimension per head for keys (default: d_model // n_heads)
        d_values (int, optional): Dimension per head for values (default: d_model // n_heads)
    """

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        # Set default dimensions per head
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation

        # Multi-head projections
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # Output projection
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        Apply multi-head AutoCorrelation.

        Args:
            queries (Tensor): Query tensor [batch, length, d_model]
            keys (Tensor): Key tensor [batch, length, d_model]
            values (Tensor): Value tensor [batch, length, d_model]
            attn_mask (Tensor, optional): Attention mask

        Returns:
            tuple: (output, attention_weights)
                - output: [batch, length, d_model]
                - attention_weights: Autocorrelation weights (if enabled)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project to multiple heads
        # [batch, length, d_model] -> [batch, length, n_heads, d_keys]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply AutoCorrelation mechanism
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        # Reshape back to [batch, length, d_model]
        out = out.view(B, L, -1)

        # Project output back to original dimension
        return self.out_projection(out), attn
