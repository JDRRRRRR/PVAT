"""
Embedding Layers Module

This module provides various embedding layers for time series data, including:
- Positional embeddings (fixed and learnable)
- Token embeddings (value embeddings via convolution)
- Temporal embeddings (time-based features)
- Combined data embeddings

These embeddings are essential for encoding both the values and temporal information
of time series data into the model's embedding space.

Author: PAViT Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


# =============================================================================
# Positional Embeddings
# =============================================================================
class LeanablePE(nn.Module):
    """
    Learnable Positional Embedding.

    Creates a learnable positional encoding that can be optimized during training.
    Unlike fixed positional embeddings, these can adapt to the specific dataset.

    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length (default: 50)
    """

    def __init__(self, d_model, max_len=50):
        super(LeanablePE, self).__init__()
        # Initialize with sinusoidal pattern
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # Apply sinusoidal pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Make it a learnable parameter
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        """
        Get positional embeddings for sequence.

        Args:
            x (Tensor): Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor: Positional embeddings [1, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


class PositionalEmbedding(nn.Module):
    """
    Fixed Positional Embedding using sinusoidal pattern.

    Implements the standard sinusoidal positional encoding from "Attention is All You Need".
    This is fixed and not learnable, providing consistent positional information.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length (default: 5000)
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # Compute division term: 10000^(2i/d_model)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # Apply sinusoidal pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # Register as buffer (not a parameter, won't be updated)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Get positional embeddings for sequence.

        Args:
            x (Tensor): Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor: Positional embeddings [1, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


# =============================================================================
# Token/Value Embeddings
# =============================================================================
class TokenEmbedding(nn.Module):
    """
    Token Embedding using 1D Convolution.

    Embeds raw time series values into the model's embedding space using
    a 1D convolutional layer with circular padding. This captures local patterns
    in the time series.

    Architecture:
        - 1D Conv: kernel_size=3, circular padding
        - Kaiming initialization for better convergence

    Args:
        c_in (int): Number of input channels (variables)
        d_model (int): Output embedding dimension
    """

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # Determine padding based on PyTorch version
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # 1D convolution with circular padding
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',  # Circular padding for time series
            bias=False
        )

        # Initialize weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Embed time series values.

        Args:
            x (Tensor): Input tensor [batch, seq_len, c_in]

        Returns:
            Tensor: Embedded values [batch, seq_len, d_model]
        """
        # Permute to [batch, c_in, seq_len] for Conv1d
        x = self.tokenConv(x.permute(0, 2, 1))
        # Permute back to [batch, seq_len, d_model]
        return x.transpose(1, 2)


# =============================================================================
# Fixed Embeddings
# =============================================================================
class FixedEmbedding(nn.Module):
    """
    Fixed Embedding using sinusoidal pattern.

    Similar to PositionalEmbedding but used for embedding categorical indices
    (like month, day, hour) rather than positions. The embedding weights are
    fixed sinusoidal patterns and not learnable.

    Args:
        c_in (int): Number of categories (e.g., 12 for months, 24 for hours)
        d_model (int): Embedding dimension
    """

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # Initialize embedding weights with sinusoidal pattern
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        # Position indices: [0, 1, 2, ..., c_in-1]
        position = torch.arange(0, c_in).float().unsqueeze(1)

        # Compute division term
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # Apply sinusoidal pattern
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # Create embedding layer with fixed weights
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        Embed categorical indices.

        Args:
            x (Tensor): Categorical indices [batch, seq_len]

        Returns:
            Tensor: Embeddings [batch, seq_len, d_model]
        """
        return self.emb(x).detach()


# =============================================================================
# Temporal Embeddings
# =============================================================================
class TemporalEmbedding(nn.Module):
    """
    Temporal Embedding for time-based features.

    Embeds temporal information (month, day, weekday, hour, minute) into
    the model's embedding space. Each temporal component has its own embedding.

    Temporal Components:
        - Month: 1-12 (size: 13)
        - Day: 1-31 (size: 32)
        - Weekday: 0-6 (size: 7)
        - Hour: 0-23 (size: 24)
        - Minute: 0-3 for 15-min intervals (size: 4)

    Args:
        d_model (int): Embedding dimension
        embed_type (str): Type of embedding ('fixed' or 'learnable')
        freq (str): Data frequency ('h' for hourly, 't' for minute, etc.)
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # Define embedding sizes for each temporal component
        minute_size = 4      # 15-minute intervals: 0-3
        hour_size = 24       # Hours: 0-23
        weekday_size = 7     # Weekdays: 0-6 (Mon-Sun)
        day_size = 32        # Days: 1-31
        month_size = 13      # Months: 1-12

        # Choose embedding type
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # Create embeddings for each temporal component
        if freq == 't':  # Minute-level data
            self.minute_embed = Embed(minute_size, d_model)

        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Embed temporal features.

        Args:
            x (Tensor): Temporal features [batch, seq_len, 5]
                - x[:, :, 0]: month (1-12)
                - x[:, :, 1]: day (1-31)
                - x[:, :, 2]: weekday (0-6)
                - x[:, :, 3]: hour (0-23)
                - x[:, :, 4]: minute (0-3 for 15-min intervals)

        Returns:
            Tensor: Temporal embeddings [batch, seq_len, d_model]
        """
        x = x.long()

        # Embed each temporal component
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # Sum all temporal embeddings
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Time Feature Embedding using linear projection.

    Projects continuous time features (from time_features function) to
    the model's embedding space using a linear layer.

    Frequency Mapping:
        - 'h' (hourly): 4 features
        - 't' (minute): 5 features
        - 's' (second): 6 features
        - 'm', 'a' (monthly, annual): 1 feature
        - 'w' (weekly): 2 features
        - 'd', 'b' (daily, business): 3 features

    Args:
        d_model (int): Embedding dimension
        embed_type (str): Embedding type (default: 'timeF')
        freq (str): Data frequency
    """

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # Map frequency to number of input features
        freq_map = {
            'h': 4,   # Hourly: 4 features
            't': 5,   # Minute: 5 features
            's': 6,   # Second: 6 features
            'm': 1,   # Monthly: 1 feature
            'a': 1,   # Annual: 1 feature
            'w': 2,   # Weekly: 2 features
            'd': 3,   # Daily: 3 features
            'b': 3    # Business: 3 features
        }
        d_inp = freq_map[freq]

        # Linear projection from time features to embedding space
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        Embed time features.

        Args:
            x (Tensor): Time features [batch, seq_len, d_inp]

        Returns:
            Tensor: Embeddings [batch, seq_len, d_model]
        """
        return self.embed(x)


# =============================================================================
# Combined Data Embeddings
# =============================================================================
class DataEmbedding(nn.Module):
    """
    Combined Data Embedding with value, position, and temporal information.

    Combines three types of embeddings:
    1. Value Embedding: Embeds raw time series values
    2. Positional Embedding: Encodes position in sequence
    3. Temporal Embedding: Encodes time-based features

    Args:
        c_in (int): Number of input channels
        d_model (int): Embedding dimension
        embed_type (str): Type of temporal embedding ('fixed' or 'timeF')
        freq (str): Data frequency
        dropout (float): Dropout rate
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # Value embedding: embed raw time series values
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # Positional embedding: encode position in sequence
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # Temporal embedding: encode time-based features
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Embed time series data with temporal information.

        Args:
            x (Tensor): Time series values [batch, seq_len, c_in]
            x_mark (Tensor, optional): Temporal features [batch, seq_len, n_temporal_features]

        Returns:
            Tensor: Combined embeddings [batch, seq_len, d_model]
        """
        if x_mark is None:
            # Only value and positional embeddings
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # All three embeddings
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)

        return self.dropout(x)


# =============================================================================
# Inverted Data Embeddings
# =============================================================================
class DataEmbedding_inverted(nn.Module):
    """
    Inverted Data Embedding for variable-wise processing.

    Used in inverted Transformers where attention is applied across variables
    instead of time steps. The embedding treats sequence length as the input
    dimension and embeds each variable independently.

    Args:
        c_in (int): Number of input channels (treated as input dimension)
        d_model (int): Embedding dimension
        embed_type (str): Type of embedding (default: 'fixed')
        freq (str): Data frequency (default: 'h')
        dropout (float): Dropout rate
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()

        # Linear projection from sequence length to embedding dimension
        # This treats each variable independently
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Embed time series for inverted processing.

        Args:
            x (Tensor): Time series values [batch, seq_len, n_vars]
            x_mark (Tensor, optional): Temporal features (unused)

        Returns:
            Tensor: Embedded values [batch, n_vars, d_model]
        """
        # Permute to [batch, n_vars, seq_len]
        x = x.permute(0, 2, 1)

        # Project from seq_len to d_model
        # [batch, n_vars, seq_len] -> [batch, n_vars, d_model]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # Concatenate temporal features if provided
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        return self.dropout(x)


# =============================================================================
# Patch Embeddings
# =============================================================================
class PatchEmbedding(nn.Module):
    """
    Patch Embedding for patch-based time series processing.

    Divides time series into patches and embeds each patch. Useful for
    capturing local patterns and reducing sequence length.

    Architecture:
        1. Padding: Add padding to ensure divisibility by patch_len
        2. Patching: Unfold into patches
        3. Embedding: Project each patch to embedding space
        4. Positional Encoding: Add positional information

    Args:
        d_model (int): Embedding dimension
        patch_len (int): Length of each patch
        stride (int): Stride for patching
        padding (int): Padding to add
        dropout (float): Dropout rate
    """

    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()

        # Patching configuration
        self.patch_len = patch_len
        self.stride = stride

        # Padding layer for ensuring divisibility
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Patch embedding: project patch_len values to d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding for patch positions
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def patch(self, x):
        """
        Divide time series into patches.

        Args:
            x (Tensor): Input tensor [batch, n_vars, seq_len]

        Returns:
            Tensor: Patches [batch, n_vars, n_patches, patch_len]
        """
        # Add padding
        x = self.padding_patch_layer(x)

        # Unfold into patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        return x

    def encode_patch(self, x):
        """
        Encode patches into embeddings.

        Args:
            x (Tensor): Input tensor [batch, n_vars, seq_len]

        Returns:
            tuple: (embeddings, n_vars)
                - embeddings: [batch * n_vars, n_patches, d_model]
                - n_vars: Number of variables
        """
        n_vars = x.shape[1]

        # Divide into patches
        x = self.patch(x)

        # Reshape for batch processing
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Project patches to embedding space
        x = self.value_embedding(x)

        return x, n_vars

    def pos_and_dropout(self, x):
        """
        Add positional encoding and dropout.

        Args:
            x (Tensor): Patch embeddings

        Returns:
            Tensor: Embeddings with positional encoding and dropout
        """
        x += self.position_embedding(x)
        return self.dropout(x)

    def forward(self, x):
        """
        Embed time series into patches with positional encoding.

        Args:
            x (Tensor): Input tensor [batch, n_vars, seq_len]

        Returns:
            tuple: (embeddings, n_vars)
                - embeddings: [batch * n_vars, n_patches, d_model]
                - n_vars: Number of variables
        """
        # Get number of variables
        n_vars = x.shape[1]

        # Add padding
        x = self.padding_patch_layer(x)

        # Divide into patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Reshape for batch processing
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Project patches to embedding space and add positional encoding
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), n_vars


# =============================================================================
# Data Embedding without Position (for TimeMixer, TimesNet, etc.)
# =============================================================================
class DataEmbedding_wo_pos(nn.Module):
    """
    Data Embedding without Positional Encoding.

    Similar to DataEmbedding but without positional encoding.
    Used in models that don't require explicit positional information.

    Args:
        c_in (int): Number of input channels
        d_model (int): Embedding dimension
        embed_type (str): Type of temporal embedding ('fixed' or 'timeF')
        freq (str): Data frequency
        dropout (float): Dropout rate
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        # Value embedding: embed raw time series values
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # Temporal embedding: encode time-based features
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Embed time series data without positional information.

        Args:
            x (Tensor): Time series values [batch, seq_len, c_in]
            x_mark (Tensor, optional): Temporal features [batch, seq_len, n_temporal_features]

        Returns:
            Tensor: Combined embeddings [batch, seq_len, d_model]
        """
        if x_mark is None:
            # Only value embedding
            x = self.value_embedding(x)
        else:
            # Value and temporal embeddings
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
