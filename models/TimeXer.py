import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

class FlattenHead(nn.Module):
    """
    Flatten and project patch embeddings to prediction horizon.

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
        """Project patch embeddings to prediction horizon."""
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """
    Patch Embedding for encoder input.

    Divides input into patches and embeds each patch with positional encoding.
    Also adds a learnable global token for aggregating patch information.

    Args:
        n_vars (int): Number of variables
        d_model (int): Embedding dimension
        patch_len (int): Length of each patch
        dropout (float): Dropout rate
    """

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len

        # Linear projection from patch_len to d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Learnable global token for aggregating patch information
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))

        # Positional encoding for patch positions
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Embed patches with positional encoding and global token.

        Args:
            x (Tensor): Input [batch, n_vars, seq_len]

        Returns:
            tuple: (embeddings, n_vars)
        """
        # Get number of variables
        n_vars = x.shape[1]

        # Repeat global token for batch size
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # Divide into patches: [batch, n_vars, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # Reshape for batch processing: [batch * n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Project patches and add positional encoding
        x = self.value_embedding(x) + self.position_embedding(x)

        # Reshape back: [batch, n_vars, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # Concatenate global token: [batch, n_vars, patch_num + 1, d_model]
        x = torch.cat([x, glb], dim=2)

        # Reshape for encoder: [batch * n_vars, patch_num + 1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    Encoder with multiple layers and optional normalization/projection.

    Args:
        layers (list): List of encoder layers
        norm_layer (nn.Module, optional): Normalization layer
        projection (nn.Module, optional): Output projection layer
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Apply encoder layers with cross-attention.

        Args:
            x (Tensor): Patch embeddings
            cross (Tensor): Variable embeddings for cross-attention
            x_mask (Tensor, optional): Self-attention mask
            cross_mask (Tensor, optional): Cross-attention mask
            tau (float, optional): Temperature parameter
            delta (float, optional): Time delay parameter

        Returns:
            Tensor: Encoded output
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer with self-attention and cross-attention.

    Combines patch information (self-attention) with variable information
    (cross-attention) through a global token mechanism.

    Args:
        self_attention (AttentionLayer): Self-attention module
        cross_attention (AttentionLayer): Cross-attention module
        d_model (int): Embedding dimension
        d_ff (int, optional): Feed-forward dimension
        dropout (float): Dropout rate
        activation (str): Activation function
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
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
        """
        Apply encoder layer with self and cross-attention.

        Args:
            x (Tensor): Patch embeddings [bs * n_vars, patch_num + 1, d_model]
            cross (Tensor): Variable embeddings [bs, n_vars, d_model]
            x_mask (Tensor, optional): Self-attention mask
            cross_mask (Tensor, optional): Cross-attention mask
            tau (float, optional): Temperature parameter
            delta (float, optional): Time delay parameter

        Returns:
            Tensor: Updated embeddings
        """
        B, L, D = cross.shape

        # Self-attention on patches
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # Extract global token (last position) for cross-attention
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [bs * n_vars, 1, d_model]
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # [bs, n_vars, d_model]

        # Cross-attention: global token attends to variable embeddings
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        # Reshape back and add to global token
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # Concatenate updated global token with patches
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # Feed-forward network
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


# =============================================================================
# TimeXer Model
# =============================================================================
class Model(nn.Module):
    """
    TimeXer: Time Series Exogenous Modeling with Transformers.

    Combines patch-based embeddings with cross-variable attention for
    time series forecasting with exogenous variables.

    Architecture:
        1. Patch Embedding: Divide input into patches
        2. Variable Embedding: Extract variable-wise patterns
        3. Encoder: Apply self and cross-attention
        4. Prediction Head: Generate forecasts

    Args:
        task_name (str): Task type (default: 'long_term_forecast')
        features (str): Forecasting task ('M' for multivariate, 'S' for univariate)
        seq_len (int): Input sequence length
        pred_len (int): Prediction horizon length
        use_norm (bool): Whether to use normalization
        patch_len (int): Patch length (typically 16 for TimeXer)
        enc_in (int): Number of input channels (variables)
        d_model (int): Embedding dimension
        dropout (float): Dropout rate
        factor (int): Attention factor
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        activation (str): Activation function
        e_layers (int): Number of encoder layers
    """

    def __init__(self,
                 task_name='long_term_forecast',
                 features='M',
                 seq_len=96,
                 pred_len=96,
                 use_norm=1,
                 patch_len=16,
                 enc_in=7,
                 d_model=256,
                 dropout=0.1,
                 factor=3,
                 n_heads=8,
                 d_ff=1024,
                 activation='gelu',
                 e_layers=1):
        super(Model, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.task_name = task_name
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)

        # For MS task, only predict single variable
        self.n_vars = 1 if features == 'MS' else enc_in

        # =====================================================================
        # Embeddings
        # =====================================================================
        # Patch embedding for encoder input
        self.en_embedding = EnEmbedding(self.n_vars, d_model, self.patch_len, dropout)

        # Variable embedding using inverted Transformer
        self.ex_embedding = DataEmbedding_inverted(c_in=seq_len, d_model=d_model, dropout=dropout)

        # =====================================================================
        # Encoder
        # =====================================================================
        # Encoder-only architecture with self and cross-attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self-attention on patches
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads
                    ),
                    # Cross-attention with variables
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Calculate number of features: d_model * (patch_num + 1) for global token
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task for univariate or MS task.

        Args:
            x_enc (Tensor): Encoder input [batch, seq_len, n_vars]
            x_mark_enc (Tensor): Encoder time features
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)

        Returns:
            Tensor: Predictions [batch, pred_len, 1]
        """
        # =====================================================================
        # Normalization
        # =====================================================================
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # =====================================================================
        # Embedding
        # =====================================================================
        # Patch embedding: use only target variable (last column)
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))

        # Variable embedding: use exogenous variables (all except last)
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        # =====================================================================
        # Encoder
        # =====================================================================
        enc_out = self.encoder(en_embed, ex_embed)

        # Reshape: [batch, n_vars, patch_num + 1, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # Permute: [batch, n_vars, d_model, patch_num + 1]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # =====================================================================
        # Prediction Head
        # =====================================================================
        dec_out = self.head(enc_out)  # [batch, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)

        # =====================================================================
        # Denormalization
        # =====================================================================
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting task for multivariate task.

        Args:
            x_enc (Tensor): Encoder input [batch, seq_len, n_vars]
            x_mark_enc (Tensor): Encoder time features
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)

        Returns:
            Tensor: Predictions [batch, pred_len, n_vars]
        """
        # Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding: use all variables
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out = self.encoder(en_embed, ex_embed)

        # Reshape and permute
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction head
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # Denormalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass: route to appropriate task-specific method.

        Args:
            x_enc (Tensor): Encoder input
            x_mark_enc (Tensor): Encoder time features
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)
            mask (Tensor, optional): Mask (unused)

        Returns:
            Tensor: Task-specific output
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                # Multivariate forecasting
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
            else:
                # Univariate or MS forecasting
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
        else:
            return None
