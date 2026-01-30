"""
Attention Masking Module

This module provides masking utilities for attention mechanisms in Transformer models.
It includes causal masking for autoregressive models and probabilistic masking for
sparse attention mechanisms.

Masking Types:
    - TriangularCausalMask: Prevents attention to future positions (causal masking)
    - ProbMask: Probabilistic masking for sparse attention

Author: PAViT Team
"""

import torch


# =============================================================================
# Causal Masking
# =============================================================================
class TriangularCausalMask:
    """
    Triangular Causal Mask for autoregressive attention.

    Creates a lower triangular mask that prevents the model from attending to
    future positions. This is essential for autoregressive models where each
    position should only attend to previous positions.

    The mask is a lower triangular matrix where:
    - True (1) indicates positions that CAN be attended to
    - False (0) indicates positions that CANNOT be attended to (future positions)

    Args:
        B (int): Batch size
        L (int): Sequence length
        device (str): Device to place the mask on (default: "cpu")

    Example:
        >>> mask = TriangularCausalMask(B=32, L=96, device="cuda")
        >>> # mask.mask shape: [32, 1, 96, 96]
        >>> # Position i can attend to positions 0..i-1 (not including i itself)
    """

    def __init__(self, B, L, device="cpu"):
        """
        Initialize triangular causal mask.

        Args:
            B (int): Batch size
            L (int): Sequence length
            device (str): Device to place the mask on
        """
        # Create mask shape: [batch, heads, seq_len, seq_len]
        mask_shape = [B, 1, L, L]

        with torch.no_grad():
            # Create upper triangular matrix (positions to mask out)
            # triu with diagonal=1 creates a matrix where:
            # - Upper triangle (including diagonal) = True (mask out)
            # - Lower triangle = False (allow attention)
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        Get the causal mask.

        Returns:
            Tensor: Causal mask of shape [B, 1, L, L]
                   True values indicate positions to mask (future positions)
                   False values indicate positions to attend to (past positions)
        """
        return self._mask


# =============================================================================
# Probabilistic Masking
# =============================================================================
class ProbMask:
    """
    Probabilistic Mask for sparse attention.

    Creates a sparse attention mask based on top-k attention scores.
    This is used in ProbSparse attention mechanisms to reduce computational
    complexity by only attending to the most important positions.

    The mask selects which positions each query can attend to based on
    the top-k scores, while maintaining causal structure.

    Args:
        B (int): Batch size
        H (int): Number of attention heads
        L (int): Sequence length (query length)
        index (Tensor): Indices of top-k positions for each query
                       Shape: [B, H, L, k] where k is the number of top positions
        scores (Tensor): Attention scores
                        Shape: [B, H, L, L_k] where L_k is the key sequence length
        device (str): Device to place the mask on (default: "cpu")

    Example:
        >>> B, H, L, L_k = 32, 8, 96, 96
        >>> index = torch.randint(0, L_k, (B, H, L, 10))  # Top-10 positions
        >>> scores = torch.randn(B, H, L, L_k)
        >>> mask = ProbMask(B, H, L, index, scores, device="cuda")
        >>> # mask.mask shape: [32, 8, 96, 96]
    """

    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Initialize probabilistic mask.

        Args:
            B (int): Batch size
            H (int): Number of attention heads
            L (int): Query sequence length
            index (Tensor): Indices of top-k positions [B, H, L, k]
            scores (Tensor): Attention scores [B, H, L, L_k]
            device (str): Device to place the mask on
        """
        # Create base causal mask: upper triangular matrix
        # Shape: [L, L_k] where L_k is the key sequence length
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)

        # Expand mask to match batch and head dimensions
        # Shape: [B, H, L, L_k]
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])

        # Select mask values at the top-k positions for each query
        # This creates a sparse mask that only allows attention to top-k positions
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)

        # Reshape to match scores shape
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        Get the probabilistic mask.

        Returns:
            Tensor: Probabilistic mask of shape [B, H, L, L_k]
                   True values indicate positions to mask out
                   False values indicate positions to attend to (top-k)
        """
        return self._mask
