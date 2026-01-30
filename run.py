"""
Time Series Forecasting Training Script

This script provides a unified training pipeline for multiple time series forecasting models,
including DLinear, PatchTST, iTransformer, TimeXer, PAViT, and Autoformer.

Main Features:
    - Automatic device detection (CUDA, MPS, CPU)
    - Configurable model architecture via command-line arguments
    - Support for multiple datasets (ETT, electricity, weather, etc.)
    - Training, validation, and testing pipeline
    - Learning rate scheduling
    - Evaluation metrics logging

Usage:
    python run.py --model PAViT --data ETTh1 --seq_len 96 --pred_len 96

Author: PAViT Team
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from dataset.data_factory import data_provider
from utils.tools import adjust_learning_rate
from utils.metrics import metric

# Import all available model modules
import models.DLinear as DLinear
import models.PatchTST as PatchTST
import models.iTransformer as iTransformer
import models.TimeXer as TimeXer
import models.PVAT as PVAT
import models.TimesNet as TimesNet
import models.TimeMixer as TimeMixer

# =============================================================================
# Model Registry
# =============================================================================
# Maps model names to their corresponding module implementations.
# To add a new model, import it above and add an entry here.
MODEL_REGISTRY = {
    'DLinear': DLinear,          # Simple linear model for time series
    'PatchTST': PatchTST,        # Patch-based Transformer
    'iTransformer': iTransformer, # Inverted Transformer (variable-wise attention)
    'TimeXer': TimeXer,          # Time series Transformer with exogenous variables
    'PVAT': PVAT,                # Patch- and Variable-Aligned Transformer
    'TimesNet': TimesNet,        # TimesNet with 2D convolution
    'TimeMixer': TimeMixer,      # Multi-scale mixing model
}


# =============================================================================
# Device Management
# =============================================================================
def get_device(gpu_id=None):
    """
    Automatically detect and return the best available computing device.

    The function checks for available devices in the following priority order:
    1. CUDA GPU (if available and gpu_id is specified or default)
    2. Apple MPS (Metal Performance Shaders for M1/M2 Macs)
    3. CPU (fallback)

    Args:
        gpu_id (int, optional): Specific GPU device ID to use.
                                If None, uses the default CUDA device.

    Returns:
        torch.device: The selected computing device.

    Example:
        >>> device = get_device(gpu_id=0)  # Use GPU 0
        >>> device = get_device()           # Auto-detect best device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2) GPU support
        return torch.device('mps')
    return torch.device('cpu')


# =============================================================================
# Model Building
# =============================================================================
def build_model(args, enc_in):
    """
    Build and initialize a forecasting model based on configuration.

    This factory function creates the appropriate model instance based on
    the model name specified in args. Each model has different architectural
    parameters that are configured through the args namespace.

    Args:
        args (argparse.Namespace): Configuration containing:
            - model (str): Name of the model to build
            - seq_len (int): Input sequence length
            - pred_len (int): Prediction horizon length
            - patch_len (int): Patch size for patch-based models
            - d_model (int): Model embedding dimension
            - dropout (float): Dropout rate
            - factor (int): Attention factor for sparse attention
            - n_heads (int): Number of attention heads
            - en_d_ff (int): Encoder feed-forward dimension
            - de_d_ff (int): Decoder feed-forward dimension
            - en_layers (int): Number of encoder layers
            - de_layers (int): Number of decoder layers
            - device (torch.device): Target device for the model
        enc_in (int): Number of input variables/channels in the data.

    Returns:
        nn.Module: Initialized model moved to the specified device.

    Raises:
        ValueError: If the specified model name is not in MODEL_REGISTRY.

    Example:
        >>> model = build_model(args, enc_in=7)  # 7 input variables
    """
    model_module = MODEL_REGISTRY.get(args.model)
    if model_module is None:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_REGISTRY.keys())}")

    # Build model based on type - each model has different parameter requirements
    if args.model == 'DLinear':
        # DLinear: Simple decomposition-based linear model
        # Only requires sequence lengths and input channels
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=enc_in
        )
    elif args.model == 'PatchTST':
        # PatchTST: Patch-based Transformer for time series
        # Divides input into patches and applies channel-independent processing
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers,
            enc_in=enc_in
        )
    elif args.model == 'iTransformer':
        # iTransformer: Inverted Transformer
        # Applies attention across variables instead of time steps
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers,
            enc_in=enc_in
        )
    elif args.model == 'TimeXer':
        # TimeXer: Transformer with exogenous variable handling
        # Combines patch embeddings with cross-variable attention
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            enc_in=enc_in,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers
        )
    elif args.model == 'PVAT':
        # PVAT: Patch- and Variable-Aligned Transformer
        # Uses Target Decoder to fuse patch and Auxiliary Encoding Information
        model = model_module.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            n_vars=enc_in,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            en_d_ff=args.en_d_ff,
            de_d_ff=args.de_d_ff,
            en_layers=args.en_layers,
            de_layers=args.de_layers
        )
    elif args.model == 'TimesNet':
        # TimesNet: 2D convolution based model with FFT period detection
        # Uses Inception blocks for multi-scale feature extraction
        from argparse import Namespace
        configs = Namespace(
            task_name='long_term_forecast',
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=enc_in,
            c_out=enc_in,
            d_model=args.d_model,
            d_ff=args.en_d_ff,
            dropout=args.dropout,
            e_layers=args.en_layers,
            top_k=args.top_k,
            num_kernels=args.num_kernels,
            embed='timeF',
            freq='h'
        )
        model = model_module.Model(configs)
    elif args.model == 'TimeMixer':
        # TimeMixer: Multi-scale season/trend mixing model
        # Uses decomposition and multi-scale mixing layers
        from argparse import Namespace
        configs = Namespace(
            task_name='long_term_forecast',
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=enc_in,
            c_out=enc_in,
            d_model=args.d_model,
            d_ff=args.en_d_ff,
            dropout=args.dropout,
            e_layers=args.en_layers,
            down_sampling_layers=args.down_sampling_layers,
            down_sampling_window=args.down_sampling_window,
            down_sampling_method=args.down_sampling_method,
            channel_independence=args.channel_independence,
            decomp_method=args.decomp_method,
            moving_avg=args.moving_avg,
            top_k=args.top_k,
            use_norm=1,
            embed='timeF',
            freq='h'
        )
        model = model_module.Model(configs)

    return model.to(args.device)


# =============================================================================
# Data Preparation
# =============================================================================
def prepare_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, args):
    """
    Prepare a batch of data for model input.

    This function performs two main operations:
    1. Moves all tensors to the target device (GPU/CPU)
    2. Constructs the decoder input for autoregressive models

    The decoder input is constructed by concatenating:
    - The last `label_len` time steps from batch_y (known context)
    - Zero padding for the `pred_len` time steps to be predicted

    Args:
        batch_x (Tensor): Encoder input sequence.
                         Shape: [batch_size, seq_len, n_features]
        batch_y (Tensor): Target sequence (includes label_len + pred_len).
                         Shape: [batch_size, label_len + pred_len, n_features]
        batch_x_mark (Tensor): Time features for encoder input.
                              Shape: [batch_size, seq_len, time_features]
        batch_y_mark (Tensor): Time features for decoder input.
                              Shape: [batch_size, label_len + pred_len, time_features]
        args (Namespace): Configuration containing device, label_len, pred_len.

    Returns:
        tuple: (batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp)
            - All tensors moved to device
            - dec_inp: Decoder input with zero-masked prediction portion
                      Shape: [batch_size, label_len + pred_len, n_features]

    Note:
        The decoder input structure follows the standard Transformer decoder
        pattern where future positions are masked with zeros during training.
    """
    # Move all tensors to the target device and ensure float32 dtype
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)
    batch_x_mark = batch_x_mark.float().to(args.device)
    batch_y_mark = batch_y_mark.float().to(args.device)

    # Construct decoder input:
    # [label_len known values] + [pred_len zeros to be predicted]
    # This provides the decoder with context while masking future values
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

    return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp


def get_output_slice(outputs, batch_y, args):
    """
    Extract the relevant prediction portion from model outputs.

    Different forecasting tasks require different output handling:
    - 'M' (Multivariate): Predict all variables -> use all channels (f_dim=0)
    - 'S' (Univariate): Predict single variable -> use all channels (f_dim=0)
    - 'MS' (Multivariate-to-Single): Use all variables to predict one -> last channel (f_dim=-1)

    Args:
        outputs (Tensor): Raw model output.
                         Shape: [batch_size, seq_len, n_features]
        batch_y (Tensor): Ground truth target.
                         Shape: [batch_size, label_len + pred_len, n_features]
        args (Namespace): Configuration containing features type and pred_len.

    Returns:
        tuple: (outputs, batch_y) - Both sliced to prediction horizon
            - outputs: Shape [batch_size, pred_len, output_features]
            - batch_y: Shape [batch_size, pred_len, output_features]
    """
    # Determine feature dimension based on forecasting task type
    # MS task: predict only the last (target) variable
    # M/S tasks: predict all variables
    f_dim = -1 if args.features == 'MS' else 0

    # Extract only the prediction horizon portion
    outputs = outputs[:, -args.pred_len:, f_dim:]
    batch_y = batch_y[:, -args.pred_len:, f_dim:]

    return outputs, batch_y


# =============================================================================
# Training and Evaluation Functions
# =============================================================================
def train_epoch(model, train_loader, optimizer, loss_func, args):
    """
    Execute one complete training epoch.

    Performs the standard training loop:
    1. Set model to training mode
    2. Iterate through all batches
    3. Forward pass -> compute loss -> backward pass -> update weights

    Args:
        model (nn.Module): The forecasting model to train.
        train_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer for parameter updates.
        loss_func (nn.Module): Loss function (e.g., MSELoss).
        args (Namespace): Configuration parameters.

    Returns:
        float: Average training loss over all batches.

    Note:
        Gradient accumulation is not used; each batch updates weights immediately.
    """
    model.train()  # Enable training mode (dropout, batch norm updates, etc.)
    train_losses = []

    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        # Reset gradients from previous iteration
        optimizer.zero_grad()

        # Prepare batch data and move to device
        batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, args
        )

        # Forward pass through the model
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # Extract prediction portion and compute loss
        outputs, batch_y = get_output_slice(outputs, batch_y, args)
        loss = loss_func(outputs, batch_y)
        train_losses.append(loss.item())

        # Backward pass and parameter update
        loss.backward()
        optimizer.step()

    return np.mean(train_losses)


def validate(model, val_loader, loss_func, args):
    """
    Evaluate model on validation set.

    Runs inference on the validation set without gradient computation
    to assess model performance during training.

    Args:
        model (nn.Module): The forecasting model to evaluate.
        val_loader (DataLoader): Validation data loader.
        loss_func (nn.Module): Loss function for evaluation.
        args (Namespace): Configuration parameters.

    Returns:
        float: Average validation loss over all batches.

    Note:
        Model is set to eval mode and gradients are disabled for efficiency.
    """
    model.eval()  # Disable dropout, use running stats for batch norm
    val_losses = []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, args
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y = get_output_slice(outputs, batch_y, args)

            loss = loss_func(outputs, batch_y)
            val_losses.append(loss.item())

    return np.mean(val_losses)


def test(model, test_loader, args):
    """
    Run final evaluation on test set and collect predictions.

    Unlike validation, this function collects all predictions and ground truth
    values for comprehensive metric computation (MSE, MAE, etc.).

    Args:
        model (nn.Module): The trained forecasting model.
        test_loader (DataLoader): Test data loader.
        args (Namespace): Configuration parameters.

    Returns:
        tuple: (predictions, ground_truth)
            - predictions: numpy array of shape [n_samples, pred_len, n_features]
            - ground_truth: numpy array of shape [n_samples, pred_len, n_features]

    Note:
        Results are moved to CPU and converted to numpy for metric computation.
    """
    model.eval()
    preds, trues = [], []

    # Determine output feature dimension
    f_dim = -1 if args.features == 'MS' else 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, args
            )

            # Get model predictions
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Extract prediction horizon
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]

            # Move to CPU and convert to numpy, then slice features
            outputs = outputs.detach().cpu().numpy()[:, :, f_dim:]
            batch_y = batch_y.detach().cpu().numpy()[:, :, f_dim:]

            preds.append(outputs)
            trues.append(batch_y)

    # Concatenate all batches along the sample dimension
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    return preds, trues


# =============================================================================
# Main Training Pipeline
# =============================================================================
def work_process(args):
    """
    Execute the complete training and evaluation pipeline.

    This function orchestrates the entire workflow:
    1. Load train/validation/test datasets
    2. Build and initialize the model
    3. Train for specified epochs with validation
    4. Evaluate on test set and compute metrics
    5. Save results to file

    Args:
        args (Namespace): Complete configuration including:
            - Data parameters (data path, sequence lengths)
            - Model parameters (architecture, dimensions)
            - Training parameters (epochs, learning rate, batch size)
            - Output parameters (evaluation file path)

    Side Effects:
        - Prints training progress to stdout
        - Writes evaluation metrics to args.evaluation file
    """
    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    # Load datasets for each split; enc_in is the number of input variables
    train_loader, enc_in = data_provider(args, flag='train')
    val_loader, _ = data_provider(args, flag='val')
    test_loader, _ = data_provider(args, flag='test')

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------
    model = build_model(args, enc_in)
    loss_func = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.train_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_func, args)

        # Evaluate on validation set
        val_loss = validate(model, val_loader, loss_func, args)

        # Log progress
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.7f} Vali Loss: {val_loss:.7f}")

        # Adjust learning rate according to schedule
        adjust_learning_rate(optimizer, epoch + 1, args)

    # -------------------------------------------------------------------------
    # Testing and Evaluation
    # -------------------------------------------------------------------------
    preds, trues = test(model, test_loader, args)
    print(f'Test shape: {preds.shape}, {trues.shape}')

    # Reshape for metric computation: flatten batch dimension if needed
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print(f'Test shape: {preds.shape}, {trues.shape}')

    # Compute evaluation metrics
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'MSE: {mse}, MAE: {mae}')

    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.evaluation), exist_ok=True)

    # Append results to evaluation file
    with open(args.evaluation, 'a') as f:
        f.write(f'MSE: {mse}, MAE: {mae}\n')


# =============================================================================
# Command Line Interface
# =============================================================================
def run():
    """
    Parse command-line arguments and launch training.

    This function defines all configurable parameters for the training pipeline
    and validates the configuration before starting training.

    Command-line Arguments:
        Basic Configuration:
            --model: Model architecture to use
            --evaluation: Directory for saving evaluation results
            --data: Dataset name
            --root_path: Root directory containing data files
            --gpu: GPU device ID (optional, auto-detects if not specified)

        Sequence Parameters:
            --seq_len: Input sequence length (lookback window)
            --label_len: Decoder start token length (overlap with encoder)
            --patch_len: Patch size for patch-based models
            --pred_len: Prediction horizon length
            --features: Forecasting task type (M/S/MS)
            --target: Target variable name for S/MS tasks

        Model Architecture:
            --d_model: Embedding dimension
            --dropout: Dropout rate
            --factor: Attention sparsity factor
            --n_heads: Number of attention heads
            --en_d_ff: Encoder feed-forward dimension
            --de_d_ff: Decoder feed-forward dimension
            --en_layers: Number of encoder layers
            --de_layers: Number of decoder layers

        Training:
            --batch_size: Training batch size
            --train_epochs: Number of training epochs
            --learning_rate: Initial learning rate
            --lradj: Learning rate adjustment strategy
    """
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # -------------------------------------------------------------------------
    # Basic Configuration
    # -------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='PAViT',
                        help=f'Model architecture. Options: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('--evaluation', type=str, default='./evaluation/',
                        help='Directory for saving evaluation results')
    parser.add_argument('--data', type=str, default='ETTh1',
                        help='Dataset name: [ETTh1, ETTm1, ETTh2, ETTm2, electricity, exchange_rate, traffic, weather]')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small/',
                        help='Root directory containing the data files')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID. If not specified, auto-detects best available device')

    # -------------------------------------------------------------------------
    # Sequence Parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length (lookback window size)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Decoder start token length (overlap between encoder and decoder)')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch size for patch-based models. Must divide seq_len evenly')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction horizon length (how far ahead to forecast)')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task: M (multivariate), S (univariate), MS (multivariate-to-single)')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target variable name for S/MS forecasting tasks')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='Number of input variables (auto-detected from data)')

    # -------------------------------------------------------------------------
    # Model Architecture Parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('--factor', type=int, default=3,
                        help='Attention factor for ProbSparse attention')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--en_d_ff', type=int, default=2048,
                        help='Encoder feed-forward network dimension (typically 4x d_model)')
    parser.add_argument('--de_d_ff', type=int, default=2048,
                        help='Decoder feed-forward network dimension (typically 4x d_model)')
    parser.add_argument('--en_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--de_layers', type=int, default=2,
                        help='Number of decoder layers')

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate for Adam optimizer')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='Loss function (currently only MSE supported)')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='Learning rate adjustment: type1 (exponential decay) or type2 (step schedule)')

    # -------------------------------------------------------------------------
    # TimesNet Specific Parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top k frequencies for TimesNet/TimeMixer')
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='Number of kernels for Inception block in TimesNet')

    # -------------------------------------------------------------------------
    # TimeMixer Specific Parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--down_sampling_layers', type=int, default=2,
                        help='Number of down sampling layers for TimeMixer')
    parser.add_argument('--down_sampling_window', type=int, default=2,
                        help='Down sampling window size for TimeMixer')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='Down sampling method: avg, max, conv')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='Channel independence for TimeMixer (0 or 1)')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='Decomposition method: moving_avg or dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='Moving average window size for decomposition')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Configuration Validation
    # -------------------------------------------------------------------------
    # Ensure patch_len divides seq_len evenly for patch-based models
    if args.seq_len % args.patch_len != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be divisible by patch_len ({args.patch_len})")

    # -------------------------------------------------------------------------
    # Device Setup
    # -------------------------------------------------------------------------
    args.device = get_device(args.gpu)
    print(f'Using device: {args.device}')

    # -------------------------------------------------------------------------
    # Output Path Configuration
    # -------------------------------------------------------------------------
    # Build descriptive filename encoding all experiment parameters
    file_name = (f"{args.model}{args.data}PatL{args.patch_len}PreL{args.pred_len}"
                 f"Fea{args.features}DM{args.d_model}ED{args.en_d_ff}"
                 f"DD{args.de_d_ff}EL{args.en_layers}DL{args.de_layers}")
    args.evaluation = os.path.join(args.evaluation, f'{file_name}.txt')

    # Print full configuration for logging
    print(args)

    # Launch training pipeline
    work_process(args)

    # Clean up GPU memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    # Set random seeds for reproducibility across all libraries
    fix_seed = 42

    # Limit CPU threads to prevent over-subscription on multi-core systems
    torch.set_num_threads(4)

    # Set seeds for all random number generators
    random.seed(fix_seed)           # Python random
    torch.manual_seed(fix_seed)     # PyTorch CPU
    np.random.seed(fix_seed)        # NumPy

    # Note: For full reproducibility with CUDA, also set:
    # torch.cuda.manual_seed(fix_seed)
    # torch.cuda.manual_seed_all(fix_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    run()
