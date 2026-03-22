"""
Training Utilities Module

This module provides utility functions and classes for model training, evaluation,
and visualization. It includes learning rate scheduling, early stopping, data
normalization, and result visualization tools.

Key Components:
    - clever_format: Format large numbers with SI prefixes
    - adjust_learning_rate: Learning rate scheduling strategies
    - EarlyStopping: Early stopping mechanism for training
    - StandardScaler: Data normalization and denormalization
    - visual: Visualization of predictions vs ground truth
    - adjustment: Anomaly detection result adjustment
    - cal_accuracy: Accuracy calculation for classification

Author: PAViT Team
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable

# Use non-interactive backend for server environments
plt.switch_backend('agg')


# =============================================================================
# Number Formatting
# =============================================================================
def clever_format(nums, format="%.2f"):
    """
    Format numbers with SI prefixes for readability.

    Converts large numbers to human-readable format with SI suffixes:
    - T (Tera): 10^12
    - G (Giga): 10^9
    - M (Mega): 10^6
    - K (Kilo): 10^3
    - B (Base): < 10^3

    Args:
        nums (int/float/list): Number(s) to format
        format (str): Format string for decimal places (default: "%.2f")

    Returns:
        str or tuple: Formatted number(s) with SI suffix
            - Single number: returns string
            - Multiple numbers: returns tuple of strings

    Examples:
        >>> clever_format(1234567890)
        '1.23G'
        >>> clever_format([1000, 1000000, 1000000000])
        ('1.00K', '1.00M', '1.00G')
    """
    # Convert single number to list for uniform processing
    if not isinstance(nums, Iterable):
        nums = [nums]

    clever_nums = []

    for num in nums:
        # Apply SI prefix based on magnitude
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    # Return single string or tuple based on input
    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


# =============================================================================
# Learning Rate Scheduling
# =============================================================================
def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust learning rate based on predefined schedule.

    Supports two learning rate adjustment strategies:
    1. type1: Exponential decay - halves learning rate every epoch
    2. type2: Step-based schedule - adjusts LR at specific epochs

    The learning rate schedule is crucial for training stability and convergence.
    Different schedules work better for different datasets and models.

    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer with parameter groups
        epoch (int): Current epoch number (1-indexed)
        args (Namespace): Configuration containing:
            - lradj (str): Learning rate adjustment type ('type1' or 'type2')
            - learning_rate (float): Initial learning rate (for type1)

    Side Effects:
        - Updates learning rate in optimizer parameter groups
        - Prints learning rate update to stdout

    Examples:
        Type1 (Exponential Decay):
            Epoch 1: lr = 0.0001
            Epoch 2: lr = 0.00005
            Epoch 3: lr = 0.000025
            ...

        Type2 (Step Schedule):
            Epoch 2: lr = 5e-5
            Epoch 4: lr = 1e-5
            Epoch 6: lr = 5e-6
            ...
    """
    if args.lradj == 'type1':
        # Exponential decay: multiply by 0.5 for each epoch
        # Formula: lr = initial_lr * (0.5 ^ (epoch - 1))
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

    elif args.lradj == 'type2':
        # Step-based schedule: predefined LR at specific epochs
        # Useful for fine-tuning and convergence in later stages
        lr_adjust = {
            2: 5e-5,   # Epoch 2
            4: 1e-5,   # Epoch 4
            6: 5e-6,   # Epoch 6
            8: 1e-6,   # Epoch 8
            10: 5e-7,  # Epoch 10
            15: 1e-7,  # Epoch 15
            20: 5e-8   # Epoch 20
        }
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f'Updating learning rate to {lr}')


# =============================================================================
# Early Stopping
# =============================================================================
class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.

    Monitors validation loss and stops training if no improvement is observed
    for a specified number of epochs (patience). This prevents the model from
    overfitting to the training data.

    Attributes:
        patience (int): Number of epochs with no improvement before stopping
        verbose (bool): Whether to print detailed messages
        counter (int): Number of epochs without improvement
        best_score (float): Best validation score seen so far
        early_stop (bool): Flag indicating whether to stop training
        val_loss_min (float): Minimum validation loss seen
        delta (float): Minimum change to qualify as an improvement

    Example:
        >>> early_stopping = EarlyStopping(patience=7, verbose=True)
        >>> for epoch in range(100):
        ...     val_loss = train_one_epoch()
        ...     early_stopping(val_loss, model, checkpoint_dir)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize early stopping.

        Args:
            patience (int): Number of epochs with no improvement before stopping
            verbose (bool): Whether to print checkpoint messages
            delta (float): Minimum change in validation loss to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to save if improvement is found
            path (str): Directory to save model checkpoint
        """
        score = -val_loss  # Negative because lower loss is better

        if self.best_score is None:
            # First epoch: save checkpoint
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # No improvement: increment counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found: reset counter and save checkpoint
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save model checkpoint when validation loss improves.

        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to save
            path (str): Directory to save checkpoint
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


# =============================================================================
# Dictionary with Dot Notation Access
# =============================================================================
class dotdict(dict):
    """
    Dictionary that supports dot notation access.

    Allows accessing dictionary values using dot notation (obj.key) instead of
    bracket notation (obj['key']). Useful for configuration objects.

    Example:
        >>> config = dotdict({'learning_rate': 0.001, 'batch_size': 32})
        >>> config.learning_rate
        0.001
        >>> config.batch_size
        32
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# =============================================================================
# Data Normalization
# =============================================================================
class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.

    This is a custom implementation of StandardScaler for time series data.
    It stores mean and standard deviation computed from training data and
    applies the same transformation to all splits (train/val/test).

    Attributes:
        mean (ndarray): Mean values computed from training data
        std (ndarray): Standard deviation computed from training data

    Note:
        Unlike sklearn's StandardScaler, this version requires pre-computed
        mean and std values, preventing data leakage during preprocessing.

    Example:
        >>> scaler = StandardScaler(mean=train_mean, std=train_std)
        >>> normalized = scaler.transform(data)
        >>> original = scaler.inverse_transform(normalized)
    """

    def __init__(self, mean, std):
        """
        Initialize scaler with pre-computed statistics.

        Args:
            mean (ndarray): Mean values from training data
            std (ndarray): Standard deviation from training data
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Normalize data using stored mean and std.

        Formula: (data - mean) / std

        Args:
            data (ndarray): Data to normalize

        Returns:
            ndarray: Normalized data with zero mean and unit variance
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverse normalization to original scale.

        Formula: data * std + mean

        Args:
            data (ndarray): Normalized data

        Returns:
            ndarray: Data in original scale
        """
        return (data * self.std) + self.mean


# =============================================================================
# Visualization
# =============================================================================
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualize ground truth and predictions.

    Creates a line plot comparing ground truth values with model predictions.
    Useful for qualitative evaluation of forecasting performance.

    Args:
        true (ndarray): Ground truth values
        preds (ndarray, optional): Predicted values. If None, only plots ground truth.
        name (str): Output file path for saving the plot (default: './pic/test.pdf')

    Side Effects:
        - Creates and saves a matplotlib figure
        - Requires directory to exist or will raise error

    Example:
        >>> true_values = np.array([1, 2, 3, 4, 5])
        >>> pred_values = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> visual(true_values, pred_values, './results/forecast.pdf')
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


# =============================================================================
# Anomaly Detection Utilities
# =============================================================================
def adjustment(gt, pred):
    """
    Adjust anomaly detection predictions based on ground truth patterns.

    This function implements a post-processing step for anomaly detection results.
    It fills in missing anomaly predictions within continuous anomaly regions
    based on ground truth labels.

    Algorithm:
        1. Iterate through ground truth labels
        2. When an anomaly is detected (gt[i]==1 and pred[i]==1):
           - Mark anomaly_state as True
           - Fill backward: set pred[j]=1 for all j < i where gt[j]==1
           - Fill forward: set pred[j]=1 for all j > i where gt[j]==1
        3. Reset anomaly_state when gt[i]==0

    This adjustment helps recover missed anomalies within known anomaly periods.

    Args:
        gt (ndarray): Ground truth binary labels (0=normal, 1=anomaly)
        pred (ndarray): Predicted binary labels (0=normal, 1=anomaly)

    Returns:
        tuple: (gt, pred) - Adjusted predictions

    Example:
        >>> gt = np.array([0, 1, 1, 1, 0, 1, 1, 0])
        >>> pred = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        >>> gt_adj, pred_adj = adjustment(gt, pred)
        >>> pred_adj
        array([0, 1, 1, 1, 0, 1, 1, 0])
    """
    anomaly_state = False

    for i in range(len(gt)):
        # Detect start of anomaly region
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True

            # Fill backward: mark all preceding anomalies in gt as predicted
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1

            # Fill forward: mark all following anomalies in gt as predicted
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1

        # Reset anomaly state when normal region is reached
        elif gt[i] == 0:
            anomaly_state = False

        # Continue marking as anomaly while in anomaly state
        if anomaly_state:
            pred[i] = 1

    return gt, pred


# =============================================================================
# Classification Metrics
# =============================================================================
def cal_accuracy(y_pred, y_true):
    """
    Calculate classification accuracy.

    Computes the fraction of correct predictions.

    Formula: accuracy = (number of correct predictions) / (total predictions)

    Args:
        y_pred (ndarray): Predicted labels
        y_true (ndarray): Ground truth labels

    Returns:
        float: Accuracy score in range [0, 1]

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> cal_accuracy(y_pred, y_true)
        0.8
    """
    return np.mean(y_pred == y_true)
