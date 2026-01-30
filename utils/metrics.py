"""
Evaluation Metrics Module

This module provides various evaluation metrics for time series forecasting tasks.
It includes both error metrics (MAE, MSE, RMSE, MAPE, MSPE) and correlation metrics.

Metrics:
    - RSE: Relative Squared Error
    - CORR: Correlation coefficient
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - MSPE: Mean Squared Percentage Error

Author: PAViT Team
"""

import numpy as np


# =============================================================================
# Error Metrics
# =============================================================================
def RSE(pred, true):
    """
    Relative Squared Error (RSE).

    Measures the ratio of prediction error to the variance of the ground truth.
    Lower values indicate better predictions.

    Formula:
        RSE = sqrt(sum((true - pred)^2) / sum((true - mean(true))^2))

    Args:
        pred (ndarray): Predicted values
        true (ndarray): Ground truth values

    Returns:
        float: RSE value
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Correlation Coefficient.

    Measures the linear correlation between predictions and ground truth.
    Values range from -1 to 1, with 1 indicating perfect positive correlation.

    Formula:
        CORR = mean(cov(true, pred) / (std(true) * std(pred)))

    Args:
        pred (ndarray): Predicted values [n_samples, seq_len, n_features]
        true (ndarray): Ground truth values [n_samples, seq_len, n_features]

    Returns:
        float: Average correlation coefficient across all features
    """
    # Compute covariance between true and pred
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)

    # Compute standard deviations
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))

    # Return mean correlation across features
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Mean Absolute Error (MAE).

    Measures the average absolute difference between predictions and ground truth.
    More robust to outliers than MSE.

    Formula:
        MAE = mean(|true - pred|)

    Args:
        pred (ndarray): Predicted values
        true (ndarray): Ground truth values

    Returns:
        float: MAE value
    """
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    """
    Mean Squared Error (MSE).

    Measures the average squared difference between predictions and ground truth.
    Penalizes large errors more heavily than MAE.

    Formula:
        MSE = mean((true - pred)^2)

    Args:
        pred (ndarray): Predicted values
        true (ndarray): Ground truth values

    Returns:
        float: MSE value
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    """
    Root Mean Squared Error (RMSE).

    Square root of MSE, providing error in the same units as the original data.
    Commonly used metric for regression tasks.

    Formula:
        RMSE = sqrt(MSE)

    Args:
        pred (ndarray): Predicted values
        true (ndarray): Ground truth values

    Returns:
        float: RMSE value
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Mean Absolute Percentage Error (MAPE).

    Measures the average percentage difference between predictions and ground truth.
    Useful for comparing errors across different scales.

    Formula:
        MAPE = mean(|true - pred| / |true|)

    Args:
        pred (ndarray): Predicted values
        true (ndarray): Ground truth values

    Returns:
        float: MAPE value (as a decimal, not percentage)

    Note:
        Can be undefined or very large when true values are close to zero.
    """
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    """
    Mean Squared Percentage Error (MSPE).

    Measures the average squared percentage difference between predictions and ground truth.
    Similar to MAPE but squares the percentage errors.

    Formula:
        MSPE = mean(((true - pred) / true)^2)

    Args:
        pred (ndarray): Predicted values
        true (ndarray): Ground truth values

    Returns:
        float: MSPE value

    Note:
        Can be undefined or very large when true values are close to zero.
    """
    return np.mean(np.square((true - pred) / true))


# =============================================================================
# Metric Aggregation
# =============================================================================
def metric(pred, true):
    """
    Compute all evaluation metrics.

    Computes MAE, MSE, RMSE, MAPE, and MSPE for comprehensive evaluation
    of forecasting performance.

    Args:
        pred (ndarray): Predicted values [n_samples, seq_len, n_features]
        true (ndarray): Ground truth values [n_samples, seq_len, n_features]

    Returns:
        tuple: (mae, mse, rmse, mape, mspe)
            - mae (float): Mean Absolute Error
            - mse (float): Mean Squared Error
            - rmse (float): Root Mean Squared Error
            - mape (float): Mean Absolute Percentage Error
            - mspe (float): Mean Squared Percentage Error

    Example:
        >>> pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> true = np.array([[1.1, 2.1], [2.9, 3.9]])
        >>> mae, mse, rmse, mape, mspe = metric(pred, true)
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
