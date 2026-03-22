# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Time Feature Extraction Module

This module provides utilities for extracting temporal features from timestamps.
It converts datetime information into normalized numerical features suitable for
machine learning models.

Features:
    - SecondOfMinute: Second within a minute (0-59)
    - MinuteOfHour: Minute within an hour (0-59)
    - HourOfDay: Hour within a day (0-23)
    - DayOfWeek: Day within a week (0-6)
    - DayOfMonth: Day within a month (1-31)
    - DayOfYear: Day within a year (1-365)
    - MonthOfYear: Month within a year (1-12)
    - WeekOfYear: Week within a year (1-52)

Author: GluonTS Team (adapted for PAViT)
"""

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


# =============================================================================
# Base Time Feature Class
# =============================================================================
class TimeFeature:
    """
    Base class for time features.

    All time features inherit from this class and implement the __call__ method
    to extract a specific temporal component from a DatetimeIndex.
    """

    def __init__(self):
        """Initialize time feature."""
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract time feature from datetime index.

        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from

        Returns:
            np.ndarray: Extracted feature values
        """
        pass

    def __repr__(self):
        """String representation of the time feature."""
        return self.__class__.__name__ + "()"


# =============================================================================
# Time Feature Implementations
# =============================================================================
class SecondOfMinute(TimeFeature):
    """
    Second of minute encoded as value between [-0.5, 0.5].

    Extracts the second component (0-59) and normalizes to [-0.5, 0.5] range.
    This normalization helps the model treat the feature as circular
    (59 seconds is close to 0 seconds).
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract second of minute."""
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    Minute of hour encoded as value between [-0.5, 0.5].

    Extracts the minute component (0-59) and normalizes to [-0.5, 0.5] range.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract minute of hour."""
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5].

    Extracts the hour component (0-23) and normalizes to [-0.5, 0.5] range.
    Useful for capturing daily patterns in time series data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract hour of day."""
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    Day of week encoded as value between [-0.5, 0.5].

    Extracts the day of week (0-6, Monday-Sunday) and normalizes to [-0.5, 0.5] range.
    Useful for capturing weekly patterns in time series data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of week."""
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    Day of month encoded as value between [-0.5, 0.5].

    Extracts the day of month (1-31) and normalizes to [-0.5, 0.5] range.
    Useful for capturing monthly patterns in time series data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of month."""
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    Day of year encoded as value between [-0.5, 0.5].

    Extracts the day of year (1-365) and normalizes to [-0.5, 0.5] range.
    Useful for capturing seasonal patterns in time series data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract day of year."""
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    Month of year encoded as value between [-0.5, 0.5].

    Extracts the month component (1-12) and normalizes to [-0.5, 0.5] range.
    Useful for capturing seasonal patterns in time series data.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract month of year."""
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    Week of year encoded as value between [-0.5, 0.5].

    Extracts the ISO week number (1-52) and normalizes to [-0.5, 0.5] range.
    Useful for capturing weekly patterns across the year.
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extract week of year."""
        return (index.isocalendar().week - 1) / 52.0 - 0.5


# =============================================================================
# Time Feature Selection
# =============================================================================
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Select appropriate time features based on data frequency.

    Returns a list of time features that are relevant for the given frequency.
    For example, hourly data includes hour, day of week, day of month, and day of year.
    Yearly data includes no additional features.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity]
                       Examples: "12H" (12 hourly), "5min" (5 minutely), "1D" (daily)

    Returns:
        List[TimeFeature]: List of time feature extractors appropriate for the frequency

    Raises:
        RuntimeError: If the frequency string is not supported

    Supported Frequencies:
        - Y, A: Yearly
        - M: Monthly
        - W: Weekly
        - D: Daily
        - B: Business days
        - H: Hourly
        - T, min: Minutely
        - S: Secondly

    Example:
        >>> features = time_features_from_frequency_str('H')
        >>> # Returns [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
    """

    # Map frequency offsets to their corresponding time features
    # Features are ordered from finest to coarsest granularity
    features_by_offsets = {
        offsets.YearEnd: [],  # Yearly: no additional features
        offsets.QuarterEnd: [MonthOfYear],  # Quarterly: month
        offsets.MonthEnd: [MonthOfYear],  # Monthly: month
        offsets.Week: [DayOfMonth, WeekOfYear],  # Weekly: day of month, week of year
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # Daily: day of week, day of month, day of year
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # Business days: same as daily
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # Hourly: hour, day of week, day of month, day of year
        offsets.Minute: [  # Minutely: minute, hour, day of week, day of month, day of year
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [  # Secondly: second, minute, hour, day of week, day of month, day of year
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # Convert frequency string to offset object
    offset = to_offset(freq_str)

    # Find matching offset type and return corresponding features
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # Raise error if frequency is not supported
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


# =============================================================================
# Time Feature Extraction
# =============================================================================
def time_features(dates, freq='h'):
    """
    Extract time features from datetime index.

    Extracts multiple time features appropriate for the given frequency
    and stacks them into a 2D array.

    Args:
        dates (pd.DatetimeIndex or np.ndarray): Datetime index or array of timestamps
        freq (str): Frequency string (default: 'h' for hourly)
                   Examples: 'h' (hourly), 'd' (daily), 't' (minutely)

    Returns:
        np.ndarray: Time features array of shape [n_features, n_timestamps]
                   Each row contains one type of time feature

    Example:
        >>> dates = pd.date_range('2020-01-01', periods=24, freq='H')
        >>> features = time_features(dates, freq='h')
        >>> features.shape
        (4, 24)  # 4 features (hour, day of week, day of month, day of year) for 24 timestamps
    """
    # Convert numpy array to DatetimeIndex if necessary
    if isinstance(dates, np.ndarray):
        dates = pd.DatetimeIndex(dates)
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
