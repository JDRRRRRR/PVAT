"""
Time Series Dataset Loading Module

This module provides PyTorch Dataset implementations for loading and preprocessing
time series data from various sources. It supports multiple dataset types with
different temporal granularities and split strategies.

Supported Datasets:
    - ETT (Electricity Transformer Temperature): Hourly and minute-level data
    - Custom datasets: Electricity, weather, traffic, exchange rate, etc.

Key Features:
    - Automatic data normalization using StandardScaler
    - Flexible train/val/test splitting strategies
    - Time feature extraction (month, day, weekday, hour, minute)
    - Support for multivariate, univariate, and multivariate-to-single forecasting
    - Efficient numpy-based data loading

Author: PAViT Team
"""

import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


# =============================================================================
# Base Dataset Class
# =============================================================================
class BaseTimeSeriesDataset(Dataset):
    """
    Base class for time series datasets with common functionality.

    This abstract base class implements the core data loading and preprocessing
    logic shared across all time series datasets. Subclasses override specific
    methods to handle dataset-specific splitting and preprocessing.

    Architecture:
        1. __init__: Initialize dataset parameters
        2. _read_data: Load and preprocess data (calls template methods)
        3. _get_borders: Define train/val/test split boundaries (subclass override)
        4. _preprocess_dataframe: Optional dataframe preprocessing (subclass override)
        5. _extract_time_features: Extract temporal features from timestamps
        6. __getitem__: Return a single sample with encoder/decoder inputs
        7. __len__: Return dataset size

    Attributes:
        seq_len (int): Input sequence length (lookback window)
        label_len (int): Decoder start token length (overlap with encoder)
        pred_len (int): Prediction horizon length
        features (str): Forecasting task type ('M', 'S', or 'MS')
        enc_in (int): Number of input variables/channels
        scaler (StandardScaler): For data normalization
        data_x (ndarray): Input sequences [n_samples, seq_len, n_features]
        data_y (ndarray): Target sequences [n_samples, label_len+pred_len, n_features]
        data_stamp (ndarray): Time features [n_samples, seq_len, n_time_features]
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        """
        Initialize the dataset.

        Args:
            args (Namespace): Configuration object (currently unused but kept for compatibility)
            root_path (str): Root directory containing data files
            flag (str): Data split type - 'train', 'val', or 'test'
            size (list, optional): [seq_len, label_len, pred_len]. If None, uses defaults.
            features (str): Forecasting task:
                - 'M': Multivariate (predict all variables)
                - 'S': Univariate (predict single variable)
                - 'MS': Multivariate-to-single (use all variables to predict one)
            data_path (str): Filename of the CSV data file
            target (str): Name of target column for 'S' and 'MS' tasks
            scale (bool): Whether to normalize data using StandardScaler
            timeenc (int): Time encoding type:
                - 0: Discrete features (month, day, weekday, hour, minute)
                - 1: Continuous features from time_features function
            freq (str): Frequency of data ('h' for hourly, 't' for minute, etc.)
            seasonal_patterns (str, optional): Seasonal pattern type (unused)
        """
        self.args = args

        # =====================================================================
        # Sequence Length Configuration
        # =====================================================================
        # Set default sequence lengths if not provided
        # Defaults: 96 hours of input, 48 hours of overlap, 96 hours to predict
        if size is None:
            self.seq_len = 24 * 4 * 4      # 96 time steps
            self.label_len = 24 * 4        # 48 time steps
            self.pred_len = 24 * 4         # 96 time steps
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # =====================================================================
        # Data Split Configuration
        # =====================================================================
        # Map split type string to numeric index for border selection
        assert flag in ['train', 'test', 'val'], f"Invalid flag: {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # =====================================================================
        # Feature and Preprocessing Configuration
        # =====================================================================
        self.features = features      # Forecasting task type
        self.target = target          # Target variable name
        self.scale = scale            # Whether to normalize data
        self.timeenc = timeenc        # Time encoding type
        self.freq = freq              # Data frequency
        self.root_path = root_path    # Data directory
        self.data_path = data_path    # Data filename

        # Load and preprocess data
        self._read_data()

    # =========================================================================
    # Template Methods (to be overridden by subclasses)
    # =========================================================================
    def _get_borders(self, df_raw):
        """
        Calculate train/val/test split boundaries.

        This method must be implemented by subclasses to define how the dataset
        is split into train, validation, and test sets. Different datasets may
        have different split strategies (fixed time-based vs. percentage-based).

        Args:
            df_raw (DataFrame): Raw data loaded from CSV

        Returns:
            tuple: (border1s, border2s) where:
                - border1s: List of 3 start indices [train_start, val_start, test_start]
                - border2s: List of 3 end indices [train_end, val_end, test_end]

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _get_borders()")

    def _preprocess_dataframe(self, df_raw):
        """
        Preprocess dataframe before splitting (optional).

        This method can be overridden by subclasses to perform dataset-specific
        preprocessing such as column reordering or data cleaning.

        Args:
            df_raw (DataFrame): Raw data loaded from CSV

        Returns:
            DataFrame: Preprocessed dataframe
        """
        return df_raw

    def _add_extra_time_features(self, df_stamp):
        """
        Add extra time features for specific datasets (optional).

        This method can be overridden by subclasses to add dataset-specific
        time features beyond the standard month/day/weekday/hour.

        Args:
            df_stamp (DataFrame): DataFrame with date column and basic time features
        """
        pass

    # =========================================================================
    # Core Data Loading and Preprocessing
    # =========================================================================
    def _extract_time_features(self, df_stamp):
        """
        Extract temporal features from timestamp column.

        Supports two encoding modes:
        1. Discrete (timeenc=0): Extracts month, day, weekday, hour, minute
        2. Continuous (timeenc=1): Uses time_features function for continuous encoding

        Args:
            df_stamp (DataFrame): DataFrame with 'date' column containing timestamps

        Returns:
            ndarray: Time features array of shape [n_samples, n_time_features]
        """
        df_stamp = df_stamp.copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            # Discrete time encoding: extract individual time components
            df_stamp['month'] = df_stamp['date'].dt.month      # 1-12
            df_stamp['day'] = df_stamp['date'].dt.day          # 1-31
            df_stamp['weekday'] = df_stamp['date'].dt.weekday  # 0-6 (Mon-Sun)
            df_stamp['hour'] = df_stamp['date'].dt.hour        # 0-23

            # Allow subclasses to add extra features (e.g., minute for minute-level data)
            self._add_extra_time_features(df_stamp)

            # Return all features except the original date column
            return df_stamp.drop(columns=['date']).values
        else:
            # Continuous time encoding: use specialized time feature extraction
            data_stamp = time_features(df_stamp['date'].values, freq=self.freq)
            # Transpose from [n_features, n_samples] to [n_samples, n_features]
            return data_stamp.transpose(1, 0)

    def _read_data(self):
        """
        Load and preprocess the complete dataset.

        This method orchestrates the entire data loading pipeline:
        1. Load CSV file
        2. Preprocess dataframe (subclass-specific)
        3. Calculate train/val/test split boundaries
        4. Select features based on task type
        5. Normalize data if requested
        6. Extract time features
        7. Store processed data in memory

        The method uses template methods (_get_borders, _preprocess_dataframe,
        _extract_time_features) to allow subclass customization.
        """
        # Initialize scaler for data normalization
        self.scaler = StandardScaler()

        # Load raw data from CSV
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Apply dataset-specific preprocessing
        df_raw = self._preprocess_dataframe(df_raw)

        # Get split boundaries from subclass implementation
        border1s, border2s = self._get_borders(df_raw)
        border1 = border1s[self.set_type]  # Start index for current split
        border2 = border2s[self.set_type]  # End index for current split

        # =====================================================================
        # Feature Selection
        # =====================================================================
        # Select which columns to use based on forecasting task
        if self.features in ['M', 'MS']:
            # Multivariate: use all columns except date (first column)
            cols_data = df_raw.columns[1:]
            self.enc_in = len(cols_data)
            df_data = df_raw[cols_data]
        else:  # 'S' - Univariate
            # Univariate: use only the target column
            self.enc_in = 1
            df_data = df_raw[[self.target]]

        # =====================================================================
        # Data Normalization
        # =====================================================================
        # Fit scaler on training data and apply to all splits
        # This prevents data leakage from val/test to training
        if self.scale:
            # Fit scaler on training data only
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # Transform all data using training statistics
            data = self.scaler.transform(df_data.values)
        else:
            # Use raw data without normalization
            data = df_data.values

        # =====================================================================
        # Time Feature Extraction
        # =====================================================================
        # Extract time features for the current split
        df_stamp = df_raw[['date']][border1:border2]
        self.data_stamp = self._extract_time_features(df_stamp)

        # =====================================================================
        # Store Processed Data
        # =====================================================================
        # Extract data for current split
        # Note: data_x and data_y are the same (input and target are identical)
        # The actual input/target split happens in __getitem__
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    # =========================================================================
    # PyTorch Dataset Interface
    # =========================================================================
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Constructs encoder and decoder inputs from the stored data:
        - Encoder input: seq_len consecutive time steps
        - Decoder input: label_len overlapping steps + pred_len future steps

        The decoder input structure:
            [label_len known values from target] + [pred_len zeros to predict]

        Args:
            index (int): Sample index (0 to len(self)-1)

        Returns:
            tuple: (seq_x, seq_y, seq_x_mark, seq_y_mark)
                - seq_x: Encoder input [seq_len, n_features]
                - seq_y: Decoder target [label_len + pred_len, n_features]
                - seq_x_mark: Encoder time features [seq_len, n_time_features]
                - seq_y_mark: Decoder time features [label_len + pred_len, n_time_features]
        """
        # Calculate slice boundaries
        s_begin = index                              # Encoder start
        s_end = s_begin + self.seq_len              # Encoder end
        r_begin = s_end - self.label_len            # Decoder start (overlaps with encoder)
        r_end = r_begin + self.label_len + self.pred_len  # Decoder end

        # Extract sequences
        seq_x = self.data_x[s_begin:s_end]          # [seq_len, n_features]
        seq_y = self.data_y[r_begin:r_end]          # [label_len + pred_len, n_features]
        seq_x_mark = self.data_stamp[s_begin:s_end] # [seq_len, n_time_features]
        seq_y_mark = self.data_stamp[r_begin:r_end] # [label_len + pred_len, n_time_features]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Return the number of valid samples in the dataset.

        A valid sample requires:
        - seq_len time steps for encoder input
        - label_len + pred_len time steps for decoder target
        - Total: seq_len + pred_len time steps

        Returns:
            int: Number of valid samples
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Reverse the normalization applied during preprocessing.

        Converts normalized predictions back to original data scale for
        evaluation and visualization.

        Args:
            data (ndarray): Normalized data

        Returns:
            ndarray: Data in original scale
        """
        return self.scaler.inverse_transform(data)


# =============================================================================
# ETT Dataset - Hourly Granularity
# =============================================================================
class Dataset_ETT_hour(BaseTimeSeriesDataset):
    """
    ETT (Electricity Transformer Temperature) dataset with hourly granularity.

    Dataset Characteristics:
        - Frequency: Hourly (24 samples per day)
        - Duration: ~2 years of data
        - Variables: 7 (OT, HUFL, HULL, MUFL, MULL, LUFL, LULL)
        - Split: 12 months train, 4 months val, 4 months test

    Split Strategy (Fixed Time-Based):
        - Train: First 12 months (12 * 30 * 24 = 10,080 hours)
        - Val: Next 4 months (4 * 30 * 24 = 2,880 hours)
        - Test: Final 4 months (4 * 30 * 24 = 2,880 hours)

    Note:
        Uses approximate month length (30 days) for simplicity.
        Actual month lengths vary but this is standard in time series literature.
    """

    def _get_borders(self, df_raw):
        """
        Calculate train/val/test split boundaries for hourly ETT data.

        Returns:
            tuple: (border1s, border2s) with indices for each split
        """
        # Approximate month length: 30 days * 24 hours = 720 hours
        # 12 months train, 4 months val, 4 months test
        border1s = [
            0,                                    # Train start
            12 * 30 * 24 - self.seq_len,         # Val start (with overlap)
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len  # Test start (with overlap)
        ]
        border2s = [
            12 * 30 * 24,                        # Train end
            12 * 30 * 24 + 4 * 30 * 24,         # Val end
            12 * 30 * 24 + 8 * 30 * 24          # Test end
        ]
        return border1s, border2s


# =============================================================================
# ETT Dataset - Minute Granularity
# =============================================================================
class Dataset_ETT_minute(BaseTimeSeriesDataset):
    """
    ETT (Electricity Transformer Temperature) dataset with minute granularity.

    Dataset Characteristics:
        - Frequency: 15-minute intervals (96 samples per day)
        - Duration: ~2 years of data
        - Variables: 7 (same as hourly version)
        - Split: 12 months train, 4 months val, 4 months test

    Split Strategy (Fixed Time-Based):
        - Train: First 12 months (12 * 30 * 24 * 4 = 40,320 samples)
        - Val: Next 4 months (4 * 30 * 24 * 4 = 11,520 samples)
        - Test: Final 4 months (4 * 30 * 24 * 4 = 11,520 samples)

    Note:
        Minute-level data has 4x more samples than hourly data.
        The minute feature is extracted as minute // 15 (0-3 for 15-min intervals).
    """

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        """
        Initialize minute-level ETT dataset.

        Args:
            freq (str): Set to 't' for minute-level frequency encoding
            data_path (str): Default to 'ETTm1.csv' for minute data
            Other args: Same as BaseTimeSeriesDataset
        """
        super().__init__(args, root_path, flag, size, features, data_path,
                         target, scale, timeenc, freq, seasonal_patterns)

    def _get_borders(self, df_raw):
        """
        Calculate train/val/test split boundaries for minute-level ETT data.

        Multiplies hourly boundaries by 4 (since 4 samples per hour).

        Returns:
            tuple: (border1s, border2s) with indices for each split
        """
        # 4x more samples than hourly (15-minute intervals)
        border1s = [
            0,                                         # Train start
            12 * 30 * 24 * 4 - self.seq_len,          # Val start
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len  # Test start
        ]
        border2s = [
            12 * 30 * 24 * 4,                         # Train end
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,      # Val end
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4       # Test end
        ]
        return border1s, border2s

    def _add_extra_time_features(self, df_stamp):
        """
        Add minute-of-hour feature for minute-level data.

        Extracts the minute component and quantizes to 15-minute intervals (0-3).
        This provides finer temporal granularity than hour alone.

        Args:
            df_stamp (DataFrame): DataFrame with date column and basic time features
        """
        # Convert minute (0-59) to 15-minute interval (0-3)
        # 0-14 min -> 0, 15-29 min -> 1, 30-44 min -> 2, 45-59 min -> 3
        df_stamp['minute'] = df_stamp['date'].dt.minute // 15


# =============================================================================
# Custom Dataset - Percentage-Based Split
# =============================================================================
class Dataset_Custom(BaseTimeSeriesDataset):
    """
    Custom dataset with percentage-based train/val/test split.

    Supports arbitrary time series datasets with flexible splitting strategy.
    Unlike ETT datasets with fixed time-based splits, this uses percentage-based
    splits (70% train, 10% val, 20% test).

    Supported Datasets:
        - Electricity: Electricity consumption data
        - Weather: Weather measurements
        - Traffic: Traffic flow data
        - Exchange Rate: Currency exchange rates

    Split Strategy (Percentage-Based):
        - Train: First 70% of data
        - Val: Next 10% of data
        - Test: Final 20% of data

    Note:
        This split strategy is more flexible but may not preserve temporal
        properties as well as fixed time-based splits.
    """

    def _preprocess_dataframe(self, df_raw):
        """
        Reorder columns to put target variable at the end.

        This ensures consistent column ordering across different datasets:
        [date, feature1, feature2, ..., target]

        Args:
            df_raw (DataFrame): Raw dataframe with arbitrary column order

        Returns:
            DataFrame: Reordered dataframe with target at the end
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)  # Remove target from current position
        cols.remove('date')       # Remove date from current position
        # Reconstruct with date first, target last
        return df_raw[['date'] + cols + [self.target]]

    def _get_borders(self, df_raw):
        """
        Calculate train/val/test split boundaries using percentage-based strategy.

        Splits data into 70% train, 10% val, 20% test based on total length.

        Args:
            df_raw (DataFrame): Raw data

        Returns:
            tuple: (border1s, border2s) with indices for each split
        """
        # Calculate split sizes
        num_train = int(len(df_raw) * 0.7)   # 70% for training
        num_test = int(len(df_raw) * 0.2)    # 20% for testing
        num_vali = len(df_raw) - num_train - num_test  # Remaining for validation

        # Define split boundaries
        border1s = [
            0,                                  # Train start
            num_train - self.seq_len,          # Val start (with overlap)
            len(df_raw) - num_test - self.seq_len  # Test start (with overlap)
        ]
        border2s = [
            num_train,                         # Train end
            num_train + num_vali,              # Val end
            len(df_raw)                        # Test end
        ]
        return border1s, border2s
