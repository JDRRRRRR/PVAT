"""
Data Factory Module

This module provides a factory function for creating data loaders for different
time series datasets. It abstracts away dataset-specific configuration and provides
a unified interface for loading train/val/test data.

Supported Datasets:
    - ETTh1, ETTh2: Electricity Transformer Temperature (hourly)
    - ETTm1, ETTm2: Electricity Transformer Temperature (15-minute)
    - electricity: Electricity consumption data
    - exchange_rate: Currency exchange rates
    - traffic: Traffic flow data
    - weather: Weather measurements

The factory pattern allows easy addition of new datasets without modifying
the main training code.

Author: PAViT Team
"""

from dataset.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

# =============================================================================
# Dataset Registry
# =============================================================================
# Maps dataset names to their corresponding Dataset class implementations.
# This registry enables the factory function to instantiate the correct dataset
# class based on the dataset name provided in command-line arguments.
#
# Dataset Classes:
#   - Dataset_ETT_hour: ETT data with hourly granularity (24 samples/day)
#   - Dataset_ETT_minute: ETT data with 15-minute granularity (96 samples/day)
#   - Dataset_Custom: Generic dataset with percentage-based train/val/test split
data_dict = {
    'ETTh1': Dataset_ETT_hour,      # Electricity Transformer Temperature - hourly (station 1)
    'ETTh2': Dataset_ETT_hour,      # Electricity Transformer Temperature - hourly (station 2)
    'ETTm1': Dataset_ETT_minute,    # Electricity Transformer Temperature - 15-minute (station 1)
    'ETTm2': Dataset_ETT_minute,    # Electricity Transformer Temperature - 15-minute (station 2)
    'electricity': Dataset_Custom,   # Electricity consumption data
    'exchange_rate': Dataset_Custom, # Currency exchange rates
    'traffic': Dataset_Custom,       # Traffic flow data
    'weather': Dataset_Custom        # Weather measurements
}


# =============================================================================
# Data Provider Factory Function
# =============================================================================
def data_provider(args, flag):
    """
    Create and return a DataLoader for the specified dataset and split.

    This factory function encapsulates all dataset-specific configuration,
    providing a unified interface for loading data. It handles:
    1. Dataset selection based on args.data
    2. DataLoader creation with appropriate batch size and shuffling
    3. Logging of dataset size

    Args:
        args (Namespace): Configuration object containing:
            - data (str): Dataset name (must be in data_dict)
            - root_path (str): Root directory containing data files
            - seq_len (int): Input sequence length
            - label_len (int): Decoder start token length
            - pred_len (int): Prediction horizon length
            - features (str): Forecasting task type ('M', 'S', or 'MS')
            - target (str): Target variable name
            - batch_size (int): Batch size for DataLoader
        flag (str): Data split type - 'train', 'val', or 'test'

    Returns:
        tuple: (data_loader, enc_in)
            - data_loader (DataLoader): PyTorch DataLoader for the specified split
            - enc_in (int): Number of input variables/channels in the dataset

    Raises:
        KeyError: If args.data is not in data_dict

    Example:
        >>> args = argparse.Namespace(
        ...     data='ETTh1',
        ...     root_path='dataset/ETT-small/',
        ...     seq_len=96,
        ...     label_len=48,
        ...     pred_len=96,
        ...     features='M',
        ...     target='OT',
        ...     batch_size=32
        ... )
        >>> train_loader, enc_in = data_provider(args, flag='train')
        >>> print(f"Number of input variables: {enc_in}")
        >>> for batch in train_loader:
        ...     print(f"Batch shape: {batch[0].shape}")
        ...     break
    """
    # =========================================================================
    # Dataset Selection
    # =========================================================================
    # Retrieve the appropriate Dataset class from the registry
    Data = data_dict[args.data]

    # =========================================================================
    # Time Encoding Configuration
    # =========================================================================
    # Use continuous time encoding (timeenc=1) for all datasets
    # This provides better temporal information than discrete encoding
    # timeenc=0: Discrete features (month, day, weekday, hour, minute)
    # timeenc=1: Continuous features from time_features function
    timeenc = 1

    # =========================================================================
    # DataLoader Configuration
    # =========================================================================
    # Determine whether to shuffle data based on split type
    # - Training: Shuffle to improve generalization
    # - Validation/Test: No shuffle to maintain temporal order
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True

    # Whether to drop the last incomplete batch
    # Set to False to ensure all samples are used
    drop_last = False

    # Batch size for training
    batch_size = args.batch_size

    # Data frequency for time feature extraction
    # 'h' for hourly data (used for all datasets in this implementation)
    # Can be customized per dataset if needed
    freq = 'h'

    # =========================================================================
    # Dataset Instantiation
    # =========================================================================
    # Create dataset instance with all necessary parameters
    data_set = Data(
        args=args,                                    # Configuration object
        root_path=args.root_path,                    # Data directory
        data_path=args.data + '.csv',                # CSV filename
        flag=flag,                                    # Split type (train/val/test)
        size=[args.seq_len, args.label_len, args.pred_len],  # Sequence lengths
        features=args.features,                      # Forecasting task type
        target=args.target,                          # Target variable
        timeenc=timeenc,                             # Time encoding type
        freq=freq,                                    # Data frequency
        seasonal_patterns='Monthly'                  # Seasonal pattern (unused)
    )

    # Log dataset size for debugging and monitoring
    print(f"{flag}: {len(data_set)} samples")

    # =========================================================================
    # DataLoader Creation
    # =========================================================================
    # Wrap dataset in PyTorch DataLoader for efficient batch loading
    data_loader = DataLoader(
        data_set,                    # Dataset to load from
        batch_size=batch_size,       # Number of samples per batch
        shuffle=shuffle_flag,        # Whether to shuffle samples
        drop_last=drop_last          # Whether to drop incomplete last batch
    )

    # =========================================================================
    # Return Results
    # =========================================================================
    # Return both the DataLoader and the number of input variables
    # enc_in is needed to initialize the model with correct input dimension
    return data_loader, data_set.enc_in
