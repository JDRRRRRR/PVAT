import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

class Model(nn.Module):
    """
    DLinear: Decomposition Linear Model for Time Series Forecasting.

    A simple baseline that decomposes time series into trend and seasonal
    components, then applies linear transformations to each component.

    Architecture:
        1. Series Decomposition: Split input into trend and seasonal
        2. Linear Projection: Apply linear layer to each component
        3. Reconstruction: Add trend and seasonal predictions

    The model can use either individual linear layers per variable or
    shared linear layers across all variables.

    Args:
        task_name (str): Task type (default: 'long_term_forecast')
        seq_len (int): Input sequence length
        pred_len (int): Prediction horizon length
        moving_avg (int): Window size for moving average decomposition
        individual (bool): Whether to use individual linear layers per variable
        enc_in (int): Number of input channels (variables)
        num_class (int): Number of classes for classification task
    """

    def __init__(self, task_name='long_term_forecast',
                 seq_len=96,
                 pred_len=96,
                 moving_avg=25,
                 individual=False,
                 enc_in=7,
                 num_class=10):
        """
        Initialize DLinear model.

        Args:
            individual (bool): If True, use separate linear layers for each variable.
                              If False, use shared linear layers across all variables.
        """
        super(Model, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.task_name = task_name
        self.seq_len = seq_len

        # For non-forecasting tasks, output length equals input length
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = seq_len
        else:
            self.pred_len = pred_len

        # =====================================================================
        # Series Decomposition
        # =====================================================================
        # Decompose input into trend and seasonal components
        # Uses moving average with specified kernel size
        self.decompsition = series_decomp(moving_avg)

        # =====================================================================
        # Model Configuration
        # =====================================================================
        self.individual = individual
        self.channels = enc_in

        # =====================================================================
        # Linear Layers
        # =====================================================================
        if self.individual:
            # Individual linear layers for each variable
            # Each variable has its own trend and seasonal projections
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                # Linear layer: seq_len -> pred_len
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                # Initialize weights to uniform average (1/seq_len)
                # This provides a reasonable initialization for forecasting
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            # Shared linear layers across all variables
            # All variables use the same trend and seasonal projections
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Initialize weights to uniform average
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # =====================================================================
        # Task-specific Projections
        # =====================================================================
        if self.task_name == 'classification':
            # Classification head: flatten and project to class logits
            self.projection = nn.Linear(enc_in * seq_len, num_class)

    # =========================================================================
    # Core Encoder
    # =========================================================================
    def encoder(self, x):
        """
        Encode input using decomposition and linear projections.

        Algorithm:
            1. Decompose input into seasonal and trend components
            2. Apply linear projection to each component
            3. Combine projections

        Args:
            x (Tensor): Input time series [batch, seq_len, n_vars]

        Returns:
            Tensor: Predictions [batch, pred_len, n_vars]
        """
        # =====================================================================
        # Series Decomposition
        # =====================================================================
        # Decompose into seasonal and trend components
        # seasonal_init: [batch, seq_len, n_vars]
        # trend_init: [batch, seq_len, n_vars]
        seasonal_init, trend_init = self.decompsition(x)

        # Permute to [batch, n_vars, seq_len] for linear layer processing
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # =====================================================================
        # Linear Projections
        # =====================================================================
        if self.individual:
            # Process each variable separately with its own linear layer
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype
            ).to(seasonal_init.device)

            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype
            ).to(trend_init.device)

            # Apply individual linear layers to each variable
            for i in range(self.channels):
                # seasonal_init[:, i, :]: [batch, seq_len]
                # seasonal_output[:, i, :]: [batch, pred_len]
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            # Apply shared linear layers to all variables
            # seasonal_init: [batch, n_vars, seq_len]
            # seasonal_output: [batch, n_vars, pred_len]
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # =====================================================================
        # Reconstruction
        # =====================================================================
        # Combine trend and seasonal predictions
        x = seasonal_output + trend_output

        # Permute back to [batch, pred_len, n_vars]
        return x.permute(0, 2, 1)

    # =========================================================================
    # Task-specific Methods
    # =========================================================================
    def forecast(self, x_enc):
        """
        Forecasting task: predict future values.

        Args:
            x_enc (Tensor): Encoder input [batch, seq_len, n_vars]

        Returns:
            Tensor: Predictions [batch, pred_len, n_vars]
        """
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        """
        Imputation task: fill missing values.

        Args:
            x_enc (Tensor): Input with missing values [batch, seq_len, n_vars]

        Returns:
            Tensor: Imputed values [batch, seq_len, n_vars]
        """
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task: identify anomalies.

        Args:
            x_enc (Tensor): Input time series [batch, seq_len, n_vars]

        Returns:
            Tensor: Anomaly scores [batch, seq_len, n_vars]
        """
        return self.encoder(x_enc)

    def classification(self, x_enc):
        """
        Classification task: classify time series.

        Args:
            x_enc (Tensor): Input time series [batch, seq_len, n_vars]

        Returns:
            Tensor: Class logits [batch, num_class]
        """
        # Encode input
        enc_out = self.encoder(x_enc)

        # Flatten: [batch, seq_len, n_vars] -> [batch, seq_len * n_vars]
        output = enc_out.reshape(enc_out.shape[0], -1)

        # Project to class logits: [batch, num_class]
        output = self.projection(output)

        return output

    # =========================================================================
    # Main Forward Method
    # =========================================================================
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass: route to appropriate task-specific method.

        Args:
            x_enc (Tensor): Encoder input
            x_mark_enc (Tensor): Encoder time features (unused)
            x_dec (Tensor): Decoder input (unused)
            x_mark_dec (Tensor): Decoder time features (unused)
            mask (Tensor, optional): Mask (unused)

        Returns:
            Tensor: Task-specific output
        """
        # Route to appropriate task handler
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Forecasting: return only prediction horizon
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]

        if self.task_name == 'imputation':
            # Imputation: return full sequence
            dec_out = self.imputation(x_enc)
            return dec_out

        if self.task_name == 'anomaly_detection':
            # Anomaly detection: return anomaly scores
            dec_out = self.anomaly_detection(x_enc)
            return dec_out

        if self.task_name == 'classification':
            # Classification: return class logits
            dec_out = self.classification(x_enc)
            return dec_out

        # Unknown task
        return None
