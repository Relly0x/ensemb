# models/ensemble/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNTimeSeriesModel(nn.Module):
    """
    CNN model for time series pattern recognition
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model parameters
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # CNN parameters
        cnn_config = config.get('ensemble', {}).get('cnn', {})
        self.num_filters = cnn_config.get('num_filters', [32, 64, 128])
        self.kernel_sizes = cnn_config.get('kernel_sizes', [3, 5, 7])
        self.pool_sizes = cnn_config.get('pool_sizes', [2, 2, 2])

        # Dynamic initialization
        self.input_dim = None
        self.conv_layers = None
        self.adaptive_pool = None
        self.classifier = None
        self._initialized = False

    def _initialize_layers(self, input_dim, seq_len):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Convolutional layers
        conv_layers = []
        in_channels = input_dim
        current_seq_len = seq_len

        for i, (num_filters, kernel_size, pool_size) in enumerate(zip(
                self.num_filters, self.kernel_sizes, self.pool_sizes
        )):
            # Conv1d layer (along time dimension)
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            conv_layers.append(nn.BatchNorm1d(num_filters))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(self.dropout))

            # Max pooling
            conv_layers.append(nn.MaxPool1d(
                kernel_size=pool_size,
                stride=pool_size
            ))

            in_channels = num_filters
            current_seq_len = current_seq_len // pool_size

        self.conv_layers = nn.Sequential(*conv_layers)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        final_features = self.num_filters[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_features, final_features // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_features // 2, self.forecast_horizon * self.num_quantiles)
        )

        self._initialized = True

    def forward(self, batch_data):
        """
        Forward pass

        Parameters:
        - batch_data: Dictionary with 'past', 'future', 'static' keys

        Returns:
        - Quantile forecasts [batch_size, forecast_horizon, num_quantiles]
        """
        # Use past data as main input
        past_data = batch_data['past']  # [batch_size, seq_len, features]
        batch_size, seq_len, input_dim = past_data.shape

        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(input_dim, seq_len)
            device = past_data.device
            self.conv_layers = self.conv_layers.to(device)
            self.adaptive_pool = self.adaptive_pool.to(device)
            self.classifier = self.classifier.to(device)

        # Transpose for Conv1d: [batch_size, features, seq_len]
        x = past_data.transpose(1, 2)

        # Apply convolutional layers
        x = self.conv_layers(x)  # [batch_size, final_filters, reduced_seq_len]

        # Global average pooling
        x = self.adaptive_pool(x)  # [batch_size, final_filters, 1]
        x = x.squeeze(-1)  # [batch_size, final_filters]

        # Classification/regression
        output = self.classifier(x)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape to [batch_size, forecast_horizon, num_quantiles]
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN that captures patterns at different scales
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model parameters
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # Multi-scale parameters
        self.scales = config.get('ensemble', {}).get('multiscale_cnn', {}).get('scales', [1, 2, 4])
        self.num_filters = config.get('ensemble', {}).get('multiscale_cnn', {}).get('num_filters', 64)

        # Dynamic initialization
        self.input_dim = None
        self.scale_convs = None
        self.fusion_layer = None
        self.classifier = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Multi-scale convolutional branches
        scale_convs = nn.ModuleList()

        for scale in self.scales:
            kernel_size = 3 * scale
            padding = kernel_size // 2

            branch = nn.Sequential(
                nn.Conv1d(input_dim, self.num_filters, kernel_size, padding=padding),
                nn.BatchNorm1d(self.num_filters),
                nn.ReLU(),
                nn.Conv1d(self.num_filters, self.num_filters, kernel_size, padding=padding),
                nn.BatchNorm1d(self.num_filters),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )

            scale_convs.append(branch)

        self.scale_convs = scale_convs

        # Fusion layer
        total_features = self.num_filters * len(self.scales)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features // 2, total_features // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(total_features // 4, self.forecast_horizon * self.num_quantiles)
        )

        self._initialized = True

    def forward(self, batch_data):
        """Forward pass with multi-scale processing"""
        past_data = batch_data['past']  # [batch_size, seq_len, features]
        batch_size, seq_len, input_dim = past_data.shape

        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(input_dim)
            device = past_data.device
            self.scale_convs = self.scale_convs.to(device)
            self.fusion_layer = self.fusion_layer.to(device)
            self.classifier = self.classifier.to(device)

        # Transpose for Conv1d: [batch_size, features, seq_len]
        x = past_data.transpose(1, 2)

        # Apply multi-scale convolutions
        scale_outputs = []
        for scale_conv in self.scale_convs:
            scale_out = scale_conv(x)  # [batch_size, num_filters, 1]
            scale_out = scale_out.squeeze(-1)  # [batch_size, num_filters]
            scale_outputs.append(scale_out)

        # Concatenate all scales
        fused = torch.cat(scale_outputs, dim=1)  # [batch_size, total_features]

        # Fusion
        fused = self.fusion_layer(fused)

        # Generate output
        output = self.classifier(fused)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)


class ResidualBlock(nn.Module):
    """Residual block for deep CNN"""

    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out += residual
        out = F.relu(out)

        return out


class DeepResidualCNN(nn.Module):
    """
    Deep residual CNN for complex pattern recognition
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model parameters
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # ResNet parameters
        resnet_config = config.get('ensemble', {}).get('resnet_cnn', {})
        self.num_filters = resnet_config.get('num_filters', 64)
        self.num_blocks = resnet_config.get('num_blocks', 4)
        self.kernel_size = resnet_config.get('kernel_size', 3)

        # Dynamic initialization
        self.input_dim = None
        self.input_conv = None
        self.residual_blocks = None
        self.global_pool = None
        self.classifier = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, self.num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(self.num_filters, self.kernel_size, self.dropout)
            for _ in range(self.num_blocks)
        ])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.num_filters, self.num_filters // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_filters // 2, self.forecast_horizon * self.num_quantiles)
        )

        self._initialized = True

    def forward(self, batch_data):
        """Forward pass with deep residual architecture"""
        past_data = batch_data['past']  # [batch_size, seq_len, features]
        batch_size, seq_len, input_dim = past_data.shape

        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(input_dim)
            device = past_data.device
            self.input_conv = self.input_conv.to(device)
            self.residual_blocks = self.residual_blocks.to(device)
            self.global_pool = self.global_pool.to(device)
            self.classifier = self.classifier.to(device)

        # Transpose for Conv1d: [batch_size, features, seq_len]
        x = past_data.transpose(1, 2)

        # Input convolution
        x = self.input_conv(x)

        # Residual blocks
        x = self.residual_blocks(x)

        # Global pooling
        x = self.global_pool(x)  # [batch_size, num_filters, 1]
        x = x.squeeze(-1)  # [batch_size, num_filters]

        # Classification
        output = self.classifier(x)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)


class WaveNet(nn.Module):
    """
    WaveNet-inspired model for time series
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model parameters
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # WaveNet parameters
        wavenet_config = config.get('ensemble', {}).get('wavenet', {})
        self.residual_channels = wavenet_config.get('residual_channels', 32)
        self.skip_channels = wavenet_config.get('skip_channels', 256)
        self.dilation_cycles = wavenet_config.get('dilation_cycles', 3)
        self.dilation_depth = wavenet_config.get('dilation_depth', 8)

        # Dynamic initialization
        self.input_dim = None
        self.input_conv = None
        self.dilated_convs = None
        self.skip_conv = None
        self.output_conv = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Input convolution
        self.input_conv = nn.Conv1d(input_dim, self.residual_channels, kernel_size=1)

        # Dilated convolutions
        dilated_convs = nn.ModuleList()
        skip_convs = nn.ModuleList()

        for cycle in range(self.dilation_cycles):
            for depth in range(self.dilation_depth):
                dilation = 2 ** depth

                # Dilated convolution
                conv = nn.Conv1d(
                    self.residual_channels,
                    2 * self.residual_channels,  # For gating
                    kernel_size=2,
                    dilation=dilation,
                    padding=dilation
                )
                dilated_convs.append(conv)

                # Skip connection
                skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
                skip_convs.append(skip_conv)

        self.dilated_convs = dilated_convs
        self.skip_convs = skip_convs

        # Output layers
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.skip_channels, self.forecast_horizon * self.num_quantiles, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

        self._initialized = True

    def forward(self, batch_data):
        """Forward pass with dilated convolutions"""
        past_data = batch_data['past']  # [batch_size, seq_len, features]
        batch_size, seq_len, input_dim = past_data.shape

        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(input_dim)
            device = past_data.device
            self.input_conv = self.input_conv.to(device)
            self.dilated_convs = self.dilated_convs.to(device)
            self.skip_convs = self.skip_convs.to(device)
            self.output_conv = self.output_conv.to(device)

        # Transpose for Conv1d: [batch_size, features, seq_len]
        x = past_data.transpose(1, 2)

        # Input convolution
        x = self.input_conv(x)  # [batch_size, residual_channels, seq_len]

        # Accumulate skip connections
        skip_outputs = []

        # Apply dilated convolutions
        for dilated_conv, skip_conv in zip(self.dilated_convs, self.skip_convs):
            # Dilated convolution with gating
            conv_out = dilated_conv(x)

            # Split for gating
            filter_out, gate_out = torch.chunk(conv_out, 2, dim=1)
            gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)

            # Residual connection
            residual = nn.Conv1d(self.residual_channels, self.residual_channels, kernel_size=1)(gated)
            x = x + residual.to(x.device)

            # Skip connection
            skip_out = skip_conv(gated)
            skip_outputs.append(skip_out)

        # Sum all skip connections
        skip_sum = torch.stack(skip_outputs, dim=0).sum(dim=0)

        # Output convolution
        output = self.output_conv(skip_sum)  # [batch_size, forecast_horizon * num_quantiles, 1]
        output = output.squeeze(-1)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)