# models/ensemble/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionMechanism(nn.Module):
    """Attention mechanism for LSTM"""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]

        # Calculate attention weights
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize along sequence

        # Apply attention weights
        context_vector = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_size]

        return context_vector, attention_weights


class AttentionLSTM(nn.Module):
    """
    LSTM model with attention mechanism for time series forecasting
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('model', {}).get('hidden_size', 64)
        self.num_layers = config.get('model', {}).get('lstm_layers', 2)
        self.dropout = config.get('model', {}).get('dropout', 0.1)
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))

        # Expected input dimensions (will be set dynamically)
        self.input_dim = None

        # Layers (will be initialized when we see first input)
        self.input_projection = None
        self.lstm = None
        self.attention = None
        self.output_projection = None

        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, self.hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention mechanism
        self.attention = AttentionMechanism(self.hidden_size)

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.forecast_horizon * self.num_quantiles)
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
            self._initialize_layers(input_dim)
            # Move to same device as input
            device = past_data.device
            self.input_projection = self.input_projection.to(device)
            self.lstm = self.lstm.to(device)
            self.attention = self.attention.to(device)
            self.output_projection = self.output_projection.to(device)

        # Project input to hidden dimension
        projected_input = self.input_projection(past_data)  # [batch_size, seq_len, hidden_size]

        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(projected_input)  # [batch_size, seq_len, hidden_size]

        # Apply attention
        context_vector, attention_weights = self.attention(lstm_output)  # [batch_size, hidden_size]

        # Generate output
        output = self.output_projection(context_vector)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape to [batch_size, forecast_horizon, num_quantiles]
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)


class LSTMFeatureExtractor(nn.Module):
    """
    Feature extractor using LSTM for ensemble learning
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('model', {}).get('hidden_size', 64)
        self.num_layers = config.get('model', {}).get('lstm_layers', 2)
        self.dropout = config.get('model', {}).get('dropout', 0.1)

        # Will be initialized dynamically
        self.input_dim = None
        self.input_projection = None
        self.lstm = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        self.input_projection = nn.Linear(input_dim, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better feature extraction
        )

        self._initialized = True

    def forward(self, x):
        """Extract features from input sequence"""
        batch_size, seq_len, input_dim = x.shape

        # Initialize if needed
        if not self._initialized:
            self._initialize_layers(input_dim)
            device = x.device
            self.input_projection = self.input_projection.to(device)
            self.lstm = self.lstm.to(device)

        # Project input
        projected = self.input_projection(x)

        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(projected)

        # For bidirectional LSTM, concatenate the final hidden states
        # hidden: [num_layers * 2, batch_size, hidden_size]
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_size * 2]

        return final_hidden, lstm_output


class MultiModalLSTM(nn.Module):
    """
    Multi-modal LSTM that can handle different input types
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('model', {}).get('hidden_size', 64)
        self.num_layers = config.get('model', {}).get('lstm_layers', 2)
        self.dropout = config.get('model', {}).get('dropout', 0.1)
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))

        # Feature extractors for different input types
        self.past_extractor = LSTMFeatureExtractor(config)
        self.future_extractor = LSTMFeatureExtractor(config)

        # Static feature processing
        self.static_processor = nn.Sequential(
            nn.Linear(1, self.hidden_size // 4),  # Assuming small static input
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Fusion layer
        fusion_input_size = self.hidden_size * 2 * 2 + self.hidden_size // 4  # past + future + static
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.forecast_horizon * self.num_quantiles)
        )

    def forward(self, batch_data):
        """Forward pass with multi-modal inputs"""
        past_data = batch_data['past']
        future_data = batch_data['future']
        static_data = batch_data['static']

        batch_size = past_data.shape[0]

        # Extract features from different modalities
        past_features, _ = self.past_extractor(past_data)
        future_features, _ = self.future_extractor(future_data)
        static_features = self.static_processor(static_data)

        # Concatenate all features
        combined_features = torch.cat([past_features, future_features, static_features], dim=1)

        # Fusion
        fused_features = self.fusion(combined_features)

        # Generate output
        output = self.output_layer(fused_features)

        # Reshape to [batch_size, forecast_horizon, num_quantiles]
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)