import torch
import torch.nn as nn
from .layers import VariableSelectionNetwork, GatedResidualNetwork
from .attention import TemporalSelfAttention


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Get dimensions from config with sensible defaults
        self.hidden_size = config.get('hidden_size', 64)
        self.past_seq_len = config.get('past_sequence_length', 120)
        self.forecast_horizon = config.get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('quantiles', [0.1, 0.5, 0.9]))

        # Auto-detect input dimensions or use config
        # We'll determine actual dimensions from the first batch
        self.past_input_dim = None
        self.future_input_dim = None
        self.static_input_dim = None

        # These will be initialized when we see the first batch
        self.static_encoder = None
        self.past_encoder = None
        self.future_encoder = None

        # Core model components (initialize with placeholders)
        self.lstm_encoder = None
        self.lstm_decoder = None
        self.attention = None
        self.output_projection = None

        self._initialized = False

    def _initialize_layers(self, past_shape, future_shape, static_shape):
        """Initialize layers based on actual input shapes"""
        if self._initialized:
            return

        self.past_input_dim = past_shape[-1]
        self.future_input_dim = future_shape[-1] if len(future_shape) > 2 else 1
        self.static_input_dim = static_shape[-1] if len(static_shape) > 1 else 1

        # Input processing networks
        self.static_encoder = nn.Linear(self.static_input_dim, self.hidden_size)

        self.past_encoder = nn.Sequential(
            nn.Linear(self.past_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.1))
        )

        self.future_encoder = nn.Sequential(
            nn.Linear(self.future_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.1))
        )

        # LSTM layers for temporal processing
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.1) if self.config.get('lstm_layers', 2) > 1 else 0,
            batch_first=True
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.1) if self.config.get('lstm_layers', 2) > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = TemporalSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.config.get('attention_heads', 4),
            dropout=self.config.get('dropout', 0.1)
        )

        # Output processing
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.1)),
            nn.Linear(self.hidden_size, self.num_quantiles)
        )

        self._initialized = True

    def forward(self, batch_data):
        """
        Forward pass of the TFT model

        Parameters:
        - batch_data: Dictionary containing 'past', 'future', 'static', 'target'

        Returns:
        - Quantile forecasts tensor [batch_size, forecast_horizon, num_quantiles]
        """
        # Extract inputs from batch
        past_inputs = batch_data['past']  # [batch_size, past_seq_len, past_features]
        future_inputs = batch_data['future']  # [batch_size, forecast_horizon, future_features]
        static_inputs = batch_data['static']  # [batch_size, static_features]

        batch_size = past_inputs.size(0)

        # Initialize layers if not done yet
        if not self._initialized:
            self._initialize_layers(past_inputs.shape, future_inputs.shape, static_inputs.shape)
            # Move to same device as inputs
            device = past_inputs.device
            self.static_encoder = self.static_encoder.to(device)
            self.past_encoder = self.past_encoder.to(device)
            self.future_encoder = self.future_encoder.to(device)
            self.lstm_encoder = self.lstm_encoder.to(device)
            self.lstm_decoder = self.lstm_decoder.to(device)
            self.attention = self.attention.to(device)
            self.output_projection = self.output_projection.to(device)

        # Encode static features
        static_encoded = self.static_encoder(static_inputs)  # [batch_size, hidden_size]

        # Encode past sequences
        past_encoded = self.past_encoder(past_inputs)  # [batch_size, past_seq_len, hidden_size]

        # Add static context to past sequence
        static_expanded = static_encoded.unsqueeze(1).expand(-1, past_encoded.size(1), -1)
        past_enhanced = past_encoded + static_expanded

        # Encode future sequences
        future_encoded = self.future_encoder(future_inputs)  # [batch_size, forecast_horizon, hidden_size]

        # Add static context to future sequence
        static_expanded_future = static_encoded.unsqueeze(1).expand(-1, future_encoded.size(1), -1)
        future_enhanced = future_encoded + static_expanded_future

        # LSTM encoder (process past sequence)
        past_lstm_out, (hidden, cell) = self.lstm_encoder(past_enhanced)

        # LSTM decoder (process future sequence with past context)
        future_lstm_out, _ = self.lstm_decoder(future_enhanced, (hidden, cell))

        # Combine past and future for attention
        # Only use the future part for final predictions
        combined_sequence = torch.cat([past_lstm_out, future_lstm_out], dim=1)

        # Apply self-attention
        attended_features = self.attention(combined_sequence)

        # Extract only the future part for predictions
        future_attended = attended_features[:, -self.forecast_horizon:, :]

        # Generate quantile forecasts
        quantile_forecasts = self.output_projection(future_attended)

        return quantile_forecasts


# Fixed SimpleTFT model that initializes all parameters immediately
class SimpleTFT(nn.Module):
    """
    Simplified TFT for testing purposes - Fixed to have proper parameter initialization
    This version initializes all layers immediately with reasonable defaults
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 64)
        self.forecast_horizon = config.get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('quantiles', [0.1, 0.5, 0.9]))
        self.lstm_layers = config.get('lstm_layers', 2)
        self.dropout = config.get('dropout', 0.1)

        # Expected input size from your data (based on the normalizer output)
        # This will be adjusted dynamically if needed
        self.expected_input_size = 27  # From your data processing

        # Initialize all layers immediately with expected dimensions
        self.feature_projection = nn.Linear(self.expected_input_size, self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_quantiles)
        )

        # Track if we need to adjust input size
        self._input_size_adjusted = False

    def _adjust_input_layer(self, actual_input_size):
        """Adjust the input layer if the actual input size differs from expected"""
        if actual_input_size != self.expected_input_size and not self._input_size_adjusted:
            print(f"Adjusting input layer from {self.expected_input_size} to {actual_input_size}")

            # Create new feature projection layer with correct input size
            old_weight = self.feature_projection.weight.data
            old_bias = self.feature_projection.bias.data
            device = old_weight.device

            # Create new layer
            self.feature_projection = nn.Linear(actual_input_size, self.hidden_size).to(device)

            # If the new input size is larger, we can preserve some weights
            if actual_input_size >= self.expected_input_size:
                with torch.no_grad():
                    self.feature_projection.weight.data[:, :self.expected_input_size] = old_weight
                    self.feature_projection.bias.data = old_bias
            # If smaller, we truncate the weights
            elif actual_input_size < self.expected_input_size:
                with torch.no_grad():
                    self.feature_projection.weight.data = old_weight[:, :actual_input_size]
                    self.feature_projection.bias.data = old_bias

            self.expected_input_size = actual_input_size
            self._input_size_adjusted = True

    def forward(self, batch_data):
        """Forward pass"""
        # Use past data for predictions
        past_inputs = batch_data['past']
        batch_size, seq_len, input_size = past_inputs.shape

        # Adjust input layer if necessary (only on first call)
        if input_size != self.expected_input_size:
            self._adjust_input_layer(input_size)

        # Project features to hidden dimension
        features = self.feature_projection(past_inputs)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)

        # Use the last output for generating forecasts
        # Take the last hidden state
        last_hidden = lstm_out[:, -1:, :]  # [batch, 1, hidden]

        # Expand to forecast horizon
        forecast_input = last_hidden.repeat(1, self.forecast_horizon, 1)

        # Generate quantile predictions
        predictions = self.output_layer(forecast_input)

        return predictions

    def get_model_info(self):
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'forecast_horizon': self.forecast_horizon,
            'num_quantiles': self.num_quantiles,
            'expected_input_size': self.expected_input_size
        }