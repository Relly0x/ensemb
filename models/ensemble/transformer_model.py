# models/ensemble/transformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


class SimpleTransformer(nn.Module):
    """
    Simple Transformer model for time series forecasting
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model dimensions
        self.d_model = config.get('model', {}).get('hidden_size', 64)
        self.nhead = config.get('ensemble', {}).get('transformer', {}).get('num_heads', 8)
        self.num_layers = config.get('ensemble', {}).get('transformer', {}).get('num_layers', 6)
        self.dropout = config.get('model', {}).get('dropout', 0.1)
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))

        # Ensure d_model is divisible by nhead
        if self.d_model % self.nhead != 0:
            self.d_model = ((self.d_model // self.nhead) + 1) * self.nhead

        # Dynamic initialization
        self.input_dim = None
        self.input_projection = None
        self.pos_encoder = None
        self.transformer = None
        self.output_projection = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=False  # [seq_len, batch_size, d_model]
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.forecast_horizon * self.num_quantiles)
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
            device = past_data.device
            self.input_projection = self.input_projection.to(device)
            self.pos_encoder = self.pos_encoder.to(device)
            self.transformer = self.transformer.to(device)
            self.output_projection = self.output_projection.to(device)

        # Project input to model dimension
        x = self.input_projection(past_data)  # [batch_size, seq_len, d_model]

        # Transpose for transformer: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        transformer_output = self.transformer(x)  # [seq_len, batch_size, d_model]

        # Use the last output for prediction
        last_output = transformer_output[-1]  # [batch_size, d_model]

        # Generate predictions
        output = self.output_projection(last_output)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape to [batch_size, forecast_horizon, num_quantiles]
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)


class TransformerWithDecoder(nn.Module):
    """
    Full Transformer with encoder-decoder architecture
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model dimensions
        self.d_model = config.get('model', {}).get('hidden_size', 64)
        self.nhead = config.get('ensemble', {}).get('transformer', {}).get('num_heads', 8)
        self.num_encoder_layers = config.get('ensemble', {}).get('transformer', {}).get('num_encoder_layers', 6)
        self.num_decoder_layers = config.get('ensemble', {}).get('transformer', {}).get('num_decoder_layers', 6)
        self.dropout = config.get('model', {}).get('dropout', 0.1)
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))

        # Ensure d_model is divisible by nhead
        if self.d_model % self.nhead != 0:
            self.d_model = ((self.d_model // self.nhead) + 1) * self.nhead

        # Dynamic initialization
        self.input_dim = None
        self.future_dim = None
        self.input_projection = None
        self.future_projection = None
        self.pos_encoder = None
        self.transformer = None
        self.output_projection = None
        self._initialized = False

    def _initialize_layers(self, input_dim, future_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim
        self.future_dim = future_dim

        # Input projections
        self.input_projection = nn.Linear(input_dim, self.d_model)
        self.future_projection = nn.Linear(future_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Full transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=False
        )

        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.num_quantiles)

        self._initialized = True

    def forward(self, batch_data):
        """
        Forward pass with encoder-decoder architecture

        Parameters:
        - batch_data: Dictionary with 'past', 'future', 'static' keys

        Returns:
        - Quantile forecasts [batch_size, forecast_horizon, num_quantiles]
        """
        past_data = batch_data['past']  # [batch_size, past_len, features]
        future_data = batch_data['future']  # [batch_size, future_len, features]

        batch_size, past_len, input_dim = past_data.shape
        _, future_len, future_dim = future_data.shape

        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(input_dim, future_dim)
            device = past_data.device
            self.input_projection = self.input_projection.to(device)
            self.future_projection = self.future_projection.to(device)
            self.pos_encoder = self.pos_encoder.to(device)
            self.transformer = self.transformer.to(device)
            self.output_projection = self.output_projection.to(device)

        # Project inputs
        src = self.input_projection(past_data)  # [batch_size, past_len, d_model]
        tgt = self.future_projection(future_data)  # [batch_size, future_len, d_model]

        # Transpose for transformer: [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Create target mask to prevent looking ahead
        tgt_mask = self.transformer.generate_square_subsequent_mask(future_len).to(tgt.device)

        # Apply transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)  # [future_len, batch_size, d_model]

        # Project to quantiles
        output = self.output_projection(output)  # [future_len, batch_size, num_quantiles]

        # Transpose back: [batch_size, future_len, num_quantiles]
        output = output.transpose(0, 1)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # Final linear transformation
        output = self.w_o(context)

        return output


class CustomTransformerBlock(nn.Module):
    """
    Custom transformer block with residual connections and layer normalization
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class LightweightTransformer(nn.Module):
    """
    Lightweight transformer for faster inference
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Smaller dimensions for speed
        self.d_model = config.get('ensemble', {}).get('lightweight_transformer', {}).get('d_model', 32)
        self.num_heads = config.get('ensemble', {}).get('lightweight_transformer', {}).get('num_heads', 4)
        self.num_layers = config.get('ensemble', {}).get('lightweight_transformer', {}).get('num_layers', 3)
        self.d_ff = self.d_model * 2  # Smaller feed-forward dimension
        self.dropout = config.get('model', {}).get('dropout', 0.1)
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))

        # Dynamic initialization
        self.input_dim = None
        self.input_projection = None
        self.transformer_blocks = None
        self.output_projection = None
        self._initialized = False

    def _initialize_layers(self, input_dim):
        """Initialize layers based on input dimensions"""
        if self._initialized:
            return

        self.input_dim = input_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, self.d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.forecast_horizon * self.num_quantiles)
        )

        self._initialized = True

    def forward(self, batch_data):
        """Forward pass"""
        past_data = batch_data['past']  # [batch_size, seq_len, features]
        batch_size, seq_len, input_dim = past_data.shape

        # Initialize layers if needed
        if not self._initialized:
            self._initialize_layers(input_dim)
            device = past_data.device
            self.input_projection = self.input_projection.to(device)
            self.transformer_blocks = self.transformer_blocks.to(device)
            self.output_projection = self.output_projection.to(device)

        # Project input
        x = self.input_projection(past_data)  # [batch_size, seq_len, d_model]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, d_model]

        # Generate output
        output = self.output_projection(x)  # [batch_size, forecast_horizon * num_quantiles]

        # Reshape
        output = output.view(batch_size, self.forecast_horizon, self.num_quantiles)

        return output

    def predict(self, batch_data):
        """Prediction method for ensemble compatibility"""
        with torch.no_grad():
            return self.forward(batch_data)