# pipelines/inference/inference_engine.py

import torch
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime


class InferenceEngine:
    """
    Inference engine for TFT model predictions
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = logging.getLogger('inference_engine')

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Track performance metrics
        self.inference_times = []
        self.max_tracked_inferences = 100

        # Feature configuration
        self.past_seq_len = config['model']['past_sequence_length']
        self.forecast_horizon = config['model']['forecast_horizon']

        self.logger.info(f"Inference engine initialized (device: {self.device})")

    def predict(self, data):
        """
        Generate predictions for market data

        Parameters:
        - data: Dictionary of dataframes by instrument

        Returns:
        - Dictionary of predictions by instrument
        """
        start_time = time.time()
        predictions = {}

        try:
            # Process each instrument
            for instrument, instrument_data in data.items():
                # Prepare model input
                model_input = self._prepare_model_input(instrument_data, instrument)

                if model_input is None:
                    continue

                # Move to device
                for key, tensor in model_input.items():
                    model_input[key] = tensor.to(self.device)

                # Run inference
                with torch.no_grad():
                    output = self.model(model_input)

                # Process output
                processed_output = self._process_model_output(output, instrument)

                predictions[instrument] = processed_output

            # Track inference time
            inference_time = time.time() - start_time
            self._track_inference_time(inference_time)

            return predictions

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return {}

    def _prepare_model_input(self, data, instrument):
        """
        Prepare input for model inference

        Parameters:
        - data: Dictionary of dataframes for timeframes
        - instrument: Instrument name

        Returns:
        - Dictionary of model inputs
        """
        try:
            # Use high timeframe data for features
            high_tf = self.config['data']['timeframes']['high']

            if high_tf not in data:
                self.logger.warning(f"Missing high timeframe data for {instrument}")
                return None

            # Get data
            df = data[high_tf].copy()

            if len(df) < self.past_seq_len:
                self.logger.warning(f"Insufficient data for {instrument}: {len(df)} < {self.past_seq_len}")
                return None

            # Create features
            try:
                # Try to use feature creator if available
                from models.feature_engineering.feature_creator import FeatureCreator
                feature_creator = FeatureCreator(self.config)
                features = feature_creator.create_features(df)
            except ImportError:
                # Fallback to basic features
                self.logger.warning("FeatureCreator not available, using basic features")
                features = self._create_basic_features(df)

            # Use most recent data for sequence
            recent_data = features.iloc[-self.past_seq_len:].copy()

            # Create dummy future data
            future_cols = recent_data.shape[1]
            future_data = np.zeros((self.forecast_horizon, future_cols))

            # Create input tensors
            past_tensor = torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0)
            future_tensor = torch.tensor(future_data, dtype=torch.float32).unsqueeze(0)
            static_tensor = torch.zeros((1, 1), dtype=torch.float32)  # Dummy static input

            return {
                'past': past_tensor,
                'future': future_tensor,
                'static': static_tensor
            }

        except Exception as e:
            self.logger.error(f"Error preparing model input: {e}")
            return None

    def _create_basic_features(self, data):
        """
        Create basic features when feature creator is not available

        Parameters:
        - data: Dataframe with OHLC data

        Returns:
        - Dataframe with features
        """
        df = data.copy()

        # Technical indicators
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

        # Price changes
        df['close_change'] = df['close'].pct_change()
        df['high_change'] = df['high'].pct_change()
        df['low_change'] = df['low'].pct_change()

        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()

        # Moving average differences
        df['ma_diff'] = df['sma_10'] - df['sma_20']

        # Fill NAs with 0
        df = df.fillna(0)

        # Remove date index for model input
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)

        return df

    def _process_model_output(self, output, instrument):
        """
        Process model output

        Parameters:
        - output: Model output tensor
        - instrument: Instrument name

        Returns:
        - Processed output
        """
        # Convert to numpy
        output_np = output.cpu().numpy()

        # Get forecast
        forecast = output_np[0]  # Remove batch dimension

        return forecast

    def _track_inference_time(self, inference_time):
        """Track inference time for performance monitoring"""
        self.inference_times.append(inference_time)

        # Trim list if needed
        if len(self.inference_times) > self.max_tracked_inferences:
            self.inference_times = self.inference_times[-self.max_tracked_inferences:]

    def get_performance_metrics(self):
        """Get inference performance metrics"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'max_inference_time': 0,
                'min_inference_time': 0,
                'total_inferences': 0
            }

        return {
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'total_inferences': len(self.inference_times)
        }