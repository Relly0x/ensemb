# pipelines/inference/ensemble_inference_engine.py

import torch
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from models.ensemble.ensemble_manager import EnsembleManager


class EnsembleInferenceEngine:
    """
    Inference engine for ensemble model predictions
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('ensemble_inference_engine')

        # Initialize ensemble manager
        self.ensemble_manager = EnsembleManager(config)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Track performance metrics
        self.inference_times = []
        self.max_tracked_inferences = 100

        # Feature configuration
        self.past_seq_len = config['model']['past_sequence_length']
        self.forecast_horizon = config['model']['forecast_horizon']

        # Store prediction history for ensemble confidence calculation
        self.ensemble_predictions = []

        self.logger.info(f"Ensemble inference engine initialized (device: {self.device})")

    def predict(self, data):
        """
        Generate ensemble predictions for market data

        Parameters:
        - data: Dictionary of dataframes by instrument

        Returns:
        - Dictionary of predictions by instrument with ensemble metadata
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

                # Move to device for neural network models
                device_input = {}
                for key, tensor in model_input.items():
                    if isinstance(tensor, torch.Tensor):
                        device_input[key] = tensor.to(self.device)
                    else:
                        device_input[key] = tensor

                # Get ensemble prediction with metadata
                ensemble_result = self.ensemble_manager.predict(device_input)

                # Process output with confidence metrics
                processed_output = self._process_ensemble_output(ensemble_result, instrument)

                predictions[instrument] = processed_output

            # Store prediction metadata for strategy
            self._store_prediction_metadata(predictions)

            # Track inference time
            inference_time = time.time() - start_time
            self._track_inference_time(inference_time)

            return predictions

        except Exception as e:
            self.logger.error(f"Error during ensemble inference: {e}")
            return {}

    def _prepare_model_input(self, data, instrument):
        """Prepare input for ensemble inference"""
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
                from models.feature_engineering.feature_creator import FeatureCreator
                feature_creator = FeatureCreator(self.config)
                features = feature_creator.create_features(df)
            except ImportError:
                self.logger.warning("FeatureCreator not available, using basic features")
                features = self._create_basic_features(df)

            # Use most recent data for sequence
            recent_data = features.iloc[-self.past_seq_len:].copy()

            # Create input tensors
            past_tensor = torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0)
            future_tensor = torch.zeros((1, self.forecast_horizon, recent_data.shape[1]), dtype=torch.float32)
            static_tensor = torch.zeros((1, 1), dtype=torch.float32)

            return {
                'past': past_tensor,
                'future': future_tensor,
                'static': static_tensor
            }

        except Exception as e:
            self.logger.error(f"Error preparing model input: {e}")
            return None

    def _create_basic_features(self, data):
        """Create basic features when feature creator is not available"""
        df = data.copy()

        # Basic technical indicators
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['close_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(window=20).std()

        # Fill NAs and remove date index
        df = df.fillna(0)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)

        return df

    def _process_ensemble_output(self, ensemble_result, instrument):
        """Process ensemble output with confidence metrics"""
        # ensemble_result should be numpy array [batch_size, forecast_horizon, num_quantiles]

        # Calculate ensemble confidence based on model agreement
        model_info = self.ensemble_manager.get_model_info()

        # Simple confidence calculation (can be enhanced)
        confidence = min(1.0, model_info['total_models'] / 5.0)  # Higher confidence with more models

        # Return prediction with metadata
        return {
            'prediction': ensemble_result,
            'metadata': {
                'confidence': confidence,
                'num_models': model_info['total_models'],
                'model_weights': model_info['model_weights'],
                'combination_method': model_info['combination_method']
            }
        }

    def _store_prediction_metadata(self, predictions):
        """Store prediction metadata for strategy access"""
        prediction_info = {
            'timestamp': datetime.now(),
            'metadata': {instrument: pred.get('metadata', {}) for instrument, pred in predictions.items()}
        }

        self.ensemble_predictions.append(prediction_info)

        # Limit history size
        if len(self.ensemble_predictions) > 10:
            self.ensemble_predictions = self.ensemble_predictions[-10:]

    def _track_inference_time(self, inference_time):
        """Track inference time for performance monitoring"""
        self.inference_times.append(inference_time)

        if len(self.inference_times) > self.max_tracked_inferences:
            self.inference_times = self.inference_times[-self.max_tracked_inferences:]

    def get_ensemble_performance_metrics(self):
        """Get ensemble-specific performance metrics"""
        base_metrics = {
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'total_inferences': len(self.inference_times)
        }

        # Add ensemble-specific metrics
        model_info = self.ensemble_manager.get_model_info()
        base_metrics.update(model_info)

        return base_metrics

    def load_ensemble(self, ensemble_path):
        """Load saved ensemble"""
        return self.ensemble_manager.load_ensemble(ensemble_path)

    def save_ensemble(self, ensemble_path):
        """Save ensemble"""
        return self.ensemble_manager.save_ensemble(ensemble_path)