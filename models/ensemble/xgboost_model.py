# models/ensemble/xgboost_model.py

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class XGBoostEnsemble:
    """XGBoost model wrapper for ensemble learning"""

    def __init__(self, config):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required but not installed")

        self.config = config
        self.logger = logging.getLogger('xgboost_ensemble')

        # Model parameters
        self.forecast_horizon = config.get('model', {}).get('forecast_horizon', 12)
        self.num_quantiles = len(config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9]))
        self.quantiles = config.get('model', {}).get('quantiles', [0.1, 0.5, 0.9])

        # XGBoost parameters
        xgb_config = config.get('ensemble', {}).get('xgboost', {})
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': xgb_config.get('n_estimators', 100),
            'max_depth': xgb_config.get('max_depth', 6),
            'learning_rate': xgb_config.get('learning_rate', 0.1),
            'subsample': xgb_config.get('subsample', 0.8),
            'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
            'random_state': xgb_config.get('random_state', 42),
            'n_jobs': xgb_config.get('n_jobs', -1)
        }

        # Models for each forecast step and quantile
        self.models = {}  # (step, quantile) -> model
        self.feature_names = None
        self.is_trained = False

        # Feature engineering parameters
        self.lookback_window = config.get('model', {}).get('past_sequence_length', 120)

        self.logger.info("XGBoost ensemble model initialized")

    def _extract_features(self, batch_data):
        """Extract features from the batch data for XGBoost"""
        past_data = batch_data['past']  # [batch_size, seq_len, features]

        # Convert to numpy if tensor
        if isinstance(past_data, torch.Tensor):
            past_data = past_data.detach().cpu().numpy()

        batch_size = past_data.shape[0]
        features_list = []

        for i in range(batch_size):
            sample_features = self._extract_sample_features(past_data[i])
            features_list.append(sample_features)

        return np.array(features_list)

    def _extract_sample_features(self, past_sequence):
        """Extract features for a single sample"""
        features = []

        if len(past_sequence.shape) == 2:
            seq_len, num_features = past_sequence.shape

            # Latest values
            features.extend(past_sequence[-1, :].flatten())

            # Lag features
            lag_steps = [1, 2, 3, 5, 10]
            for lag in lag_steps:
                if lag < seq_len:
                    features.extend(past_sequence[-lag - 1, :].flatten())
                else:
                    features.extend(np.zeros(num_features))

            # Rolling statistics
            for window in [5, 10, 20]:
                if window <= seq_len:
                    window_data = past_sequence[-window:, :]
                    features.extend(np.mean(window_data, axis=0))
                    features.extend(np.std(window_data, axis=0))
                else:
                    features.extend(np.zeros(num_features * 2))

        return np.array(features)

    def fit(self, X, y):
        """Train the XGBoost models"""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Train separate models for each forecast step and quantile
        for step in range(self.forecast_horizon):
            for q_idx, quantile in enumerate(self.quantiles):
                self.logger.info(f"Training model for step {step + 1}, quantile {quantile}")

                # Get target for this step
                y_step = y[:, step] if len(y.shape) > 1 else y

                # Create quantile-specific model
                if quantile == 0.5:  # Median - use standard regression
                    model = xgb.XGBRegressor(**self.xgb_params)
                else:  # Use quantile regression
                    params = self.xgb_params.copy()
                    params['objective'] = f'reg:quantileerror'
                    params['quantile_alpha'] = quantile
                    model = xgb.XGBRegressor(**params)

                # Train model
                model.fit(X, y_step)

                # Store model
                self.models[(step, q_idx)] = model

        self.is_trained = True
        self.logger.info("XGBoost ensemble training completed")

    def predict(self, batch_data):
        """Generate predictions"""
        if not self.is_trained:
            # Return dummy predictions if not trained
            batch_size = batch_data['past'].shape[0]
            return np.random.randn(batch_size, self.forecast_horizon, self.num_quantiles) * 0.01

        # Extract features
        X = self._extract_features(batch_data)
        batch_size = X.shape[0]

        # Initialize predictions array
        predictions = np.zeros((batch_size, self.forecast_horizon, self.num_quantiles))

        # Generate predictions for each step and quantile
        for step in range(self.forecast_horizon):
            for q_idx in range(self.num_quantiles):
                if (step, q_idx) in self.models:
                    model = self.models[(step, q_idx)]
                    pred = model.predict(X)
                    predictions[:, step, q_idx] = pred

        return predictions

    def __call__(self, batch_data):
        """Make the model callable like PyTorch models"""
        return self.predict(batch_data)