# models/ensemble/ensemble_manager.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import threading
import time


class EnsembleManager:
    """
    Advanced ensemble manager that combines multiple AI models for trading predictions
    """

    def __init__(self, config):
        self.config = config
        self.ensemble_config = config.get('ensemble', {})
        self.logger = logging.getLogger('ensemble_manager')

        # Model storage
        self.models = {}  # model_name -> model_instance
        self.model_weights = {}  # model_name -> weight
        self.model_performance = {}  # model_name -> performance_metrics

        # Ensemble settings
        self.combination_method = self.ensemble_config.get('combination_method', 'weighted_average')
        self.min_models_required = self.ensemble_config.get('min_models_required', 2)
        self.enable_dynamic_weighting = self.ensemble_config.get('enable_dynamic_weighting', True)
        self.performance_window = self.ensemble_config.get('performance_window', 100)

        # Performance tracking
        self.prediction_history = []
        self.performance_lock = threading.Lock()

        # Initialize models
        self._initialize_models()

        self.logger.info(f"Ensemble manager initialized with {len(self.models)} models")

    def _initialize_models(self):
        """Initialize all models in the ensemble"""
        enabled_models = self.ensemble_config.get('enabled_models', ['tft', 'lstm', 'xgboost'])

        for model_name in enabled_models:
            try:
                if model_name == 'tft':
                    self._load_tft_model()
                elif model_name == 'lstm':
                    self._load_lstm_model()
                elif model_name == 'xgboost':
                    self._load_xgboost_model()
                elif model_name == 'transformer':
                    self._load_transformer_model()
                elif model_name == 'cnn':
                    self._load_cnn_model()
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")

            except Exception as e:
                self.logger.error(f"Failed to load {model_name} model: {e}")

        # Initialize equal weights
        self._initialize_weights()

    def _load_tft_model(self):
        """Load the existing TFT model"""
        from models.tft.model import TemporalFusionTransformer, SimpleTFT

        model_config = self.config['model'].copy()

        # Try to load trained model if available
        model_path = self.config.get('export', {}).get('model_path')

        if model_path and os.path.exists(model_path):
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Create model
            try:
                model = TemporalFusionTransformer(model_config)
            except:
                model = SimpleTFT(model_config)

            # Load weights
            if 'model_state_dict' in checkpoint:
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
        else:
            # Create new model
            try:
                model = TemporalFusionTransformer(model_config)
            except:
                model = SimpleTFT(model_config)

        model.eval()
        self.models['tft'] = model
        self.logger.info("TFT model loaded successfully")

    def _load_lstm_model(self):
        """Load LSTM with attention model"""
        from models.ensemble.lstm_model import AttentionLSTM

        model = AttentionLSTM(self.config)
        self.models['lstm'] = model
        self.logger.info("LSTM model created successfully")

    def _load_xgboost_model(self):
        """Load XGBoost model"""
        from models.ensemble.xgboost_model import XGBoostEnsemble

        model = XGBoostEnsemble(self.config)
        self.models['xgboost'] = model
        self.logger.info("XGBoost model created successfully")

    def _load_transformer_model(self):
        """Load simple Transformer model"""
        from models.ensemble.transformer_model import SimpleTransformer

        model = SimpleTransformer(self.config)
        self.models['transformer'] = model
        self.logger.info("Transformer model created successfully")

    def _load_cnn_model(self):
        """Load CNN model for pattern recognition"""
        from models.ensemble.cnn_model import CNNTimeSeriesModel

        model = CNNTimeSeriesModel(self.config)
        self.models['cnn'] = model
        self.logger.info("CNN model created successfully")

    def _initialize_weights(self):
        """Initialize model weights"""
        if not self.models:
            return

        # Start with equal weights
        num_models = len(self.models)
        initial_weight = 1.0 / num_models

        for model_name in self.models.keys():
            self.model_weights[model_name] = initial_weight
            self.model_performance[model_name] = {
                'accuracy': 0.5,  # Start neutral
                'predictions': [],
                'errors': []
            }

    def predict(self, batch_data):
        """
        Generate ensemble predictions from all models

        Parameters:
        - batch_data: Input data in the expected format

        Returns:
        - Combined prediction tensor
        """
        if len(self.models) < self.min_models_required:
            self.logger.warning(
                f"Only {len(self.models)} models available, minimum required: {self.min_models_required}")
            # Fall back to available models or raise error
            if len(self.models) == 0:
                raise ValueError("No models available for prediction")

        # Get predictions from each model
        predictions = {}
        prediction_confidences = {}

        for model_name, model in self.models.items():
            try:
                # Get prediction from model
                if hasattr(model, 'predict'):
                    pred = model.predict(batch_data)
                else:
                    pred = model(batch_data)

                # Convert to numpy if tensor
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()

                predictions[model_name] = pred

                # Calculate confidence (simple method - can be enhanced)
                confidence = self._calculate_prediction_confidence(pred, model_name)
                prediction_confidences[model_name] = confidence

                self.logger.debug(f"{model_name} prediction shape: {pred.shape}, confidence: {confidence:.3f}")

            except Exception as e:
                self.logger.error(f"Error getting prediction from {model_name}: {e}")
                continue

        if not predictions:
            raise ValueError("No successful predictions from any model")

        # Combine predictions
        combined_prediction = self._combine_predictions(predictions, prediction_confidences)

        # Store for performance tracking
        self._store_prediction_info(predictions, combined_prediction)

        return combined_prediction

    def _calculate_prediction_confidence(self, prediction, model_name):
        """Calculate confidence score for a prediction"""
        try:
            # For quantile predictions, use the spread as confidence indicator
            if len(prediction.shape) >= 3 and prediction.shape[-1] >= 3:
                # Assuming [batch, sequence, quantiles] format
                q1 = prediction[:, :, 0]  # 0.1 quantile
                q3 = prediction[:, :, 2]  # 0.9 quantile
                median = prediction[:, :, 1]  # 0.5 quantile

                # Smaller spread = higher confidence
                spread = np.mean(q3 - q1)
                confidence = 1.0 / (1.0 + spread)  # Inverse relationship
            else:
                # For other prediction formats, use variance
                variance = np.var(prediction)
                confidence = 1.0 / (1.0 + variance)

            # Factor in model's historical performance
            historical_performance = self.model_performance[model_name]['accuracy']
            combined_confidence = 0.7 * confidence + 0.3 * historical_performance

            return np.clip(combined_confidence, 0.1, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating confidence for {model_name}: {e}")
            return 0.5  # Default neutral confidence

    def _combine_predictions(self, predictions, confidences):
        """Combine predictions from multiple models"""
        if not predictions:
            raise ValueError("No predictions to combine")

        # Get a reference shape from the first prediction
        first_pred = next(iter(predictions.values()))
        target_shape = first_pred.shape

        # Ensure all predictions have the same shape
        aligned_predictions = {}
        for model_name, pred in predictions.items():
            if pred.shape != target_shape:
                # Try to reshape or interpolate to match
                try:
                    if len(pred.shape) == 2 and len(target_shape) == 3:
                        # Add quantile dimension
                        pred = np.expand_dims(pred, axis=-1)
                        pred = np.repeat(pred, target_shape[-1], axis=-1)
                    elif pred.shape[:-1] == target_shape[:-1] and pred.shape[-1] != target_shape[-1]:
                        # Different number of quantiles - interpolate
                        pred = self._interpolate_quantiles(pred, target_shape[-1])

                    aligned_predictions[model_name] = pred
                except Exception as e:
                    self.logger.warning(f"Could not align {model_name} prediction shape: {e}")
                    continue
            else:
                aligned_predictions[model_name] = pred

        if not aligned_predictions:
            raise ValueError("No predictions could be aligned")

        # Apply combination method
        if self.combination_method == 'simple_average':
            return self._simple_average(aligned_predictions)
        elif self.combination_method == 'weighted_average':
            return self._weighted_average(aligned_predictions, confidences)
        elif self.combination_method == 'confidence_weighted':
            return self._confidence_weighted_average(aligned_predictions, confidences)
        elif self.combination_method == 'performance_weighted':
            return self._performance_weighted_average(aligned_predictions)
        elif self.combination_method == 'stacking':
            return self._stacking_combination(aligned_predictions)
        else:
            self.logger.warning(f"Unknown combination method: {self.combination_method}, using weighted average")
            return self._weighted_average(aligned_predictions, confidences)

    def _simple_average(self, predictions):
        """Simple average of all predictions"""
        pred_arrays = list(predictions.values())
        return np.mean(pred_arrays, axis=0)

    def _weighted_average(self, predictions, confidences):
        """Weighted average using model weights"""
        weighted_sum = None
        total_weight = 0

        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self._simple_average(predictions)

    def _confidence_weighted_average(self, predictions, confidences):
        """Weight predictions by their confidence scores"""
        weighted_sum = None
        total_confidence = 0

        for model_name, pred in predictions.items():
            confidence = confidences.get(model_name, 0.5)
            if weighted_sum is None:
                weighted_sum = pred * confidence
            else:
                weighted_sum += pred * confidence
            total_confidence += confidence

        if total_confidence > 0:
            return weighted_sum / total_confidence
        else:
            return self._simple_average(predictions)

    def _performance_weighted_average(self, predictions):
        """Weight predictions by historical performance"""
        weighted_sum = None
        total_performance = 0

        for model_name, pred in predictions.items():
            performance = self.model_performance[model_name]['accuracy']
            if weighted_sum is None:
                weighted_sum = pred * performance
            else:
                weighted_sum += pred * performance
            total_performance += performance

        if total_performance > 0:
            return weighted_sum / total_performance
        else:
            return self._simple_average(predictions)

    def _stacking_combination(self, predictions):
        """Advanced stacking combination (simplified version)"""
        # For now, use performance-weighted average
        # In a full implementation, this would use a meta-learner
        return self._performance_weighted_average(predictions)

    def _interpolate_quantiles(self, prediction, target_quantiles):
        """Interpolate prediction to match target number of quantiles"""
        # This is a simplified interpolation
        # In practice, you might want more sophisticated methods
        if prediction.shape[-1] == 1:
            # Single prediction - replicate for all quantiles
            return np.repeat(prediction, target_quantiles, axis=-1)
        else:
            # Linear interpolation between existing quantiles
            from scipy.interpolate import interp1d

            old_quantiles = np.linspace(0, 1, prediction.shape[-1])
            new_quantiles = np.linspace(0, 1, target_quantiles)

            interpolated = np.zeros(prediction.shape[:-1] + (target_quantiles,))

            for i in range(prediction.shape[0]):
                for j in range(prediction.shape[1]):
                    f = interp1d(old_quantiles, prediction[i, j, :], kind='linear', fill_value='extrapolate')
                    interpolated[i, j, :] = f(new_quantiles)

            return interpolated

    def _store_prediction_info(self, individual_predictions, combined_prediction):
        """Store prediction information for performance tracking"""
        prediction_info = {
            'timestamp': datetime.now(),
            'individual_predictions': individual_predictions,
            'combined_prediction': combined_prediction,
            'model_weights': self.model_weights.copy()
        }

        self.prediction_history.append(prediction_info)

        # Limit history size
        if len(self.prediction_history) > self.performance_window * 2:
            self.prediction_history = self.prediction_history[-self.performance_window:]

    def update_performance(self, actual_outcome, prediction_timestamp=None):
        """
        Update model performance based on actual outcomes

        Parameters:
        - actual_outcome: The actual market outcome
        - prediction_timestamp: When the prediction was made (optional)
        """
        if not self.prediction_history:
            return

        with self.performance_lock:
            # Find the most recent prediction or specific one
            if prediction_timestamp:
                target_prediction = None
                for pred_info in reversed(self.prediction_history):
                    if pred_info['timestamp'] == prediction_timestamp:
                        target_prediction = pred_info
                        break
            else:
                target_prediction = self.prediction_history[-1]

            if not target_prediction:
                return

            # Calculate errors for each model
            for model_name, prediction in target_prediction['individual_predictions'].items():
                try:
                    # Calculate prediction error
                    if isinstance(prediction, np.ndarray) and len(prediction.shape) >= 2:
                        # Use median prediction for error calculation
                        if prediction.shape[-1] >= 3:
                            pred_value = prediction[0, 0, 1]  # First batch, first time step, median
                        else:
                            pred_value = prediction[0, 0]
                    else:
                        pred_value = float(prediction)

                    error = abs(pred_value - actual_outcome)

                    # Update model performance
                    self.model_performance[model_name]['errors'].append(error)

                    # Keep only recent errors
                    if len(self.model_performance[model_name]['errors']) > self.performance_window:
                        self.model_performance[model_name]['errors'] = \
                            self.model_performance[model_name]['errors'][-self.performance_window:]

                    # Calculate new accuracy (inverse of average error)
                    avg_error = np.mean(self.model_performance[model_name]['errors'])
                    accuracy = 1.0 / (1.0 + avg_error)
                    self.model_performance[model_name]['accuracy'] = accuracy

                except Exception as e:
                    self.logger.error(f"Error updating performance for {model_name}: {e}")

            # Update weights if dynamic weighting is enabled
            if self.enable_dynamic_weighting:
                self._update_dynamic_weights()

    def _update_dynamic_weights(self):
        """Update model weights based on performance"""
        total_accuracy = sum(perf['accuracy'] for perf in self.model_performance.values())

        if total_accuracy > 0:
            for model_name in self.model_weights.keys():
                accuracy = self.model_performance[model_name]['accuracy']
                self.model_weights[model_name] = accuracy / total_accuracy

        self.logger.debug(f"Updated model weights: {self.model_weights}")

    def get_model_info(self):
        """Get information about all models in the ensemble"""
        info = {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'combination_method': self.combination_method,
            'model_weights': self.model_weights.copy(),
            'model_performance': {}
        }

        for model_name, perf in self.model_performance.items():
            info['model_performance'][model_name] = {
                'accuracy': perf['accuracy'],
                'recent_errors': len(perf['errors']),
                'avg_error': np.mean(perf['errors']) if perf['errors'] else 0
            }

        return info

    def save_ensemble(self, save_path):
        """Save the entire ensemble"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            ensemble_data = {
                'config': self.ensemble_config,
                'model_weights': self.model_weights,
                'model_performance': self.model_performance,
                'combination_method': self.combination_method
            }

            # Save ensemble metadata
            with open(f"{save_path}_metadata.json", 'w') as f:
                json.dump(ensemble_data, f, indent=2, default=str)

            # Save individual models
            for model_name, model in self.models.items():
                model_path = f"{save_path}_{model_name}.pt"

                if hasattr(model, 'save'):
                    model.save(model_path)
                elif hasattr(model, 'state_dict'):
                    torch.save(model.state_dict(), model_path)
                else:
                    # For non-PyTorch models (like XGBoost)
                    with open(model_path.replace('.pt', '.pkl'), 'wb') as f:
                        pickle.dump(model, f)

            self.logger.info(f"Ensemble saved to {save_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving ensemble: {e}")
            return False

    def load_ensemble(self, load_path):
        """Load a saved ensemble"""
        try:
            # Load metadata
            with open(f"{load_path}_metadata.json", 'r') as f:
                ensemble_data = json.load(f)

            self.model_weights = ensemble_data['model_weights']
            self.model_performance = ensemble_data['model_performance']
            self.combination_method = ensemble_data['combination_method']

            self.logger.info(f"Ensemble loaded from {load_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            return False