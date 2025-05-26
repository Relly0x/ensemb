# pipelines/training/ensemble_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.ensemble.ensemble_manager import EnsembleManager
from models.ensemble.xgboost_model import XGBoostEnsemble
from pipelines.training.trainer import TFTTrainer


class EnsembleTrainer:
    """
    Training pipeline for ensemble models
    """

    def __init__(self, config, train_loader, val_loader, test_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logging.getLogger('ensemble_trainer')

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize ensemble manager
        self.ensemble_manager = EnsembleManager(config)

        # Training parameters
        self.training_config = config.get('ensemble', {}).get('training', {})
        self.num_epochs = self.training_config.get('num_epochs', 50)
        self.patience = self.training_config.get('patience', 10)

        # Checkpoint dir
        self.checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'models/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger.info("Ensemble trainer initialized")

    def train_all_models(self):
        """Train all models in the ensemble"""
        results = {}

        # Train neural network models (TFT, LSTM, Transformer, CNN)
        nn_models = ['tft', 'lstm', 'transformer', 'cnn']
        for model_name in nn_models:
            if model_name in self.ensemble_manager.models:
                self.logger.info(f"Training {model_name} model...")
                try:
                    result = self._train_neural_network_model(model_name)
                    results[model_name] = result
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
                    results[model_name] = {'success': False, 'error': str(e)}

        # Train tree-based models (XGBoost)
        if 'xgboost' in self.ensemble_manager.models:
            self.logger.info("Training XGBoost model...")
            try:
                result = self._train_xgboost_model()
                results['xgboost'] = result
            except Exception as e:
                self.logger.error(f"Error training XGBoost: {e}")
                results['xgboost'] = {'success': False, 'error': str(e)}

        # Update ensemble weights based on validation performance
        self._update_ensemble_weights(results)

        # Save ensemble
        ensemble_path = os.path.join(self.checkpoint_dir, 'ensemble_models')
        self.ensemble_manager.save_ensemble(ensemble_path)

        return results

    def _train_neural_network_model(self, model_name):
        """Train a neural network model"""
        model = self.ensemble_manager.models[model_name]
        model = model.to(self.device)

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_config.get('learning_rate', 0.001),
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss function
        criterion = self._get_loss_function()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self._train_epoch(model, optimizer, criterion)

            # Validate
            val_loss = self._validate_epoch(model, criterion)

            # Update scheduler
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_model_checkpoint(model, model_name, epoch, val_loss, is_best=True)
            else:
                patience_counter += 1

            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })

            self.logger.info(f"{model_name} Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping {model_name} after {epoch + 1} epochs")
                break

        # Test evaluation
        test_loss = None
        if self.test_loader:
            test_loss = self._test_epoch(model, criterion)

        return {
            'success': True,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'epochs_trained': epoch + 1,
            'training_history': training_history
        }

    def _train_xgboost_model(self):
        """Train XGBoost model"""
        xgb_model = self.ensemble_manager.models['xgboost']

        # Convert data loaders to numpy arrays for XGBoost
        X_train, y_train = self._dataloader_to_numpy(self.train_loader)
        X_val, y_val = self._dataloader_to_numpy(self.val_loader)

        # Train model
        xgb_model.fit(X_train, y_train)

        # Validate
        val_predictions = xgb_model.predict({'past': torch.tensor(X_train[:100])})  # Sample for validation
        val_loss = np.mean((val_predictions[:, :, 1] - y_val[:100, :]) ** 2)  # MSE on median prediction

        # Test
        test_loss = None
        if self.test_loader:
            X_test, y_test = self._dataloader_to_numpy(self.test_loader)
            test_predictions = xgb_model.predict({'past': torch.tensor(X_test[:100])})
            test_loss = np.mean((test_predictions[:, :, 1] - y_test[:100, :]) ** 2)

        return {
            'success': True,
            'val_loss': val_loss,
            'test_loss': test_loss
        }

    def _dataloader_to_numpy(self, dataloader):
        """Convert dataloader to numpy arrays for tree-based models"""
        X_list = []
        y_list = []

        for batch in dataloader:
            # Extract features from past data
            past_data = batch['past'].numpy()
            target = batch['target'].numpy()

            # Flatten past data for XGBoost
            batch_size, seq_len, features = past_data.shape
            flattened_past = past_data.reshape(batch_size, -1)

            X_list.append(flattened_past)
            y_list.append(target)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return X, y

    def _get_loss_function(self):
        """Get appropriate loss function"""
        quantiles = self.config['model']['quantiles']

        def quantile_loss(y_pred, y_true):
            losses = []
            for i, q in enumerate(quantiles):
                if i < y_pred.shape[-1]:
                    errors = y_true - y_pred[:, :, i]
                    losses.append(torch.max((q - 1) * errors, q * errors).mean())

            if losses:
                return torch.mean(torch.stack(losses))
            else:
                return nn.MSELoss()(y_pred.mean(dim=-1), y_true)

        return quantile_loss

    def _train_epoch(self, model, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            target = batch['target']

            # Calculate loss
            loss = criterion(outputs, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, model, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward pass
                outputs = model(batch)
                target = batch['target']

                # Calculate loss
                loss = criterion(outputs, target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _test_epoch(self, model, criterion):
        """Test for one epoch"""
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                # Forward pass
                outputs = model(batch)
                target = batch['target']

                # Calculate loss
                loss = criterion(outputs, target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_model_checkpoint(self, model, model_name, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, f'best_{model_name}_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint_epoch_{epoch}.pt')

        torch.save(checkpoint, path)

    def _update_ensemble_weights(self, training_results):
        """Update ensemble weights based on training results"""
        # Calculate weights based on validation performance
        val_losses = {}
        for model_name, result in training_results.items():
            if result.get('success', False):
                val_loss = result.get('best_val_loss') or result.get('val_loss', float('inf'))
                val_losses[model_name] = val_loss

        if not val_losses:
            return

        # Inverse of validation loss as weight (better performance = higher weight)
        total_inverse_loss = sum(1.0 / loss for loss in val_losses.values())

        for model_name, val_loss in val_losses.items():
            weight = (1.0 / val_loss) / total_inverse_loss
            self.ensemble_manager.model_weights[model_name] = weight

        self.logger.info(f"Updated ensemble weights: {self.ensemble_manager.model_weights}")


# Enhanced main.py with ensemble support
def main_with_ensemble():
    """Enhanced main function with ensemble training"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced TFT Trading Bot with Ensemble")
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--mode', type=str, choices=['train', 'train_ensemble', 'backtest', 'live'], default='train')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble models')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Setup logging
    logger = logging.getLogger('main')

    if args.mode == 'train_ensemble' or args.ensemble:
        logger.info("Training ensemble models...")

        # Collect and process data
        from data.collectors.oanda_collector import OandaDataCollector
        from data.processors.normalizer import DataNormalizer
        from data.dataset import create_datasets

        collector = OandaDataCollector(config)
        data = collector.collect_training_data()

        processor = DataNormalizer(config)
        processed_data = processor.process(data)

        train_loader, val_loader, test_loader = create_datasets(processed_data, config)

        # Train ensemble
        ensemble_trainer = EnsembleTrainer(config, train_loader, val_loader, test_loader)
        results = ensemble_trainer.train_all_models()

        logger.info("Ensemble training completed!")
        for model_name, result in results.items():
            if result.get('success'):
                logger.info(f"{model_name}: Val Loss = {result.get('best_val_loss', 'N/A')}")
            else:
                logger.error(f"{model_name}: Failed - {result.get('error', 'Unknown error')}")

    elif args.mode == 'live' and args.ensemble:
        logger.info("Starting live trading with ensemble...")

        from strategy.strategy_factory import create_strategy
        from execution.execution_engine import ExecutionEngine
        from pipelines.inference.ensemble_inference_engine import EnsembleInferenceEngine

        # Load ensemble
        ensemble_engine = EnsembleInferenceEngine(config)
        ensemble_path = os.path.join(config.get('training', {}).get('checkpoint_dir', 'models/checkpoints'),
                                     'ensemble_models')
        ensemble_engine.load_ensemble(ensemble_path)

        # Create strategy and execution engine
        strategy = create_strategy(config)
        engine = ExecutionEngine(config, ensemble_engine, strategy)

        # Start live trading
        engine.start_live_trading()

    else:
        # Standard single model training/execution
        logger.info("Running standard mode...")
        # ... (existing main function logic)


if __name__ == "__main__":
    main_with_ensemble()