# data/storage.py

import os
import json
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime


class DataStorage:
    """
    Storage utilities for data and model files
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('data_storage')

        # Create data directories
        self.data_dir = config.get('data', {}).get('storage_dir', 'data/stored')
        self.model_dir = config.get('export', {}).get('export_dir', 'exported_models')

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger.info("Data storage initialized")

    def save_ohlcv_data(self, data, instrument, timeframe):
        """
        Save OHLCV data to CSV file

        Parameters:
        - data: DataFrame with OHLCV data
        - instrument: Instrument name
        - timeframe: Timeframe string

        Returns:
        - Path to saved file
        """
        try:
            # Create directory if it doesn't exist
            instrument_dir = os.path.join(self.data_dir, instrument.replace('/', '_'))
            os.makedirs(instrument_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d')
            filename = f"{instrument.replace('/', '_')}_{timeframe}_{timestamp}.csv"
            filepath = os.path.join(instrument_dir, filename)

            # Save to CSV
            data.to_csv(filepath)

            self.logger.info(f"Saved {len(data)} rows for {instrument} {timeframe} to {filepath}")

            return filepath

        except Exception as e:
            self.logger.error(f"Error saving OHLCV data: {e}")
            return None

    def load_ohlcv_data(self, instrument, timeframe, start_date=None, end_date=None):
        """
        Load OHLCV data from CSV file

        Parameters:
        - instrument: Instrument name
        - timeframe: Timeframe string
        - start_date: Optional start date filter
        - end_date: Optional end date filter

        Returns:
        - DataFrame with OHLCV data
        """
        try:
            # Get instrument directory
            instrument_dir = os.path.join(self.data_dir, instrument.replace('/', '_'))

            if not os.path.exists(instrument_dir):
                self.logger.warning(f"No data directory for {instrument}")
                return None

            # Find matching files
            files = [f for f in os.listdir(instrument_dir) if
                     f.startswith(f"{instrument.replace('/', '_')}_{timeframe}")]

            if not files:
                self.logger.warning(f"No data files for {instrument} {timeframe}")
                return None

            # Sort files by date (newest first)
            files.sort(reverse=True)

            # Load the newest file
            filepath = os.path.join(instrument_dir, files[0])
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)

            # Apply date filters if provided
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]

            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]

            self.logger.info(f"Loaded {len(data)} rows for {instrument} {timeframe} from {filepath}")

            return data

        except Exception as e:
            self.logger.error(f"Error loading OHLCV data: {e}")
            return None

    def save_model(self, model, model_name=None, metadata=None):
        """
        Save model to disk

        Parameters:
        - model: PyTorch model
        - model_name: Optional model name
        - metadata: Optional metadata dictionary

        Returns:
        - Path to saved model
        """
        try:
            import torch

            # Generate model name if not provided
            if model_name is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_name = f"tft_model_{timestamp}"

            # Generate filepath
            filepath = os.path.join(self.model_dir, f"{model_name}.pt")

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                'saved_at': datetime.now().isoformat(),
                'model_name': model_name
            })

            # Save model with metadata
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': metadata
            }, filepath)

            self.logger.info(f"Model saved to {filepath}")

            # Save model config
            config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            return filepath

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None

            def load_model(self, model_path):
                """
                Load model from disk

                Parameters:
                - model_path: Path to model file

                Returns:
                - Tuple of (model_state_dict, metadata)
                """
                try:
                    import torch

                    if not os.path.exists(model_path):
                        self.logger.error(f"Model file not found: {model_path}")
                        return None, None

                    # Load model
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

                    state_dict = checkpoint.get('model_state_dict')
                    metadata = checkpoint.get('metadata', {})

                    self.logger.info(f"Model loaded from {model_path}")

                    return state_dict, metadata

                except Exception as e:
                    self.logger.error(f"Error loading model: {e}")
                    return None, None

            def save_predictions(self, predictions, filename=None):
                """
                Save predictions to file

                Parameters:
                - predictions: Dictionary of predictions
                - filename: Optional filename

                Returns:
                - Path to saved file
                """
                try:
                    # Generate filename if not provided
                    if filename is None:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"predictions_{timestamp}.json"

                    filepath = os.path.join(self.data_dir, filename)

                    # Convert numpy arrays to lists for JSON serialization
                    serializable_predictions = {}
                    for key, value in predictions.items():
                        if isinstance(value, np.ndarray):
                            serializable_predictions[key] = value.tolist()
                        else:
                            serializable_predictions[key] = value

                    # Save to JSON
                    with open(filepath, 'w') as f:
                        json.dump(serializable_predictions, f, indent=2)

                    self.logger.info(f"Predictions saved to {filepath}")

                    return filepath

                except Exception as e:
                    self.logger.error(f"Error saving predictions: {e}")
                    return None

            def save_backtest_results(self, results, filename=None):
                """
                Save backtest results to file

                Parameters:
                - results: Backtest results dictionary
                - filename: Optional filename

                Returns:
                - Path to saved file
                """
                try:
                    # Generate filename if not provided
                    if filename is None:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"backtest_results_{timestamp}.json"

                    filepath = os.path.join(self.data_dir, filename)

                    # Save to JSON
                    with open(filepath, 'w') as f:
                        json.dump(results, f, indent=2, default=str)

                    self.logger.info(f"Backtest results saved to {filepath}")

                    return filepath

                except Exception as e:
                    self.logger.error(f"Error saving backtest results: {e}")
                    return None

            def list_stored_data(self):
                """
                List all stored data files

                Returns:
                - Dictionary of available data
                """
                try:
                    stored_data = {}

                    # List instruments
                    for item in os.listdir(self.data_dir):
                        item_path = os.path.join(self.data_dir, item)

                        if os.path.isdir(item_path):
                            # This is an instrument directory
                            instrument = item.replace('_', '/')
                            stored_data[instrument] = []

                            # List files in instrument directory
                            for file in os.listdir(item_path):
                                if file.endswith('.csv'):
                                    stored_data[instrument].append(file)

                    return stored_data

                except Exception as e:
                    self.logger.error(f"Error listing stored data: {e}")
                    return {}