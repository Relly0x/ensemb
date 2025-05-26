import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging


class TFTDataset(Dataset):
    """
    Simple dataset class for TFT model
    """

    def __init__(self, data, config, is_train=True):
        """
        Initialize dataset

        Parameters:
        - data: Dictionary of dataframes by instrument and timeframe
        - config: Configuration dictionary
        - is_train: Whether this is a training dataset
        """
        self.config = config
        self.is_train = is_train
        self.logger = logging.getLogger('tft_dataset')

        # Extract sequence parameters from config
        self.past_seq_len = config['model'].get('past_sequence_length', 120)
        self.forecast_horizon = config['model'].get('forecast_horizon', 12)

        # Process data to create samples
        self.samples = self._create_samples(data)
        self.logger.info(f"Created {len(self.samples)} samples for {'training' if is_train else 'validation'}")

    def _create_samples(self, data):
        """Create samples from raw data"""
        samples = []

        # Use high timeframe data for features
        high_tf = self.config['data']['timeframes']['high']

        # Process each instrument
        for instrument, timeframes in data.items():
            if high_tf not in timeframes:
                self.logger.warning(f"No {high_tf} data for {instrument}")
                continue

            df = timeframes[high_tf]

            # Skip if not enough data
            if len(df) < self.past_seq_len + self.forecast_horizon:
                self.logger.warning(
                    f"Insufficient data for {instrument}: {len(df)} < {self.past_seq_len + self.forecast_horizon}")
                continue

            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Ensure 'close' is in the data for target
            if 'close' not in numeric_cols:
                self.logger.error(f"No 'close' column found for {instrument}")
                continue

            # Create sequences
            for i in range(len(df) - self.past_seq_len - self.forecast_horizon + 1):
                # Past sequence
                past_start = i
                past_end = i + self.past_seq_len
                past_data = df.iloc[past_start:past_end]

                # Future sequence (excluding target variable for now)
                future_start = past_end
                future_end = future_start + self.forecast_horizon
                future_data = df.iloc[future_start:future_end]

                # Target (close prices for forecasting)
                target = future_data['close'].values

                # Features (all numeric columns except close for future)
                past_features = past_data[numeric_cols].values

                # For future features, use everything except close
                future_features_cols = [col for col in numeric_cols if col != 'close']
                if future_features_cols:
                    future_features = future_data[future_features_cols].values
                else:
                    # If no other features, create dummy features
                    future_features = np.zeros((self.forecast_horizon, 1))

                # Create sample
                sample = {
                    'instrument': instrument,
                    'past_data': past_features,
                    'future_data': future_features,
                    'target': target,
                    'static_data': np.array([0.0]),  # Dummy static data
                    'timestamp': past_data.index[-1] if hasattr(past_data.index, '__getitem__') else i
                }

                samples.append(sample)

        return samples

    def __len__(self):
        """Return number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample by index"""
        sample = self.samples[idx]

        # Convert to tensors
        past_tensor = torch.tensor(sample['past_data'], dtype=torch.float32)
        future_tensor = torch.tensor(sample['future_data'], dtype=torch.float32)
        static_tensor = torch.tensor(sample['static_data'], dtype=torch.float32)
        target_tensor = torch.tensor(sample['target'], dtype=torch.float32)

        return {
            'past': past_tensor,
            'future': future_tensor,
            'static': static_tensor,
            'target': target_tensor
        }


def create_datasets(data, config):
    """
    Create train, validation, and test datasets

    Parameters:
    - data: Dictionary of processed dataframes by instrument
    - config: Configuration dictionary

    Returns:
    - train_loader, val_loader, test_loader
    """
    logger = logging.getLogger('dataset_creator')

    # Dataset splits
    train_ratio = config.get('training', {}).get('train_ratio', 0.7)
    val_ratio = config.get('training', {}).get('val_ratio', 0.15)
    test_ratio = config.get('training', {}).get('test_ratio', 0.15)

    logger.info(f"Creating datasets with ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    # Split data by time
    train_data = {}
    val_data = {}
    test_data = {}

    for instrument, timeframes in data.items():
        train_data[instrument] = {}
        val_data[instrument] = {}
        test_data[instrument] = {}

        for timeframe, df in timeframes.items():
            # Sort by index if datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()

            # Calculate split points
            train_idx = int(len(df) * train_ratio)
            val_idx = int(len(df) * (train_ratio + val_ratio))

            # Split data
            train_data[instrument][timeframe] = df.iloc[:train_idx].copy()
            val_data[instrument][timeframe] = df.iloc[train_idx:val_idx].copy()
            test_data[instrument][timeframe] = df.iloc[val_idx:].copy()

            logger.info(f"{instrument} {timeframe}: Train={len(train_data[instrument][timeframe])}, "
                        f"Val={len(val_data[instrument][timeframe])}, Test={len(test_data[instrument][timeframe])}")

    # Create datasets
    train_dataset = TFTDataset(train_data, config, is_train=True)
    val_dataset = TFTDataset(val_data, config, is_train=False)
    test_dataset = TFTDataset(test_data, config, is_train=False)

    # Create data loaders
    batch_size = config['model'].get('batch_size', 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    logger.info(f"Created data loaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader