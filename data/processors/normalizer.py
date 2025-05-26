# data/processors/normalizer.py

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import os


class DataNormalizer:
    """
    Data normalization for TFT model training
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('data_normalizer')

        # Normalization method
        self.method = config.get('preprocessing', {}).get('normalization_method', 'standard')

        # Initialize scaler
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown normalization method: {self.method}")
            self.scaler = StandardScaler()

        # Track if scaler is fitted
        self.is_fitted = False
        self.feature_names = None

        self.logger.info(f"Data normalizer initialized with {self.method} scaling")

    def process(self, data):
        """
        Process raw data for training

        Parameters:
        - data: Dictionary of dataframes by instrument and timeframe

        Returns:
        - Dictionary of processed dataframes ready for model training
        """
        processed_data = {}
        all_features = []  # Collect all features for fitting scaler

        # First pass: create features for all instruments/timeframes
        for instrument, timeframes in data.items():
            processed_data[instrument] = {}

            for timeframe, df in timeframes.items():
                self.logger.info(f"Creating features for {instrument} {timeframe}")

                # Create features
                features_df = self._create_features(df)

                # Store processed data
                processed_data[instrument][timeframe] = features_df

                # Collect features for scaler fitting (use high timeframe data)
                if timeframe == self.config['data']['timeframes']['high']:
                    # Select only numeric columns
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        all_features.append(features_df[numeric_cols])

        # Fit scaler on all high timeframe data
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            self.fit(combined_features)
            self.logger.info(f"Scaler fitted on combined data with {combined_features.shape[0]} samples")

        # Second pass: normalize all data
        for instrument in processed_data:
            for timeframe in processed_data[instrument]:
                self.logger.info(f"Normalizing {instrument} {timeframe}")
                normalized_df = self._normalize_data(processed_data[instrument][timeframe])
                processed_data[instrument][timeframe] = normalized_df

        return processed_data

    def _create_features(self, df):
        """
        Create features from OHLCV data

        Parameters:
        - df: DataFrame with OHLCV data

        Returns:
        - DataFrame with features
        """
        features_df = df.copy()

        # Basic price features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(features_df) >= window:
                features_df[f'sma_{window}'] = features_df['close'].rolling(window=window).mean()
                features_df[f'ema_{window}'] = features_df['close'].ewm(span=window).mean()

        # Volatility
        features_df['volatility'] = features_df['returns'].rolling(window=20).std()

        # Price ratios
        features_df['hl_ratio'] = features_df['high'] / features_df['low']
        features_df['oc_ratio'] = features_df['open'] / features_df['close']

        # Volume features (if available)
        if 'volume' in features_df.columns:
            features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']

        # Technical indicators
        features_df = self._add_technical_indicators(features_df)

        # Fill NaN values (fix deprecation warning)
        features_df = features_df.ffill().fillna(0)

        # Replace infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)

        return features_df

    def _add_technical_indicators(self, df):
        """Add basic technical indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")

        return df

    def _normalize_data(self, df):
        """
        Normalize the data using the fitted scaler

        Parameters:
        - df: DataFrame with features

        Returns:
        - Normalized DataFrame
        """
        try:
            if not self.is_fitted:
                self.logger.warning("Scaler not fitted, skipping normalization")
                return df

            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns found")
                return df

            # Get numeric data
            numeric_data = df[numeric_cols]

            # Transform data
            normalized_data = self.scaler.transform(numeric_data)

            # Create normalized DataFrame
            normalized_df = pd.DataFrame(
                normalized_data,
                index=df.index,
                columns=numeric_cols
            )

            # Add back non-numeric columns if any
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                normalized_df[col] = df[col]

            return normalized_df

        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            return df

    def fit(self, data):
        """
        Fit the normalizer on training data

        Parameters:
        - data: DataFrame or numpy array

        Returns:
        - Self for chaining
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Select only numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found in data")

                data_array = data[numeric_cols].values
                self.feature_names = numeric_cols.tolist()
            else:
                data_array = data

            # Remove any infinite or NaN values
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Fit the scaler
            self.scaler.fit(data_array)
            self.is_fitted = True

            self.logger.info(f"Normalizer fitted on data with shape {data_array.shape}")

            return self

        except Exception as e:
            self.logger.error(f"Error fitting normalizer: {e}")
            raise

    def transform(self, data):
        """
        Transform data using fitted normalizer

        Parameters:
        - data: DataFrame or numpy array

        Returns:
        - Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        try:
            is_dataframe = isinstance(data, pd.DataFrame)

            if is_dataframe:
                # Select only numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data_array = data[numeric_cols].values
                index = data.index
                columns = numeric_cols
            else:
                data_array = data

            # Remove any infinite or NaN values
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Transform the data
            normalized_array = self.scaler.transform(data_array)

            # Return same format as input
            if is_dataframe:
                return pd.DataFrame(normalized_array, index=index, columns=columns)
            else:
                return normalized_array

        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise

    def inverse_transform(self, data):
        """
        Inverse transform normalized data back to original scale

        Parameters:
        - data: Normalized DataFrame or numpy array

        Returns:
        - Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        try:
            is_dataframe = isinstance(data, pd.DataFrame)

            if is_dataframe:
                data_array = data.values
                index = data.index
                columns = data.columns
            else:
                data_array = data

            # Inverse transform the data
            original_array = self.scaler.inverse_transform(data_array)

            # Return same format as input
            if is_dataframe:
                return pd.DataFrame(original_array, index=index, columns=columns)
            else:
                return original_array

        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {e}")
            raise

    def save_scaler(self, filepath):
        """
        Save fitted scaler to file

        Parameters:
        - filepath: Path to save scaler

        Returns:
        - True if successful
        """
        try:
            if not self.is_fitted:
                raise ValueError("Cannot save unfitted scaler")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save scaler and metadata
            scaler_data = {
                'scaler': self.scaler,
                'method': self.method,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }

            with open(filepath, 'wb') as f:
                pickle.dump(scaler_data, f)

            self.logger.info(f"Scaler saved to {filepath}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving scaler: {e}")
            return False