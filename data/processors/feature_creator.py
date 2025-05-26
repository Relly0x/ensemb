# data/processors/feature_creator.py

import pandas as pd
import numpy as np
import logging


class TechnicalIndicators:
    """
    Technical indicator calculations for feature engineering
    """

    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    @staticmethod
    def atr(high, low, close, window=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent


class PipelineFeatureCreator:
    """
    Feature creator for data processing pipeline
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('pipeline_feature_creator')

        # Feature configuration
        self.feature_config = config.get('features', {})

        self.logger.info("Pipeline feature creator initialized")

    def create_features(self, data):
        """
        Create features for the data processing pipeline

        Parameters:
        - data: Dictionary of dataframes by instrument and timeframe

        Returns:
        - Dictionary of processed data with features
        """
        processed_data = {}

        for instrument, timeframes in data.items():
            processed_data[instrument] = {}

            for timeframe, df in timeframes.items():
                # Create features for this timeframe
                features_df = self._create_timeframe_features(df, instrument, timeframe)
                processed_data[instrument][timeframe] = features_df

        return processed_data

    def _create_timeframe_features(self, df, instrument, timeframe):
        """
        Create features for a specific timeframe

        Parameters:
        - df: DataFrame with OHLCV data
        - instrument: Instrument name
        - timeframe: Timeframe string

        Returns:
        - DataFrame with features
        """
        try:
            # Copy original data
            features_df = df.copy()

            # Price-based features
            features_df = self._add_price_features(features_df)

            # Technical indicators
            features_df = self._add_technical_indicators(features_df)

            # Time-based features
            if isinstance(features_df.index, pd.DatetimeIndex):
                features_df = self._add_time_features(features_df)

            # Volume features
            if 'volume' in features_df.columns:
                features_df = self._add_volume_features(features_df)

            # Clean up features
            features_df = self._clean_features(features_df)

            self.logger.debug(f"Created {features_df.shape[1]} features for {instrument} {timeframe}")

            return features_df

        except Exception as e:
            self.logger.error(f"Error creating features for {instrument} {timeframe}: {e}")
            return df

    def _add_price_features(self, df):
        """Add price-based features"""
        # Moving averages
        for window in [10, 20, 50, 100, 200]:
            if len(df) >= window:
                df[f'sma_{window}'] = TechnicalIndicators.sma(df['close'], window)
                df[f'ema_{window}'] = TechnicalIndicators.ema(df['close'], window)

        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)

        # Volatility
        df['volatility_10'] = df['close'].rolling(window=10).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()

        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Open-Close spread
        df['oc_spread'] = (df['close'] - df['open']) / df['open']

        return df

    def _add_technical_indicators(self, df):
        """Add technical indicators"""
        # RSI
        df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)

        # MACD
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # ATR
        df['atr_14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        return df

    def _add_time_features(self, df):
        """Add time-based features"""
        # Hour of day
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Trading session flags
        df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        df['tokyo_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)

        # Remove raw time features
        df = df.drop(['hour', 'day_of_week', 'month'], axis=1)

        return df

    def _add_volume_features(self, df):
        """Add volume-based features"""
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()

        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price-volume trend
        df['pv_trend'] = (df['close'].pct_change() * df['volume']).rolling(window=10).mean()

        return df

    def _clean_features(self, df):
        """Clean and prepare features"""
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)

        return df