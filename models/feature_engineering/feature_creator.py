# models/feature_engineering/feature_creator.py

import pandas as pd
import numpy as np
import talib


class FeatureCreator:
    """
    Create features for TFT model
    """

    def __init__(self, config):
        self.config = config

        # Feature configuration
        self.use_ta_features = config.get('features', {}).get('use_ta_features', True)
        self.use_time_features = config.get('features', {}).get('use_time_features', True)
        self.use_volume_features = config.get('features', {}).get('use_volume_features', True)

    def create_features(self, data):
        """
        Create features from OHLCV data

        Parameters:
        - data: DataFrame with OHLCV data

        Returns:
        - DataFrame with features
        """
        df = data.copy()

        # Create basic price features
        df = self._create_price_features(df)

        # Create technical indicator features
        if self.use_ta_features:
            df = self._create_ta_features(df)

        # Create time-based features
        if self.use_time_features and isinstance(df.index, pd.DatetimeIndex):
            df = self._create_time_features(df)

        # Create volume-based features
        if self.use_volume_features and 'volume' in df.columns:
            df = self._create_volume_features(df)

        # Drop any NA values
        df = df.dropna()

        # Reset index for model input if DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)

        return df

    def _create_price_features(self, df):
        """Create basic price features"""
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential moving averages
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Price changes
        df['close_change'] = df['close'].pct_change()
        df['high_change'] = df['high'].pct_change()
        df['low_change'] = df['low'].pct_change()

        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()

        # Price to moving average relations
        df['close_sma_10_ratio'] = df['close'] / df['sma_10']
        df['close_sma_20_ratio'] = df['close'] / df['sma_20']
        df['close_sma_50_ratio'] = df['close'] / df['sma_50']

        # Moving average differences
        df['sma_10_20_diff'] = df['sma_10'] - df['sma_20']
        df['sma_20_50_diff'] = df['sma_20'] - df['sma_50']

        # Bollinger Bands
        df['bollinger_mid'] = df['sma_20']
        df['bollinger_std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_mid'] + 2 * df['bollinger_std']
        df['bollinger_lower'] = df['bollinger_mid'] - 2 * df['bollinger_std']
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_mid']
        df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (
                    df['bollinger_upper'] - df['bollinger_lower'])

        # Candle features
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df['close'].where(df['close'] >= df['open'], df['open'])) / df['open']
        df['lower_shadow'] = (df['close'].where(df['close'] <= df['open'], df['open']) - df['low']) / df['open']

        return df

    def _create_ta_features(self, df):
        """Create technical analysis features using TA-Lib"""
        try:
            # RSI
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            # ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

            # CCI
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

            # Normalized ATR
            df['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)

            # Ichimoku Cloud
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2

            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2

            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

        except Exception as e:
            # Handle case where TA-Lib is not available
            print(f"Error creating TA features: {e}")

        return df

    def _create_time_features(self, df):
        """Create time-based features"""
        # Basic calendar features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Cyclical time encoding (to handle periodicity)
        # Hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Market session flags
        df['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['is_new_york_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        df['is_tokyo_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)

        # Dro