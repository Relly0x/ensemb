import numpy as np
import pandas as pd


class TrendAnalyzer:
    """
    Trend analysis and detection for trading strategies
    """

    def __init__(self, config):
        self.config = config

        # Default parameters
        self.short_period = config.get('strategy', {}).get('trend_short_period', 20)
        self.medium_period = config.get('strategy', {}).get('trend_medium_period', 50)
        self.long_period = config.get('strategy', {}).get('trend_long_period', 200)
        self.rsi_period = config.get('strategy', {}).get('rsi_period', 14)
        self.rsi_overbought = config.get('strategy', {}).get('rsi_overbought', 70)
        self.rsi_oversold = config.get('strategy', {}).get('rsi_oversold', 30)
        self.adx_period = config.get('strategy', {}).get('adx_period', 14)
        self.adx_threshold = config.get('strategy', {}).get('adx_threshold', 25)

    def calculate_indicators(self, data):
        """
        Calculate all trend indicators

        Parameters:
        - data: DataFrame with OHLC data

        Returns:
        - DataFrame with added indicator columns
        """
        df = data.copy()

        # Moving Averages
        df['sma_short'] = df['close'].rolling(window=self.short_period).mean()
        df['sma_medium'] = df['close'].rolling(window=self.medium_period).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_period).mean()

        # Exponential Moving Averages
        df['ema_short'] = df['close'].ewm(span=self.short_period, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=self.medium_period, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_period, adjust=False).mean()

        # MACD (Moving Average Convergence Divergence)
        df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        # RSI (Relative Strength Index)
        df = self._calculate_rsi(df)

        # ADX (Average Directional Index)
        df = self._calculate_adx(df)

        # Bollinger Bands
        bollinger_period = 20
        std_dev = 2
        df['bollinger_middle'] = df['close'].rolling(window=bollinger_period).mean()
        df['bollinger_std'] = df['close'].rolling(window=bollinger_period).std()
        df['bollinger_upper'] = df['bollinger_middle'] + std_dev * df['bollinger_std']
        df['bollinger_lower'] = df['bollinger_middle'] - std_dev * df['bollinger_std']
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']

        # Linear Regression
        if len(df) >= 20:
            df = self._calculate_linear_regression(df, window=20)

        return df

    def determine_trend(self, data):
        """
        Determine trend direction and strength based on indicators

        Parameters:
        - data: DataFrame with calculated indicators

        Returns:
        - Dictionary with trend information
        """
        if len(data) < self.long_period:
            return {'direction': 'neutral', 'strength': 0.0}

        # Ensure indicators have been calculated
        if 'sma_short' not in data.columns:
            data = self.calculate_indicators(data)

        # Get latest values
        last_idx = len(data) - 1

        # Collect trend signals
        trend_signals = []

        # Moving Average alignment
        if data['sma_short'].iloc[last_idx] > data['sma_medium'].iloc[last_idx] > data['sma_long'].iloc[last_idx]:
            trend_signals.append(('ma_alignment', 'bullish', 0.15))
        elif data['sma_short'].iloc[last_idx] < data['sma_medium'].iloc[last_idx] < data['sma_long'].iloc[last_idx]:
            trend_signals.append(('ma_alignment', 'bearish', 0.15))

        # Price above/below moving averages
        close = data['close'].iloc[last_idx]
        if close > data['sma_long'].iloc[last_idx]:
            trend_signals.append(('price_above_ma', 'bullish', 0.1))
        elif close < data['sma_long'].iloc[last_idx]:
            trend_signals.append(('price_above_ma', 'bearish', 0.1))

        # MACD signal
        if data['macd_line'].iloc[last_idx] > data['macd_signal'].iloc[last_idx]:
            trend_signals.append(('macd', 'bullish', 0.15))
        elif data['macd_line'].iloc[last_idx] < data['macd_signal'].iloc[last_idx]:
            trend_signals.append(('macd', 'bearish', 0.15))

        # MACD histogram direction
        if len(data) > 1 and data['macd_histogram'].iloc[last_idx] > data['macd_histogram'].iloc[last_idx - 1]:
            trend_signals.append(('macd_histogram', 'bullish', 0.05))
        elif len(data) > 1 and data['macd_histogram'].iloc[last_idx] < data['macd_histogram'].iloc[last_idx - 1]:
            trend_signals.append(('macd_histogram', 'bearish', 0.05))

        # RSI signals
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[last_idx]
            if not pd.isna(rsi):
                if rsi > 50:
                    trend_signals.append(('rsi', 'bullish', 0.05 * (rsi - 50) / 50))
                elif rsi < 50:
                    trend_signals.append(('rsi', 'bearish', 0.05 * (50 - rsi) / 50))

        # ADX trend strength
        if 'adx' in data.columns and 'plus_di' in data.columns and 'minus_di' in data.columns:
            adx = data['adx'].iloc[last_idx]
            if not pd.isna(adx) and adx > self.adx_threshold:
                # Strong trend, determine direction using DI
                if data['plus_di'].iloc[last_idx] > data['minus_di'].iloc[last_idx]:
                    trend_signals.append(('adx', 'bullish', 0.15 * min(adx / 50, 1.0)))
                else:
                    trend_signals.append(('adx', 'bearish', 0.15 * min(adx / 50, 1.0)))

        # Linear regression slope
        if 'regression_slope' in data.columns:
            slope = data['regression_slope'].iloc[last_idx]
            if not pd.isna(slope):
                if slope > 0:
                    trend_signals.append(('lin_reg', 'bullish', 0.1 * min(abs(slope) / 0.001, 1.0)))
                elif slope < 0:
                    trend_signals.append(('lin_reg', 'bearish', 0.1 * min(abs(slope) / 0.001, 1.0)))

        # Bollinger Band position
        if close > data['bollinger_upper'].iloc[last_idx]:
            trend_signals.append(('bollinger', 'bullish', 0.1))
        elif close < data['bollinger_lower'].iloc[last_idx]:
            trend_signals.append(('bollinger', 'bearish', 0.1))

        # Bollinger Band width (trending vs ranging)
        bb_width = data['bollinger_width'].iloc[last_idx]
        if len(data) >= 20:
            bb_width_mean = data['bollinger_width'].rolling(window=20).mean().iloc[last_idx]
            if not pd.isna(bb_width_mean) and bb_width > bb_width_mean:
                # Wide bands suggest trending market
                trend_signals.append(('bb_width', 'trending', 0.05))

        # Calculate overall trend direction and strength
        bullish_signals = [s for s in trend_signals if s[1] == 'bullish']
        bearish_signals = [s for s in trend_signals if s[1] == 'bearish']

        bullish_strength = sum(s[2] for s in bullish_signals)
        bearish_strength = sum(s[2] for s in bearish_signals)

        # Determine trend direction
        if bullish_strength > bearish_strength + 0.1:
            direction = 'bullish'
            strength = bullish_strength
        elif bearish_strength > bullish_strength + 0.1:
            direction = 'bearish'
            strength = bearish_strength
        else:
            direction = 'neutral'
            strength = max(bullish_strength, bearish_strength)

        # Normalize strength to 0-1 range
        strength = min(strength, 1.0)

        return {
            'direction': direction,
            'strength': strength,
            'signals': trend_signals,
            'bullish_strength': bullish_strength,
            'bearish_strength': bearish_strength
        }

    def _calculate_rsi(self, df):
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def _calculate_adx(self, df):
        """Calculate ADX and Directional Indicators"""
        # True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Directional Movement
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                 np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                  np.maximum(df['low'].shift(1) - df['low'], 0), 0)

        # Smoothed averages
        df['atr'] = df['true_range'].rolling(window=self.adx_period).mean()
        df['dm_plus_smooth'] = df['dm_plus'].rolling(window=self.adx_period).mean()
        df['dm_minus_smooth'] = df['dm_minus'].rolling(window=self.adx_period).mean()

        # Directional Indicators
        df['plus_di'] = 100 * (df['dm_plus_smooth'] / df['atr'])
        df['minus_di'] = 100 * (df['dm_minus_smooth'] / df['atr'])

        # ADX calculation
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=self.adx_period).mean()

        # Clean up temporary columns
        df.drop(['tr1', 'tr2', 'tr3', 'true_range', 'dm_plus', 'dm_minus',
                 'dm_plus_smooth', 'dm_minus_smooth', 'dx'], axis=1, inplace=True)

        return df

    def _calculate_linear_regression(self, df, window=20):
        """Calculate linear regression slope"""

        def calc_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope

        df['regression_slope'] = df['close'].rolling(window=window).apply(calc_slope, raw=False)
        return df