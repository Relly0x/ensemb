import numpy as np

class FibonacciLevels:
    def __init__(self, config):
        self.levels = config['strategy']['fibonacci_levels']

    def calculate_retracement_levels(self, high, low, trend='uptrend'):
        """
        Calculate Fibonacci retracement levels

        Parameters:
        - high: The high price point
        - low: The low price point
        - trend: 'uptrend' or 'downtrend'

        Returns:
        - Dictionary of Fibonacci levels
        """
        if trend == 'uptrend':
            # For uptrend: measure from low to high
            diff = high - low
            levels = {
                level: high - diff * level for level in self.levels
            }
        else:
            # For downtrend: measure from high to low
            diff = high - low
            levels = {
                level: low + diff * level for level in self.levels
            }

        return levels

    def find_optimal_fib_points(self, prices, lookback_window=100):
        """
        Find optimal points to draw Fibonacci retracements

        Parameters:
        - prices: DataFrame with OHLC data
        - lookback_window: How far back to look for swing hi/lo

        Returns:
        - Dictionary with swing points and Fibonacci levels
        """
        if len(prices) < lookback_window:
            lookback_window = len(prices)

        data = prices.iloc[-lookback_window:]

        # Find the highest high and lowest low
        highest_high = data['high'].max()
        highest_high_idx = data['high'].idxmax()

        lowest_low = data['low'].min()
        lowest_low_idx = data['low'].idxmin()

        # Determine trend
        if highest_high_idx > lowest_low_idx:
            trend = 'uptrend'
            levels = self.calculate_retracement_levels(highest_high, lowest_low, trend)
        else:
            trend = 'downtrend'
            levels = self.calculate_retracement_levels(highest_high, lowest_low, trend)

        return {
            'trend': trend,
            'swing_high': (highest_high_idx, highest_high),
            'swing_low': (lowest_low_idx, lowest_low),
            'levels': levels
        }
