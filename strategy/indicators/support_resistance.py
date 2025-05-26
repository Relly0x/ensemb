import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN


class SupportResistanceDetector:
    """
    Support and resistance level detection using various techniques
    """

    def __init__(self, config):
        self.config = config
        self.window_size = config.get('strategy', {}).get('sr_window_size', 20)
        self.price_tolerance = config.get('strategy', {}).get('sr_price_tolerance', 0.001)
        self.min_touches = config.get('strategy', {}).get('sr_min_touches', 2)
        self.cluster_distance = config.get('strategy', {}).get('sr_cluster_distance', 0.0005)
        self.importance_window = config.get('strategy', {}).get('sr_importance_window', 50)

    def identify_support_resistance_levels(self, data, method='all'):
        """
        Identify support and resistance levels using the specified method

        Parameters:
        - data: DataFrame with OHLC data
        - method: Detection method ('extrema', 'peaks', 'fractal', 'volume', 'all')

        Returns:
        - Dictionary with support and resistance levels
        """
        if method == 'extrema' or method == 'all':
            sr_extrema = self._find_extrema_points(data)
        else:
            sr_extrema = {'support': [], 'resistance': []}

        if method == 'peaks' or method == 'all':
            sr_peaks = self._find_price_peaks(data)
        else:
            sr_peaks = {'support': [], 'resistance': []}

        if method == 'fractal' or method == 'all':
            sr_fractal = self._find_fractal_levels(data)
        else:
            sr_fractal = {'support': [], 'resistance': []}

        if method == 'volume' or method == 'all':
            sr_volume = self._find_volume_levels(data)
        else:
            sr_volume = {'support': [], 'resistance': []}

        # Combine levels from all methods
        all_support = sr_extrema['support'] + sr_peaks['support'] + sr_fractal['support'] + sr_volume['support']
        all_resistance = sr_extrema['resistance'] + sr_peaks['resistance'] + sr_fractal['resistance'] + sr_volume[
            'resistance']

        # Cluster close levels
        support_levels = self._cluster_levels(all_support) if all_support else []
        resistance_levels = self._cluster_levels(all_resistance) if all_resistance else []

        # Rate importance of levels
        if len(data) > 0:
            current_price = data['close'].iloc[-1]

            # Sort by proximity to current price
            support_levels = sorted(support_levels, key=lambda x: abs(x - current_price))
            resistance_levels = sorted(resistance_levels, key=lambda x: abs(x - current_price))

            # Keep only the closest levels
            max_levels = 5  # Keep at most 5 levels
            support_levels = support_levels[:max_levels]
            resistance_levels = resistance_levels[:max_levels]

            # Filter support below and resistance above current price
            support_levels = [level for level in support_levels if level < current_price]
            resistance_levels = [level for level in resistance_levels if level > current_price]

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def _find_extrema_points(self, data):
        """Find support and resistance using local extrema points"""
        if len(data) < self.window_size:
            return {'support': [], 'resistance': []}

        # Get high and low prices
        highs = data['high'].values
        lows = data['low'].values

        # Find local maxima and minima
        max_idx = argrelextrema(highs, np.greater, order=self.window_size)[0]
        min_idx = argrelextrema(lows, np.less, order=self.window_size)[0]

        # Get resistance and support levels
        resistance_levels = [highs[i] for i in max_idx]
        support_levels = [lows[i] for i in min_idx]

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def _find_price_peaks(self, data):
        """Find support and resistance using price peaks with multiple touches"""
        if len(data) < self.window_size:
            return {'support': [], 'resistance': []}

        price_range = data['high'].max() - data['low'].min()
        tolerance = price_range * self.price_tolerance

        # Initialize level counters
        support_touches = {}
        resistance_touches = {}

        # Scan the price history
        for i in range(len(data)):
            # Current prices
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]

            # Check for support touches
            for level in list(support_touches.keys()):
                if abs(low - level) < tolerance:
                    # Count a touch
                    support_touches[level] += 1
                    break
            else:
                # Add new potential support level
                support_touches[low] = 1

            # Check for resistance touches
            for level in list(resistance_touches.keys()):
                if abs(high - level) < tolerance:
                    # Count a touch
                    resistance_touches[level] += 1
                    break
            else:
                # Add new potential resistance level
                resistance_touches[high] = 1

        # Filter levels with minimum touches
        support_levels = [level for level, touches in support_touches.items()
                          if touches >= self.min_touches]
        resistance_levels = [level for level, touches in resistance_touches.items()
                             if touches >= self.min_touches]

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def _find_fractal_levels(self, data):
        """Find support and resistance using Bill Williams' fractals"""
        if len(data) < 5:  # Need at least 5 candles for a fractal
            return {'support': [], 'resistance': []}

        support_levels = []
        resistance_levels = []

        # Identify bullish (support) and bearish (resistance) fractals
        for i in range(2, len(data) - 2):
            # Bearish fractal (resistance)
            if (data['high'].iloc[i] > data['high'].iloc[i - 2] and
                    data['high'].iloc[i] > data['high'].iloc[i - 1] and
                    data['high'].iloc[i] > data['high'].iloc[i + 1] and
                    data['high'].iloc[i] > data['high'].iloc[i + 2]):
                resistance_levels.append(data['high'].iloc[i])

            # Bullish fractal (support)
            if (data['low'].iloc[i] < data['low'].iloc[i - 2] and
                    data['low'].iloc[i] < data['low'].iloc[i - 1] and
                    data['low'].iloc[i] < data['low'].iloc[i + 1] and
                    data['low'].iloc[i] < data['low'].iloc[i + 2]):
                support_levels.append(data['low'].iloc[i])

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def _find_volume_levels(self, data):
        """Find support and resistance using volume profile"""
        if 'volume' not in data.columns or len(data) < self.window_size:
            return {'support': [], 'resistance': []}

        # Calculate price bins
        price_bins = np.linspace(data['low'].min(), data['high'].max(), 50)
        bin_width = price_bins[1] - price_bins[0]

        # Calculate volume profile
        volume_profile = np.zeros_like(price_bins)

        for i in range(len(data)):
            candle_low = data['low'].iloc[i]
            candle_high = data['high'].iloc[i]
            candle_volume = data['volume'].iloc[i]

            # Distribute volume across price bins that the candle covers
            for j, price in enumerate(price_bins):
                if candle_low <= price <= candle_high:
                    volume_profile[j] += candle_volume

        # Find local maxima in volume profile
        high_volume_indices = argrelextrema(volume_profile, np.greater, order=3)[0]
        high_volume_prices = [price_bins[i] for i in high_volume_indices]

        # Separate into support and resistance
        current_price = data['close'].iloc[-1]
        support_levels = [price for price in high_volume_prices if price < current_price]
        resistance_levels = [price for price in high_volume_prices if price > current_price]

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def _cluster_levels(self, levels, eps=None):
        """Cluster similar price levels together using DBSCAN"""
        if not levels:
            return []

        # Convert to numpy array and reshape for DBSCAN
        levels_array = np.array(levels).reshape(-1, 1)

        # Calculate clustering distance if not provided
        if eps is None:
            price_range = np.max(levels_array) - np.min(levels_array)
            eps = price_range * self.cluster_distance

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=1).fit(levels_array)

        # Get cluster centers (average level in each cluster)
        unique_labels = set(clustering.labels_)
        clustered_levels = []

        for label in unique_labels:
            cluster_mask = clustering.labels_ == label
            cluster_center = np.mean(levels_array[cluster_mask])
            clustered_levels.append(cluster_center)

        return clustered_levels

    def calculate_level_strength(self, data, level, is_support=True):
        """Calculate the strength of a support/resistance level"""
        if len(data) == 0:
            return 0

        # Count touches
        touches = 0
        bounces = 0
        price_range = data['high'].max() - data['low'].min()
        tolerance = price_range * self.price_tolerance

        for i in range(min(len(data), self.importance_window)):
            idx = len(data) - 1 - i  # Start from the most recent candle
            if idx < 0:
                continue

            low = data['low'].iloc[idx]
            high = data['high'].iloc[idx]
            close = data['close'].iloc[idx]

            if is_support:
                # Count support touches
                if abs(low - level) < tolerance:
                    touches += 1
                    # Count if price bounced up from support
                    if close > low + tolerance:
                        bounces += 1
            else:
                # Count resistance touches
                if abs(high - level) < tolerance:
                    touches += 1
                    # Count if price bounced down from resistance
                    if close < high - tolerance:
                        bounces += 1

        # Calculate strength based on touches and bounces
        if touches == 0:
            return 0

        # More weight to bounces vs. touches that broke through
        strength = 0.5 * min(touches / 5, 1.0) + 0.5 * (bounces / max(touches, 1))

        # Add recency factor - more recent touches are more important
        recency = 1.0  # Scale factor for recency

        return min(strength * recency, 1.0)  # Cap at 1.0