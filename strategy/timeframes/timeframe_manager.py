import pandas as pd
from ..indicators.support_resistance import SupportResistanceDetector
from ..indicators.fibonacci import FibonacciLevels
from ..indicators.trend import TrendAnalyzer

class TimeframeManager:
    def __init__(self, config):
        self.config = config
        self.high_tf = config['data']['timeframes']['high']
        self.low_tf = config['data']['timeframes']['low']

        # Create indicators for each timeframe
        self.sr_detector = SupportResistanceDetector(config)
        self.fib_calculator = FibonacciLevels(config)
        self.trend_analyzer = TrendAnalyzer(config)

        # Store data for each timeframe
        self.data = {
            self.high_tf: None,
            self.low_tf: None
        }

        # Store analyzed indicators
        self.indicators = {
            self.high_tf: {},
            self.low_tf: {}
        }

    def update_data(self, timeframe, data):
        """Update data for a specific timeframe"""
        self.data[timeframe] = data
        self._analyze_timeframe(timeframe)

    def _analyze_timeframe(self, timeframe):
        """Run all indicator calculations for a timeframe"""
        if self.data[timeframe] is None:
            return

        data = self.data[timeframe]

        # Analyze trend
        trend_data = self.trend_analyzer.calculate_indicators(data.copy())
        trend_info = self.trend_analyzer.determine_trend(trend_data)

        # Find support/resistance levels
        sr_levels = self.sr_detector.identify_support_resistance_levels(data)

        # Calculate Fibonacci levels
        fib_info = self.fib_calculator.find_optimal_fib_points(data)

        # Store all indicators
        self.indicators[timeframe] = {
            'trend': trend_info,
            'support_resistance': sr_levels,
            'fibonacci': fib_info
        }

    def get_trading_signals(self):
        """
        Generate trading signals based on multi-timeframe analysis

        Returns:
        - Dictionary with signal information
        """
        if self.high_tf not in self.indicators or self.low_tf not in self.indicators:
            return {'valid': False, 'reason': 'Missing timeframe data'}

        # Get high timeframe trend
        high_tf_trend = self.indicators[self.high_tf]['trend']['direction']

        # Respect trend direction filter
        if self.config['strategy']['trend_filter'] and high_tf_trend == 'neutral':
            return {'valid': False, 'reason': 'No clear trend direction'}

        # Get support/resistance levels from high timeframe
        sr_levels = self.indicators[self.high_tf]['support_resistance']

        # Get current price from low timeframe
        if self.data[self.low_tf] is None or len(self.data[self.low_tf]) == 0:
            return {'valid': False, 'reason': 'No low timeframe data'}

        current_price = self.data[self.low_tf].iloc[-1]['close']

        # Check if price is near support/resistance
        nearest_support = min(sr_levels['support'], key=lambda x: abs(x - current_price)) if sr_levels['support'] else None
        nearest_resistance = min(sr_levels['resistance'], key=lambda x: abs(x - current_price)) if sr_levels['resistance'] else None

        # Get Fibonacci levels
        fib_levels = self.indicators[self.high_tf]['fibonacci']['levels']

        # Determine if price is at significant Fibonacci level
        fib_level = self._find_nearest_fib_level(current_price, fib_levels)

        # Calculate signal strength based on confluence factors
        signal_strength = self._calculate_signal_strength(high_tf_trend, current_price, nearest_support, nearest_resistance, fib_level)

        # Determine signal type
        if high_tf_trend == 'bullish' and abs(current_price - nearest_support) / current_price < 0.002:
            signal = 'buy'
        elif high_tf_trend == 'bearish' and abs(current_price - nearest_resistance) / current_price < 0.002:
            signal = 'sell'
        else:
            return {'valid': False, 'reason': 'No clear entry point'}

        return {
            'valid': True,
            'signal': signal,
            'strength': signal_strength,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'fib_level': fib_level,
            'trend': high_tf_trend
        }

    def _find_nearest_fib_level(self, price, fib_levels):
        """Find the nearest Fibonacci level to current price"""
        if not fib_levels or not fib_levels.values():
            return None

        all_levels = []
        for level_val, price_val in fib_levels.items():
            all_levels.append((level_val, price_val))

        nearest = min(all_levels, key=lambda x: abs(x[1] - price))
        return nearest

    def _calculate_signal_strength(self, trend, price, support, resistance, fib_level):
        """Calculate signal strength based on confluence factors"""
        strength = 0.0

        # Strong trend adds confidence
        if trend == 'bullish' or trend == 'bearish':
            trend_strength = self.indicators[self.high_tf]['trend']['strength']
            strength += trend_strength * 0.4

        # Price near support/resistance
        if support and abs(price - support) / price < 0.001:
            strength += 0.3
        if resistance and abs(price - resistance) / price < 0.001:
            strength += 0.3

        # Price at Fibonacci level
        if fib_level:
            fib_val, fib_price = fib_level
            if abs(price - fib_price) / price < 0.001:
                # Key Fibonacci levels add more strength
                if fib_val in [0.382, 0.618]:
                    strength += 0.3
                else:
                    strength += 0.2

        return min(strength, 1.0)  # Cap at 1.0
