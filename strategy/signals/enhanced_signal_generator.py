# strategy/signals/enhanced_signal_generator.py
# Enhanced signal generator focused on high-quality trend-following trades

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging


class HighQualitySignalGenerator:
    """
    Enhanced signal generator focused on high-quality trend-following trades
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('hq_signal_generator')

        # Enhanced thresholds for quality
        self.confidence_thresholds = config['strategy']['confidence_thresholds']
        self.min_signal_strength = 0.7  # Much higher threshold
        self.trend_filter = config['strategy'].get('trend_filter', True)

        # Trend following parameters
        self.trend_confirmation_required = True
        self.min_trend_strength = 0.6
        self.require_support_resistance = True
        self.require_fibonacci_confluence = True

        # Signal spacing (prevent overtrading)
        self.min_signal_gap_hours = config.get('strategy', {}).get('min_signal_gap_hours', 2)
        self.last_signals = {}  # instrument -> last_signal_time

        # Track generated signals for analysis
        self.signal_history = []
        self.max_history = 100

        self.logger.info("High-quality signal generator initialized")

    def generate_signal(self, prediction, market_data, timeframe_manager, instrument):
        """
        Generate HIGH QUALITY trading signal with strict trend following
        """

        # STEP 1: Basic data validation
        high_tf = self.config['data']['timeframes']['high']
        low_tf = self.config['data']['timeframes']['low']

        if instrument not in market_data or high_tf not in market_data[instrument]:
            return {'valid': False, 'reason': 'Missing market data'}

        high_tf_data = market_data[instrument][high_tf]
        low_tf_data = market_data[instrument][low_tf]

        if len(high_tf_data) == 0 or len(low_tf_data) == 0:
            return {'valid': False, 'reason': 'Insufficient data points'}

        current_price = low_tf_data['close'].iloc[-1]

        # STEP 2: Check minimum time gap since last signal
        if self._too_soon_for_signal(instrument):
            return {'valid': False, 'reason': 'Too soon since last signal'}

        # STEP 3: Analyze model prediction strength
        prediction_analysis = self._analyze_prediction_strength(prediction, current_price)

        if not prediction_analysis['strong_enough']:
            return {'valid': False, 'reason': f"Weak prediction: {prediction_analysis['reason']}"}

        # STEP 4: Get comprehensive market structure analysis
        market_structure = timeframe_manager.get_trading_signals()

        if not market_structure.get('valid', False):
            return {'valid': False, 'reason': market_structure.get('reason', 'Invalid market structure')}

        # STEP 5: STRICT TREND ANALYSIS - This is crucial!
        trend_analysis = self._analyze_trend_strength(high_tf_data, market_structure)

        if not trend_analysis['trend_confirmed']:
            return {'valid': False, 'reason': f"Trend not confirmed: {trend_analysis['reason']}"}

        # STEP 6: Support/Resistance confluence check
        sr_confluence = self._check_support_resistance_confluence(
            current_price, market_structure, trend_analysis['direction']
        )

        if not sr_confluence['valid']:
            return {'valid': False, 'reason': f"No S/R confluence: {sr_confluence['reason']}"}

        # STEP 7: Fibonacci confluence check
        fib_confluence = self._check_fibonacci_confluence(
            current_price, market_structure, trend_analysis['direction']
        )

        if not fib_confluence['valid']:
            return {'valid': False, 'reason': f"No Fibonacci confluence: {fib_confluence['reason']}"}

        # STEP 8: Final signal direction must align with trend
        ml_direction = prediction_analysis['direction']
        trend_direction = trend_analysis['direction']

        # CRITICAL: Only trade WITH the trend, never against it
        if ml_direction != trend_direction:
            return {'valid': False, 'reason': f"ML signal ({ml_direction}) against trend ({trend_direction})"}

        # STEP 9: Calculate composite signal strength
        composite_strength = self._calculate_composite_strength(
            prediction_analysis, trend_analysis, sr_confluence, fib_confluence
        )

        if composite_strength < self.min_signal_strength:
            return {'valid': False, 'reason': f'Composite strength too low ({composite_strength:.2f})'}

        # STEP 10: Calculate optimal entry levels
        entry_levels = self._calculate_optimal_entry(
            current_price, trend_direction, market_structure
        )

        # STEP 11: Create high-quality signal
        signal = {
            'valid': True,
            'signal': trend_direction,  # Always follow the trend
            'strength': composite_strength,
            'quality_grade': 'A' if composite_strength > 0.85 else 'B',
            'instrument': instrument,
            'current_price': current_price,
            'entry_price': entry_levels['entry'],
            'stop_loss': entry_levels['stop_loss'],
            'take_profit': entry_levels['take_profit'],

            # Analysis details
            'trend_strength': trend_analysis['strength'],
            'trend_direction': trend_direction,
            'prediction_confidence': prediction_analysis['confidence'],
            'sr_confluence': sr_confluence['strength'],
            'fib_confluence': fib_confluence['strength'],

            # Risk management
            'risk_reward_ratio': entry_levels['risk_reward'],
            'max_risk_percent': self.config['execution']['risk_per_trade'],

            # Timing
            'timestamp': datetime.now().isoformat(),
            'next_signal_allowed': self._calculate_next_signal_time(instrument),

            # Trade plan
            'trade_plan': {
                'entry_reason': f"High-quality {trend_direction} signal with {composite_strength:.1%} confidence",
                'trend_context': f"Strong {trend_direction} trend (strength: {trend_analysis['strength']:.2f})",
                'key_levels': {
                    'support': market_structure.get('nearest_support'),
                    'resistance': market_structure.get('nearest_resistance'),
                    'fibonacci': fib_confluence.get('level')
                }
            }
        }

        # Store signal for gap enforcement
        self.last_signals[instrument] = datetime.now()
        self._add_to_history(signal)

        self.logger.info(
            f"🟢 HIGH QUALITY {signal['quality_grade']}-grade signal: "
            f"{trend_direction.upper()} {instrument} "
            f"(Strength: {composite_strength:.1%}, Trend: {trend_analysis['strength']:.2f})"
        )

        return signal

    def _too_soon_for_signal(self, instrument):
        """Check if it's too soon for another signal"""
        if instrument not in self.last_signals:
            return False

        time_since_last = datetime.now() - self.last_signals[instrument]
        hours_since_last = time_since_last.total_seconds() / 3600

        return hours_since_last < self.min_signal_gap_hours

    def _analyze_prediction_strength(self, prediction, current_price):
        """Analyze ML prediction strength and direction"""
        try:
            # Extract quantile predictions [0.1, 0.5, 0.9]
            if len(prediction.shape) >= 2:
                median_pred = prediction[:, 1]  # 0.5 quantile
                lower_pred = prediction[:, 0]  # 0.1 quantile
                upper_pred = prediction[:, 2]  # 0.9 quantile
            else:
                # Fallback if different shape
                median_pred = prediction
                lower_pred = prediction * 0.9
                upper_pred = prediction * 1.1

            # Focus on short-term prediction (first step)
            short_term_median = median_pred[0] if len(median_pred) > 0 else median_pred
            short_term_lower = lower_pred[0] if len(lower_pred) > 0 else lower_pred
            short_term_upper = upper_pred[0] if len(upper_pred) > 0 else upper_pred

            # Calculate prediction change
            pred_change = (short_term_median - current_price) / current_price

            # Calculate prediction confidence (based on quantile spread)
            pred_range = short_term_upper - short_term_lower
            confidence = max(0, 1 - (pred_range / current_price * 10))  # Tighter ranges = higher confidence

            # Determine direction and strength
            min_change_threshold = 0.0005  # 0.05% minimum change (5 pips for EUR/USD)

            if pred_change > min_change_threshold:
                direction = 'buy'
                strength = min(abs(pred_change) * 1000, 1.0)  # Scale to 0-1
            elif pred_change < -min_change_threshold:
                direction = 'sell'
                strength = min(abs(pred_change) * 1000, 1.0)
            else:
                return {
                    'strong_enough': False,
                    'reason': f'Prediction change too small ({pred_change:.4f})',
                    'direction': 'neutral',
                    'strength': 0,
                    'confidence': confidence
                }

            # Check if prediction is strong enough
            min_strength = 0.3
            min_confidence = 0.4

            strong_enough = (strength >= min_strength and confidence >= min_confidence)

            return {
                'strong_enough': strong_enough,
                'reason': 'Strong prediction' if strong_enough else f'Weak: strength={strength:.2f}, conf={confidence:.2f}',
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'change_percent': pred_change
            }

        except Exception as e:
            self.logger.error(f"Error analyzing prediction: {e}")
            return {
                'strong_enough': False,
                'reason': f'Prediction analysis error: {e}',
                'direction': 'neutral',
                'strength': 0,
                'confidence': 0
            }

    def _analyze_trend_strength(self, data, market_structure):
        """Comprehensive trend analysis with strict requirements"""

        # Get trend direction from market structure
        trend_info = market_structure.get('trend', 'neutral')
        if isinstance(trend_info, dict):
            trend_direction = trend_info.get('direction', 'neutral')
            trend_strength = trend_info.get('strength', 0)
        else:
            trend_direction = trend_info
            trend_strength = 0.5  # Default if not provided

        # Additional trend confirmation using price action
        if len(data) >= 50:
            # Check moving average alignment
            close_prices = data['close']
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1]
            current_price = close_prices.iloc[-1]

            # Trend confirmation criteria
            ma_aligned = False
            price_above_ma = False

            if trend_direction == 'bullish':
                ma_aligned = sma_20 > sma_50
                price_above_ma = current_price > sma_20
            elif trend_direction == 'bearish':
                ma_aligned = sma_20 < sma_50
                price_above_ma = current_price < sma_20

            # Calculate trend score
            confirmations = 0
            if ma_aligned:
                confirmations += 1
            if price_above_ma:
                confirmations += 1
            if trend_strength > self.min_trend_strength:
                confirmations += 1

            # Require at least 2 out of 3 confirmations
            trend_confirmed = confirmations >= 2

            # Enhanced trend strength
            enhanced_strength = (trend_strength + (confirmations / 3.0)) / 2

        else:
            trend_confirmed = trend_strength > self.min_trend_strength
            enhanced_strength = trend_strength

        # Convert trend directions to signal directions
        if trend_direction == 'bullish':
            signal_direction = 'buy'
        elif trend_direction == 'bearish':
            signal_direction = 'sell'
        else:
            signal_direction = 'neutral'

        return {
            'trend_confirmed': trend_confirmed and signal_direction != 'neutral',
            'direction': signal_direction,
            'strength': enhanced_strength,
            'reason': f'Trend: {trend_direction}, strength: {enhanced_strength:.2f}, confirmed: {trend_confirmed}'
        }

    def _check_support_resistance_confluence(self, current_price, market_structure, direction):
        """Check for support/resistance confluence"""

        nearest_support = market_structure.get('nearest_support')
        nearest_resistance = market_structure.get('nearest_resistance')

        if direction == 'buy':
            # For buy signals, we want to be near support
            if nearest_support is None:
                return {'valid': False, 'reason': 'No support level identified', 'strength': 0}

            distance_to_support = abs(current_price - nearest_support) / current_price
            max_distance = 0.002  # 0.2% or 20 pips for EUR/USD

            if distance_to_support <= max_distance:
                strength = 1 - (distance_to_support / max_distance)
                return {
                    'valid': True,
                    'reason': f'Near support ({distance_to_support:.3%} away)',
                    'strength': strength,
                    'level': nearest_support
                }
            else:
                return {
                    'valid': False,
                    'reason': f'Too far from support ({distance_to_support:.3%})',
                    'strength': 0
                }

        elif direction == 'sell':
            # For sell signals, we want to be near resistance
            if nearest_resistance is None:
                return {'valid': False, 'reason': 'No resistance level identified', 'strength': 0}

            distance_to_resistance = abs(current_price - nearest_resistance) / current_price
            max_distance = 0.002  # 0.2%

            if distance_to_resistance <= max_distance:
                strength = 1 - (distance_to_resistance / max_distance)
                return {
                    'valid': True,
                    'reason': f'Near resistance ({distance_to_resistance:.3%} away)',
                    'strength': strength,
                    'level': nearest_resistance
                }
            else:
                return {
                    'valid': False,
                    'reason': f'Too far from resistance ({distance_to_resistance:.3%})',
                    'strength': 0
                }

        return {'valid': False, 'reason': 'Invalid direction', 'strength': 0}

    def _check_fibonacci_confluence(self, current_price, market_structure, direction):
        """Check for Fibonacci level confluence"""

        fib_info = market_structure.get('fibonacci', {})
        fib_levels = fib_info.get('levels', {})

        if not fib_levels:
            return {'valid': False, 'reason': 'No Fibonacci levels calculated', 'strength': 0}

        # Find closest Fibonacci level
        closest_fib = None
        min_distance = float('inf')

        for fib_ratio, fib_price in fib_levels.items():
            if fib_price is None:
                continue

            distance = abs(current_price - fib_price) / current_price
            if distance < min_distance:
                min_distance = distance
                closest_fib = (fib_ratio, fib_price)

        if closest_fib is None:
            return {'valid': False, 'reason': 'No valid Fibonacci levels', 'strength': 0}

        fib_ratio, fib_price = closest_fib
        max_distance = 0.001  # 0.1% or 10 pips

        # Check if we're close enough to a key Fibonacci level
        key_fib_levels = [0.382, 0.5, 0.618]  # Most important levels

        if min_distance <= max_distance and fib_ratio in key_fib_levels:
            strength = (1 - (min_distance / max_distance)) * 1.2  # Bonus for key levels
            strength = min(strength, 1.0)

            return {
                'valid': True,
                'reason': f'Near {fib_ratio} Fib ({min_distance:.3%} away)',
                'strength': strength,
                'level': fib_ratio,
                'price': fib_price
            }
        elif min_distance <= max_distance:
            strength = 1 - (min_distance / max_distance)
            return {
                'valid': True,
                'reason': f'Near {fib_ratio} Fib ({min_distance:.3%} away)',
                'strength': strength * 0.8,  # Lower weight for non-key levels
                'level': fib_ratio,
                'price': fib_price
            }
        else:
            return {
                'valid': False,
                'reason': f'Too far from nearest Fib {fib_ratio} ({min_distance:.3%})',
                'strength': 0
            }

    def _calculate_composite_strength(self, prediction_analysis, trend_analysis, sr_confluence, fib_confluence):
        """Calculate composite signal strength"""

        # Weighted combination of different factors
        weights = {
            'trend': 0.4,  # Trend is most important
            'prediction': 0.3,  # ML prediction strength
            'support_resistance': 0.2,  # S/R confluence
            'fibonacci': 0.1  # Fibonacci confluence
        }

        composite = (
                trend_analysis['strength'] * weights['trend'] +
                prediction_analysis['strength'] * weights['prediction'] +
                sr_confluence['strength'] * weights['support_resistance'] +
                fib_confluence['strength'] * weights['fibonacci']
        )

        return composite

    def _calculate_optimal_entry(self, current_price, direction, market_structure):
        """Calculate optimal entry, stop loss, and take profit levels"""

        # Get key levels
        nearest_support = market_structure.get('nearest_support', current_price * 0.99)
        nearest_resistance = market_structure.get('nearest_resistance', current_price * 1.01)

        if direction == 'buy':
            # Enter at current price (market order) or slightly better
            entry_price = current_price

            # Stop loss below support with buffer
            if nearest_support:
                stop_loss = nearest_support * 0.9995  # 5 pips below support
            else:
                stop_loss = current_price * 0.995  # 0.5% below current price

            # Take profit based on risk-reward ratio
            risk_distance = entry_price - stop_loss
            rr_ratio = self.config['execution']['take_profit']['value']
            take_profit = entry_price + (risk_distance * rr_ratio)

        else:  # sell
            # Enter at current price (market order) or slightly better
            entry_price = current_price

            # Stop loss above resistance with buffer
            if nearest_resistance:
                stop_loss = nearest_resistance * 1.0005  # 5 pips above resistance
            else:
                stop_loss = current_price * 1.005  # 0.5% above current price

            # Take profit based on risk-reward ratio
            risk_distance = stop_loss - entry_price
            rr_ratio = self.config['execution']['take_profit']['value']
            take_profit = entry_price - (risk_distance * rr_ratio)

        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': rr_ratio
        }

    def _calculate_next_signal_time(self, instrument):
        """Calculate when next signal is allowed for this instrument"""
        next_time = datetime.now()
        if instrument in self.last_signals:
            last_signal = self.last_signals[instrument]
            next_time = last_signal + timedelta(hours=self.min_signal_gap_hours)

        return next_time.isoformat()

    def _add_to_history(self, signal):
        """Add signal to history for analysis"""
        self.signal_history.append(signal)

        # Trim history if needed
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]

    def get_signal_statistics(self):
        """Get statistics on signal generation quality"""
        if not self.signal_history:
            return {"count": 0}

        valid_signals = [s for s in self.signal_history if s.get('valid', False)]

        if not valid_signals:
            return {"count": 0, "valid_signals": 0}

        # Count by quality grade
        a_grade = sum(1 for s in valid_signals if s.get('quality_grade') == 'A')
        b_grade = sum(1 for s in valid_signals if s.get('quality_grade') == 'B')

        # Count by direction
        buy_signals = sum(1 for s in valid_signals if s.get('signal') == 'buy')
        sell_signals = sum(1 for s in valid_signals if s.get('signal') == 'sell')

        # Average strength
        avg_strength = sum(s.get('strength', 0) for s in valid_signals) / len(valid_signals)

        return {
            "total_signals": len(self.signal_history),
            "valid_signals": len(valid_signals),
            "rejection_rate": 1 - (len(valid_signals) / len(self.signal_history)),
            "quality_grades": {"A": a_grade, "B": b_grade},
            "direction_split": {"buy": buy_signals, "sell": sell_signals},
            "average_strength": avg_strength,
            "instruments": list(set(s.get('instrument') for s in valid_signals))
        }