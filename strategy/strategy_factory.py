# Updated strategy_factory.py

import logging
from strategy.timeframes.timeframe_manager import TimeframeManager
from strategy.signals.signal_generator import HighQualitySignalGenerator  # Use enhanced version
from strategy.hooks.event_system import EventManager
from datetime import datetime, timedelta


class EnhancedTFTStrategy:
    """
    Enhanced TFT strategy focused on high-quality trend-following trades
    """

    def __init__(self, config, event_manager=None):
        self.config = config
        self.logger = logging.getLogger('enhanced_tft_strategy')

        # Initialize components with enhanced signal generator
        self.timeframe_manager = TimeframeManager(config)
        self.signal_generator = HighQualitySignalGenerator(config)  # Enhanced version

        # Set up event manager
        if event_manager:
            self.event_manager = event_manager
        else:
            self.event_manager = EventManager(config)

        # Strategy state with enhanced controls
        self.last_signals = {}  # instrument -> last signal details
        self.daily_trade_count = 0
        self.last_reset_day = None
        self.is_active = True

        # Quality controls
        self.max_daily_trades = config.get('execution', {}).get('max_daily_trades', 3)
        self.min_quality_grade = config.get('execution', {}).get('min_quality_grade', 'B')
        self.quality_filter_enabled = config.get('execution', {}).get('quality_filter_enabled', True)

        self.logger.info("Enhanced TFT Strategy initialized with quality controls")

    def update_data(self, market_data):
        """Update strategy with new market data"""
        # Reset daily counter if new day
        self._reset_daily_counters()

        # Process each instrument
        for instrument, timeframes in market_data.items():
            for timeframe, data in timeframes.items():
                # Update timeframe manager
                self.timeframe_manager.update_data(timeframe, data)

        # Notify of data update
        if self.event_manager:
            self.event_manager.emit('strategy:data_updated', {
                'instruments': list(market_data.keys()),
                'daily_trades': self.daily_trade_count,
                'max_daily_trades': self.max_daily_trades
            }, source='enhanced_tft_strategy')

    def generate_signals(self, predictions, market_data):
        """
        Generate HIGH QUALITY trading signals with strict filtering
        """
        if not self.is_active:
            self.logger.info("Strategy is inactive, no signals generated")
            return {}

        # Check daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            self.logger.info(f"Daily trade limit reached ({self.daily_trade_count}/{self.max_daily_trades})")
            return {}

        signals = {}
        high_quality_signals = 0

        for instrument, prediction in predictions.items():
            # Check if we already have a position or recent signal for this instrument
            if self._should_skip_instrument(instrument):
                continue

            # Generate signal using enhanced generator
            signal = self.signal_generator.generate_signal(
                prediction,
                market_data,
                self.timeframe_manager,
                instrument
            )

            # Apply quality filters
            if self.quality_filter_enabled and signal.get('valid', False):
                quality_check = self._apply_quality_filters(signal)
                if not quality_check['passed']:
                    signal['valid'] = False
                    signal['reason'] = f"Quality filter: {quality_check['reason']}"

            # Store signal (even if invalid for analysis)
            signals[instrument] = signal

            # Track valid high-quality signals
            if signal.get('valid', False):
                high_quality_signals += 1

                # Store as last signal for this instrument
                self.last_signals[instrument] = {
                    'signal': signal,
                    'timestamp': datetime.now()
                }

                # Increment daily counter
                self.daily_trade_count += 1

                # Emit high-quality signal event
                if self.event_manager:
                    self.event_manager.emit('strategy:hq_signal_generated', {
                        'signal': signal,
                        'quality_grade': signal.get('quality_grade', 'B'),
                        'daily_count': self.daily_trade_count,
                        'remaining_daily': self.max_daily_trades - self.daily_trade_count
                    }, source='enhanced_tft_strategy')

                self.logger.info(
                    f"✅ HIGH QUALITY {signal.get('quality_grade', 'B')}-grade signal generated: "
                    f"{signal['signal'].upper()} {instrument} "
                    f"(Strength: {signal['strength']:.1%}, Daily: {self.daily_trade_count}/{self.max_daily_trades})"
                )

        # Log generation summary
        total_signals = len([s for s in signals.values() if s.get('valid', False)])
        self.logger.info(
            f"Signal generation complete: {total_signals} valid signals from {len(predictions)} instruments")

        return signals

    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()

        if self.last_reset_day != today:
            self.daily_trade_count = 0
            self.last_reset_day = today
            self.logger.info(f"Daily counters reset for {today}")

    def _should_skip_instrument(self, instrument):
        """Check if we should skip signal generation for this instrument"""

        # Check if we have a recent signal
        if instrument in self.last_signals:
            last_signal_time = self.last_signals[instrument]['timestamp']
            time_since_last = datetime.now() - last_signal_time

            # Get minimum gap from config
            min_gap_hours = self.config.get('strategy', {}).get('min_signal_gap_hours', 2)
            min_gap = timedelta(hours=min_gap_hours)

            if time_since_last < min_gap:
                remaining = min_gap - time_since_last
                self.logger.debug(f"Skipping {instrument}: Recent signal {remaining} ago")
                return True

        return False

    def _apply_quality_filters(self, signal):
        """Apply additional quality filters to signals"""

        # Grade filter
        quality_grade = signal.get('quality_grade', 'C')
        if quality_grade < self.min_quality_grade:
            return {
                'passed': False,
                'reason': f"Grade {quality_grade} below minimum {self.min_quality_grade}"
            }

        # Strength filter
        strength = signal.get('strength', 0)
        min_strength = self.config.get('strategy', {}).get('min_signal_strength', 0.7)
        if strength < min_strength:
            return {
                'passed': False,
                'reason': f"Strength {strength:.2f} below minimum {min_strength}"
            }

        # Trend alignment filter
        trend_strength = signal.get('trend_strength', 0)
        min_trend_strength = self.config.get('strategy', {}).get('min_trend_strength', 0.6)
        if trend_strength < min_trend_strength:
            return {
                'passed': False,
                'reason': f"Trend strength {trend_strength:.2f} below minimum {min_trend_strength}"
            }

        # Risk-reward filter
        risk_reward = signal.get('risk_reward_ratio', 0)
        min_rr = self.config.get('execution', {}).get('take_profit', {}).get('value', 2.0)
        if risk_reward < min_rr:
            return {
                'passed': False,
                'reason': f"Risk-reward {risk_reward:.1f} below minimum {min_rr}"
            }

        return {'passed': True, 'reason': 'All quality filters passed'}

    def get_active_signals(self):
        """Get currently active high-quality signals"""
        active_signals = {}
        current_time = datetime.now()

        for instrument, signal_data in self.last_signals.items():
            signal = signal_data['signal']
            signal_time = signal_data['timestamp']

            # Consider signals active for 1 hour
            if signal.get('valid', False) and (current_time - signal_time).total_seconds() < 3600:
                active_signals[instrument] = signal

        return active_signals

    def validate_trade(self, instrument, direction, price):
        """
        Enhanced trade validation with quality checks
        """
        # Check if we have a recent high-quality signal for this instrument
        if instrument not in self.last_signals:
            self.logger.warning(f"No recent signal found for {instrument}")
            return False

        signal_data = self.last_signals[instrument]
        signal = signal_data['signal']
        signal_time = signal_data['timestamp']

        # Check if signal is still valid (within 30 minutes)
        time_since_signal = datetime.now() - signal_time
        if time_since_signal.total_seconds() > 1800:  # 30 minutes
            self.logger.warning(f"Signal for {instrument} too old ({time_since_signal})")
            return False

        # Check if signal is valid
        if not signal.get('valid', False):
            self.logger.warning(f"Invalid signal for {instrument}")
            return False

        # Check if directions match
        if signal.get('signal') != direction:
            self.logger.warning(f"Direction mismatch for {instrument}: {direction} vs {signal.get('signal')}")
            return False

        # Check if price hasn't moved too far from signal price
        signal_price = signal.get('current_price')
        if signal_price:
            # Allow 0.1% price deviation for quality signals
            max_deviation = signal_price * 0.001

            if abs(price - signal_price) > max_deviation:
                self.logger.warning(f"Price moved too far for {instrument}: {price} vs {signal_price}")
                return False

        # Check quality grade
        quality_grade = signal.get('quality_grade', 'C')
        if quality_grade < self.min_quality_grade:
            self.logger.warning(f"Signal quality too low for {instrument}: {quality_grade}")
            return False

        self.logger.info(f"✅ Trade validation passed for {instrument} {direction} (Grade: {quality_grade})")
        return True

    def get_strategy_stats(self):
        """Get strategy performance statistics"""
        signal_stats = self.signal_generator.get_signal_statistics()

        return {
            'daily_trades': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'remaining_daily': max(0, self.max_daily_trades - self.daily_trade_count),
            'signal_stats': signal_stats,
            'active_signals': len(self.get_active_signals()),
            'quality_filter_enabled': self.quality_filter_enabled,
            'min_quality_grade': self.min_quality_grade,
            'last_reset_day': self.last_reset_day.isoformat() if self.last_reset_day else None
        }

    def activate(self):
        """Activate the strategy"""
        self.is_active = True
        self.logger.info("Enhanced strategy activated")

        if self.event_manager:
            self.event_manager.emit('strategy:activated', {
                'strategy_type': 'enhanced_tft',
                'quality_controls': True
            }, source='enhanced_tft_strategy')

    def deactivate(self):
        """Deactivate the strategy"""
        self.is_active = False
        self.logger.info("Enhanced strategy deactivated")

        if self.event_manager:
            self.event_manager.emit('strategy:deactivated', {
                'strategy_type': 'enhanced_tft'
            }, source='enhanced_tft_strategy')


def create_strategy(config, event_manager=None, strategy_type=None):
    """
    Factory function to create enhanced strategy instance
    """
    if strategy_type is None:
        strategy_type = config.get('strategy', {}).get('type', 'enhanced_tft')

    if strategy_type == 'enhanced_tft' or strategy_type == 'tft':
        return EnhancedTFTStrategy(config, event_manager)
    else:
        # Fallback to enhanced version for any type
        return EnhancedTFTStrategy(config, event_manager)