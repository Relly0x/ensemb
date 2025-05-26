# strategy/enhanced_strategy_factory.py

import logging
from datetime import datetime, timedelta
from strategy.timeframes.timeframe_manager import TimeframeManager
from strategy.signals.enhanced_signal_generator import HighQualitySignalGenerator
from strategy.hooks.event_system import EventManager
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging


class EnsembleEnhancedStrategy:
    """
    Enhanced strategy that works with ensemble predictions
    """

    def __init__(self, config, event_manager=None, ensemble_engine=None):
        self.config = config
        self.ensemble_engine = ensemble_engine
        self.logger = logging.getLogger('ensemble_enhanced_strategy')

        # Initialize components
        self.timeframe_manager = TimeframeManager(config)
        self.signal_generator = HighQualitySignalGenerator(config)

        # Set up event manager
        if event_manager:
            self.event_manager = event_manager
        else:
            self.event_manager = EventManager(config)

        # Strategy state with ensemble controls
        self.last_signals = {}
        self.daily_trade_count = 0
        self.last_reset_day = None
        self.is_active = True

        # Ensemble-specific controls
        self.min_ensemble_confidence = config.get('strategy', {}).get('min_ensemble_confidence', 0.6)
        self.ensemble_confidence_weighting = config.get('strategy', {}).get('ensemble_confidence_weighting', True)
        self.require_model_agreement = config.get('strategy', {}).get('require_model_agreement', True)
        self.min_model_agreement = config.get('strategy', {}).get('min_model_agreement', 0.7)

        # Quality controls
        self.max_daily_trades = config.get('execution', {}).get('max_daily_trades', 3)
        self.min_quality_grade = config.get('execution', {}).get('min_quality_grade', 'B')
        self.quality_filter_enabled = config.get('execution', {}).get('quality_filter_enabled', True)

        self.logger.info("Ensemble Enhanced Strategy initialized with quality controls")

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
                'max_daily_trades': self.max_daily_trades,
                'ensemble_enabled': self.ensemble_engine is not None
            }, source='ensemble_enhanced_strategy')

    def generate_signals(self, predictions, market_data):
        """
        Generate HIGH QUALITY trading signals with ensemble prediction analysis

        Parameters:
        - predictions: Either single model predictions or ensemble predictions with metadata
        - market_data: Market data dictionary

        Returns:
        - Dictionary of enhanced signals with ensemble confidence
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

            # Extract ensemble metadata if available
            ensemble_metadata = self._extract_ensemble_metadata(prediction, instrument)

            # Apply ensemble confidence filter
            if not self._passes_ensemble_confidence_filter(ensemble_metadata):
                self.logger.debug(f"Signal for {instrument} rejected due to low ensemble confidence")
                continue

            # Generate signal using enhanced generator with ensemble context
            signal = self.signal_generator.generate_signal(
                prediction,
                market_data,
                self.timeframe_manager,
                instrument
            )

            # Enhance signal with ensemble information
            if ensemble_metadata:
                signal = self._enhance_signal_with_ensemble_data(signal, ensemble_metadata)

            # Apply quality filters (now includes ensemble filters)
            if self.quality_filter_enabled and signal.get('valid', False):
                quality_check = self._apply_enhanced_quality_filters(signal, ensemble_metadata)
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

                # Emit enhanced signal event
                if self.event_manager:
                    self.event_manager.emit('strategy:ensemble_hq_signal_generated', {
                        'signal': signal,
                        'quality_grade': signal.get('quality_grade', 'B'),
                        'ensemble_confidence': signal.get('ensemble_confidence', 0.5),
                        'model_agreement': signal.get('model_agreement', 0.5),
                        'daily_count': self.daily_trade_count,
                        'remaining_daily': self.max_daily_trades - self.daily_trade_count
                    }, source='ensemble_enhanced_strategy')

                self.logger.info(
                    f"✅ ENSEMBLE HIGH QUALITY {signal.get('quality_grade', 'B')}-grade signal: "
                    f"{signal['signal'].upper()} {instrument} "
                    f"(Strength: {signal['strength']:.1%}, "
                    f"Ensemble Conf: {signal.get('ensemble_confidence', 0.5):.1%}, "
                    f"Daily: {self.daily_trade_count}/{self.max_daily_trades})"
                )

        # Log generation summary
        total_signals = len([s for s in signals.values() if s.get('valid', False)])
        self.logger.info(
            f"Ensemble signal generation complete: {total_signals} valid signals from {len(predictions)} instruments")

        return signals

    def _extract_ensemble_metadata(self, prediction, instrument):
        """Extract ensemble metadata from prediction"""
        if self.ensemble_engine is None:
            return None

        # If prediction comes with metadata (from ensemble inference engine)
        if isinstance(prediction, dict) and 'metadata' in prediction:
            return prediction['metadata']

        # Try to get recent prediction metadata from ensemble engine
        try:
            if hasattr(self.ensemble_engine, 'ensemble_predictions') and self.ensemble_engine.ensemble_predictions:
                latest_prediction = self.ensemble_engine.ensemble_predictions[-1]
                if instrument in latest_prediction.get('metadata', {}):
                    return latest_prediction['metadata'][instrument]
        except:
            pass

        # Default metadata if not available
        return {
            'confidence': 0.5,
            'num_models': 1,
            'model_weights': {},
            'combination_method': 'single_model'
        }

    def _passes_ensemble_confidence_filter(self, ensemble_metadata):
        """Check if ensemble confidence meets minimum requirements"""
        if not ensemble_metadata:
            return True  # Pass if no ensemble metadata (single model)

        confidence = ensemble_metadata.get('confidence', 0.5)
        num_models = ensemble_metadata.get('num_models', 1)

        # Check minimum confidence
        if confidence < self.min_ensemble_confidence:
            return False

        # Check minimum number of models if required
        min_models = self.config.get('ensemble', {}).get('min_models_required', 2)
        if self.ensemble_engine and num_models < min_models:
            return False

        return True

    def _enhance_signal_with_ensemble_data(self, signal, ensemble_metadata):
        """Enhance signal with ensemble-specific information"""
        if not ensemble_metadata:
            return signal

        # Add ensemble metrics to signal
        signal['ensemble_confidence'] = ensemble_metadata.get('confidence', 0.5)
        signal['num_models'] = ensemble_metadata.get('num_models', 1)
        signal['model_weights'] = ensemble_metadata.get('model_weights', {})
        signal['combination_method'] = ensemble_metadata.get('combination_method', 'single')

        # Calculate model agreement score
        model_weights = ensemble_metadata.get('model_weights', {})
        if len(model_weights) > 1:
            # High agreement = similar weights, Low agreement = very different weights
            weight_values = list(model_weights.values())
            weight_std = np.std(weight_values) if len(weight_values) > 1 else 0
            # Convert to agreement score (lower std = higher agreement)
            signal['model_agreement'] = max(0, 1.0 - (weight_std * 2))
        else:
            signal['model_agreement'] = 1.0  # Single model = perfect agreement

        # Adjust signal strength based on ensemble confidence
        if self.ensemble_confidence_weighting:
            ensemble_confidence = ensemble_metadata.get('confidence', 0.5)
            original_strength = signal.get('strength', 0.5)
            # Weighted combination of original strength and ensemble confidence
            signal['strength'] = 0.7 * original_strength + 0.3 * ensemble_confidence

        return signal

    def _apply_enhanced_quality_filters(self, signal, ensemble_metadata):
        """Apply enhanced quality filters including ensemble-specific checks"""
        # Apply standard quality filters first
        base_quality_check = self._apply_quality_filters(signal)
        if not base_quality_check['passed']:
            return base_quality_check

        # Apply ensemble-specific filters
        if ensemble_metadata and self.ensemble_engine:
            # Model agreement filter
            if self.require_model_agreement:
                model_agreement = signal.get('model_agreement', 0.5)
                if model_agreement < self.min_model_agreement:
                    return {
                        'passed': False,
                        'reason': f"Model agreement {model_agreement:.2f} below minimum {self.min_model_agreement}"
                    }

            # Ensemble confidence filter (additional check)
            ensemble_confidence = signal.get('ensemble_confidence', 0.5)
            if ensemble_confidence < self.min_ensemble_confidence:
                return {
                    'passed': False,
                    'reason': f"Ensemble confidence {ensemble_confidence:.2f} below minimum {self.min_ensemble_confidence}"
                }

        return {'passed': True, 'reason': 'All enhanced quality filters passed'}

    def _apply_quality_filters(self, signal):
        """Apply standard quality filters (inherited from base class)"""
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

        return {'passed': True, 'reason': 'All standard quality filters passed'}

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

    def validate_trade(self, instrument, direction, price):
        """Enhanced trade validation with ensemble confidence checks"""
        # Standard validation first
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

        # Check ensemble confidence if available
        if self.ensemble_engine and 'ensemble_confidence' in signal:
            ensemble_confidence = signal.get('ensemble_confidence', 0.5)
            if ensemble_confidence < self.min_ensemble_confidence:
                self.logger.warning(f"Ensemble confidence too low for {instrument}: {ensemble_confidence:.2%}")
                return False

        # Check price deviation
        signal_price = signal.get('current_price')
        if signal_price:
            max_deviation = signal_price * 0.001
            if abs(price - signal_price) > max_deviation:
                self.logger.warning(f"Price moved too far for {instrument}: {price} vs {signal_price}")
                return False

        # Check quality grade
        quality_grade = signal.get('quality_grade', 'C')
        if quality_grade < self.min_quality_grade:
            self.logger.warning(f"Signal quality too low for {instrument}: {quality_grade}")
            return False

        ensemble_info = ""
        if 'ensemble_confidence' in signal:
            ensemble_info = f", Ensemble Conf: {signal['ensemble_confidence']:.1%}"

        self.logger.info(
            f"✅ Enhanced trade validation passed for {instrument} {direction} (Grade: {quality_grade}{ensemble_info})")
        return True

    def get_strategy_stats(self):
        """Get enhanced strategy performance statistics"""
        signal_stats = self.signal_generator.get_signal_statistics()

        base_stats = {
            'daily_trades': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'remaining_daily': max(0, self.max_daily_trades - self.daily_trade_count),
            'signal_stats': signal_stats,
            'active_signals': len(self.get_active_signals()),
            'quality_filter_enabled': self.quality_filter_enabled,
            'min_quality_grade': self.min_quality_grade,
            'last_reset_day': self.last_reset_day.isoformat() if self.last_reset_day else None
        }

        # Add ensemble-specific stats
        if self.ensemble_engine:
            ensemble_stats = {
                'ensemble_enabled': True,
                'min_ensemble_confidence': self.min_ensemble_confidence,
                'require_model_agreement': self.require_model_agreement,
                'min_model_agreement': self.min_model_agreement,
                'ensemble_confidence_weighting': self.ensemble_confidence_weighting
            }

            # Get ensemble performance metrics if available
            if hasattr(self.ensemble_engine, 'get_ensemble_performance_metrics'):
                try:
                    ensemble_performance = self.ensemble_engine.get_ensemble_performance_metrics()
                    ensemble_stats['ensemble_performance'] = ensemble_performance
                except:
                    pass

            base_stats.update(ensemble_stats)
        else:
            base_stats['ensemble_enabled'] = False

        return base_stats

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


def create_enhanced_strategy(config, event_manager=None, ensemble_engine=None, strategy_type=None):
    """
    Factory function to create enhanced strategy instance with optional ensemble support
    """
    if strategy_type is None:
        strategy_type = config.get('strategy', {}).get('type', 'enhanced_tft')

    # Determine if ensemble should be used
    use_ensemble = (
            ensemble_engine is not None or
            config.get('execution', {}).get('use_ensemble_predictions', False) or
            config.get('ensemble', {}).get('enabled', False)
    )

    if use_ensemble:
        return EnsembleEnhancedStrategy(config, event_manager, ensemble_engine)
    else:
        # Fall back to standard enhanced strategy
        from strategy.strategy_factory import EnhancedTFTStrategy
        return EnhancedTFTStrategy(config, event_manager)