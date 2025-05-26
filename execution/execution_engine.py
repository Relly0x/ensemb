# execution/execution_engine.py
# Updated execution engine with enhanced signal validation

import logging
import time
import os
import json
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from execution.position.position_manager import PositionManager
from execution.risk.risk_manager import RiskManager
from utils.error_handling.error_manager import ErrorManager, handle_errors
from strategy.hooks.event_system import EventManager
from utils.time_utils import TradingHoursManager


class ExecutionEngine:
    """
    Main execution engine for implementing trading decisions
    """

    def __init__(self, config, model, strategy):
        self.config = config
        self.model = model
        self.strategy = strategy
        self.logger = logging.getLogger('execution_engine')

        # Create broker instance based on configuration
        broker_type = config.get('execution', {}).get('broker', 'oanda')
        self.broker = self._create_broker(broker_type)

        # Initialize position manager
        self.position_manager = PositionManager(config, self.broker)

        # Initialize risk manager
        self.risk_manager = RiskManager(config)

        # Initialize error manager
        self.error_manager = ErrorManager(config)

        # Initialize event manager
        self.event_manager = EventManager(config)

        # Set event manager in other components
        self.strategy.event_manager = self.event_manager
        self.error_manager.set_event_manager(self.event_manager)

        # Initialize trading hours manager
        self.trading_hours_manager = TradingHoursManager(config)

        # Engine state
        self.is_running = False
        self.execution_thread = None
        self.last_data_update = None
        self.last_model_inference = None
        self.last_data = {}
        self.last_predictions = {}

        # Track trade timing for quality control
        self.last_trade_times = {}  # instrument -> datetime

        # Subscribe to events
        self._subscribe_to_events()

        self.logger.info("Execution Engine initialized")

    def _create_broker(self, broker_type):
        """Create broker instance based on type"""
        if broker_type.lower() == 'oanda':
            from execution.broker.oanda_broker import OandaBroker
            return OandaBroker(self.config)
        elif broker_type.lower() == 'mt5':
            from execution.broker.mt5_broker import MT5Broker
            return MT5Broker(self.config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")

    def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        self.event_manager.subscribe('strategy:signal_generated', self._on_signal_generated)
        self.event_manager.subscribe('strategy:hq_signal_generated', self._on_hq_signal_generated)
        self.event_manager.subscribe('trade:entry', self._on_trade_entry)
        self.event_manager.subscribe('trade:exit', self._on_trade_exit)
        self.event_manager.subscribe('system:error', self._on_system_error)

    def _on_signal_generated(self, event):
        """Handle signal generated event"""
        signal = event.get('data', {})

        if not signal.get('valid', False):
            return

        instrument = signal.get('instrument')
        direction = signal.get('signal')

        self.logger.info(f"Signal received: {direction} {instrument} with strength {signal.get('strength', 0):.2f}")

        # Process signal automatically if configured
        if self.config.get('execution', {}).get('auto_execute_signals', True):
            self._execute_signal(signal)

    def _on_hq_signal_generated(self, event):
        """Handle high-quality signal generated event"""
        signal_data = event.get('data', {})
        signal = signal_data.get('signal', {})

        if not signal.get('valid', False):
            return

        instrument = signal.get('instrument')
        direction = signal.get('signal')
        quality_grade = signal.get('quality_grade', 'B')

        self.logger.info(f"HIGH QUALITY {quality_grade}-grade signal received: {direction} {instrument}")

        # Process high-quality signal automatically
        if self.config.get('execution', {}).get('auto_execute_signals', True):
            self._execute_signal(signal)

    def _on_trade_entry(self, event):
        """Handle trade entry event"""
        trade = event.get('data', {})
        self.logger.info(
            f"Trade executed: {trade.get('direction')} {trade.get('instrument')} at {trade.get('entry_price')}")

    def _on_trade_exit(self, event):
        """Handle trade exit event"""
        trade = event.get('data', {})
        self.logger.info(
            f"Trade closed: {trade.get('direction')} {trade.get('instrument')} at {trade.get('exit_price')} with P&L {trade.get('pnl', 0):.2f}")

    def _on_system_error(self, event):
        """Handle system error event"""
        error = event.get('data', {})
        self.logger.error(f"System error: {error.get('message')} in {error.get('component')}")

    @handle_errors(error_type='execution_error', max_retries=3)
    def _execute_signal(self, signal):
        """
        Execute a trading signal with enhanced validation and quality controls
        """

        # FIRST CHECK: Trading hours (most important)
        if not self.trading_hours_manager.is_trading_time():
            self.logger.info(f"Signal rejected: Outside trading hours")
            return False, "Outside trading hours"

        # SECOND CHECK: Extract signal details
        instrument = signal.get('instrument')
        direction = signal.get('signal')
        current_price = signal.get('current_price')
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        signal_strength = signal.get('strength', 0)
        quality_grade = signal.get('quality_grade', 'C')

        # THIRD CHECK: Validate signal parameters
        if not all([instrument, direction, current_price, stop_loss, take_profit]):
            return False, "Invalid signal parameters"

        # FOURTH CHECK: Signal quality threshold
        min_signal_strength = self.config.get('strategy', {}).get('min_signal_strength', 0.7)
        if signal_strength < min_signal_strength:
            self.logger.info(f"Signal rejected: Low quality (strength={signal_strength:.2f} < {min_signal_strength})")
            return False, f"Signal quality too low ({signal_strength:.2f})"

        # FIFTH CHECK: Quality grade filter
        min_quality_grade = self.config.get('execution', {}).get('min_quality_grade', 'B')
        if quality_grade < min_quality_grade:
            self.logger.info(f"Signal rejected: Grade {quality_grade} below minimum {min_quality_grade}")
            return False, f"Signal grade too low ({quality_grade})"

        # SIXTH CHECK: Check if we're already in a position for this instrument
        current_positions = self.position_manager.get_position_summary()
        existing_position = next((p for p in current_positions.get('positions', [])
                                  if p['instrument'] == instrument), None)

        if existing_position:
            # Don't take new position if we already have one
            self.logger.info(f"Signal rejected: Already have position in {instrument}")
            return False, f"Already have position in {instrument}"

        # SEVENTH CHECK: Validate with max open positions
        max_positions = self.config.get('execution', {}).get('max_open_positions', 2)
        if len(current_positions.get('positions', [])) >= max_positions:
            return False, f"Maximum open positions reached ({max_positions})"

        # EIGHTH CHECK: Wait time between trades for same instrument
        min_wait_minutes = self.config.get('strategy', {}).get('min_wait_between_trades', 60)
        if instrument in self.last_trade_times:
            time_since_last = (datetime.now() - self.last_trade_times[instrument]).total_seconds() / 60
            if time_since_last < min_wait_minutes:
                self.logger.info(
                    f"Signal rejected: Too soon after last trade ({time_since_last:.1f} < {min_wait_minutes} minutes)")
                return False, f"Too soon after last trade"

        # NINTH CHECK: Daily trade limit
        max_daily_trades = self.config.get('execution', {}).get('max_daily_trades', 3)
        if hasattr(self.strategy, 'daily_trade_count') and self.strategy.daily_trade_count >= max_daily_trades:
            self.logger.info(
                f"Signal rejected: Daily trade limit reached ({self.strategy.daily_trade_count}/{max_daily_trades})")
            return False, f"Daily trade limit reached"

        # TENTH CHECK: Strategy validation
        if hasattr(self.strategy, 'validate_trade'):
            if not self.strategy.validate_trade(instrument, direction, current_price):
                self.logger.info(f"Signal rejected: Strategy validation failed")
                return False, "Strategy validation failed"

        # Calculate position size with risk management
        position_size = self.risk_manager.calculate_position_size(
            instrument, current_price, stop_loss, direction
        )

        # Execute the trade
        result, position_or_error = self.position_manager.open_position(
            instrument, direction, current_price, position_size, stop_loss, take_profit
        )

        if result:
            # Track last trade time
            self.last_trade_times[instrument] = datetime.now()

            # Log successful execution with quality details
            self.logger.info(
                f"🟢 HIGH QUALITY {quality_grade}-grade position opened: "
                f"{direction.upper()} {instrument} "
                f"size={position_size:.2f} entry={current_price:.5f} "
                f"sl={stop_loss:.5f} tp={take_profit:.5f} "
                f"strength={signal_strength:.1%}"
            )

            # Emit enhanced trade entry event
            self.event_manager.emit('trade:hq_entry', {
                'instrument': instrument,
                'direction': direction,
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_strength': signal_strength,
                'quality_grade': quality_grade,
                'timestamp': datetime.now().isoformat()
            }, source='execution_engine')

            return True, f"High quality {quality_grade}-grade position opened successfully"
        else:
            self.logger.error(f"Failed to open position: {position_or_error}")
            return False, f"Failed to open position: {position_or_error}"

    def start_live_trading(self):
        """Start live trading execution"""
        if self.is_running:
            self.logger.warning("Execution engine already running")
            return

        self.is_running = True

        # Emit startup event
        self.event_manager.emit('system:startup', {
            'time': datetime.now().isoformat(),
            'mode': 'live',
            'trading_hours': self.config.get('trading_hours')
        }, source='execution_engine')

        # Start execution thread
        self.execution_thread = threading.Thread(target=self._live_trading_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()

        self.logger.info("Live trading started with enhanced quality controls")

    def _live_trading_loop(self):
        """Main execution loop for live trading"""
        while self.is_running:
            try:
                # Check if it's trading time
                if not self.trading_hours_manager.is_trading_time():
                    # Sleep until next trading session
                    next_session = self.trading_hours_manager.next_trading_session()
                    if next_session:
                        now = datetime.now(next_session.tzinfo)
                        wait_seconds = (next_session - now).total_seconds()

                        if wait_seconds > 0:
                            self.logger.info(
                                f"Outside trading hours. Next session starts in {wait_seconds:.1f} seconds")
                            # Sleep for at most 5 minutes at a time
                            time.sleep(min(wait_seconds, 300))
                            continue

                # Collect latest market data
                market_data = self._collect_market_data()

                if not market_data:
                    self.logger.warning("No market data available")
                    time.sleep(10)
                    continue

                # Save data for reference
                self.last_data = market_data
                self.last_data_update = datetime.now()

                # Update strategy with new data
                self.strategy.update_data(market_data)

                # Run model inference
                predictions = self._run_model_inference(market_data)

                if not predictions:
                    self.logger.warning("No predictions available")
                    time.sleep(10)
                    continue

                # Save predictions for reference
                self.last_predictions = predictions
                self.last_model_inference = datetime.now()

                # Generate trading signals with enhanced quality controls
                signals = self.strategy.generate_signals(predictions, market_data)

                # Process signals - they are already filtered by quality
                valid_signals = 0
                for instrument, signal in signals.items():
                    if signal.get('valid', False):
                        valid_signals += 1
                        # Signal is automatically executed via event system

                if valid_signals > 0:
                    self.logger.info(f"Generated {valid_signals} high-quality signals")

                # Update positions with latest prices
                current_prices = self._get_current_prices()
                self.position_manager.update_positions(current_prices)

                # Sleep between iterations
                update_interval = self.config.get('execution', {}).get('update_interval', 60)
                time.sleep(update_interval)

            except Exception as e:
                # Handle error and continue
                self.error_manager.handle_error(e, 'execution_error', {
                    'component': 'execution_loop',
                    'time': datetime.now().isoformat()
                })
                time.sleep(10)  # Sleep before retrying

    def stop_live_trading(self):
        """Stop live trading execution"""
        if not self.is_running:
            self.logger.warning("Execution engine not running")
            return

        self.is_running = False

        # Wait for execution thread to finish
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=10)

        # Emit shutdown event
        self.event_manager.emit('system:shutdown', {
            'time': datetime.now().isoformat(),
            'mode': 'live'
        }, source='execution_engine')

        self.logger.info("Live trading stopped")

    def _collect_market_data(self):
        """Collect latest market data for all instruments"""
        instruments = self.config['data']['instruments']
        timeframes = [
            self.config['data']['timeframes']['high'],
            self.config['data']['timeframes']['low']
        ]

        result = {}

        try:
            # Use broker to get market data
            for instrument in instruments:
                result[instrument] = {}

                for timeframe in timeframes:
                    data = self.broker.get_candles(instrument, timeframe, 200)

                    if data is not None and not data.empty:
                        result[instrument][timeframe] = data

            return result
        except Exception as e:
            self.error_manager.handle_error(e, 'data_error', {
                'component': 'market_data_collection'
            })
            return {}

    def _run_model_inference(self, market_data):
        """Run model inference on latest market data"""
        instruments = self.config['data']['instruments']
        predictions = {}

        try:
            # Process each instrument
            for instrument in instruments:
                # Skip if missing data
                if instrument not in market_data:
                    continue

                # Prepare model input
                model_input = self._prepare_model_input(market_data, instrument)

                if model_input is None:
                    continue

                # Run inference
                if hasattr(self.model, 'forward'):
                    # PyTorch model
                    import torch
                    with torch.no_grad():
                        prediction = self.model(model_input)
                elif hasattr(self.model, 'run'):
                    # ONNX model
                    input_names = [inp.name for inp in self.model.get_inputs()]
                    outputs = self.model.run(None, {name: model_input[name] for name in input_names})
                    prediction = outputs[0]  # Assuming first output is the prediction
                else:
                    self.logger.error("Unknown model type, cannot run inference")
                    return {}

                # Store prediction
                predictions[instrument] = prediction

            return predictions
        except Exception as e:
            self.error_manager.handle_error(e, 'model_error', {
                'component': 'model_inference'
            })
            return {}

    def _prepare_model_input(self, market_data, instrument):
        """Prepare input for model inference"""
        try:
            # Use high timeframe data for features
            high_tf = self.config['data']['timeframes']['high']

            if instrument not in market_data or high_tf not in market_data[instrument]:
                return None

            # Get data
            data = market_data[instrument][high_tf].copy()

            if len(data) < 50:  # Need enough data for feature calculation
                self.logger.warning(f"Insufficient data for {instrument}")
                return None

            # Create features
            try:
                # Try to use feature creator if available
                from models.feature_engineering.feature_creator import FeatureCreator
                feature_creator = FeatureCreator(self.config)
                features = feature_creator.create_features(data)
            except ImportError:
                # Fallback to basic features
                self.logger.warning("FeatureCreator not available, using basic features")
                features = self._create_basic_features(data)

            # Check if it's a PyTorch or ONNX model
            if hasattr(self.model, 'forward'):
                # PyTorch model - create tensors
                import torch

                # Get sequence lengths from config
                past_seq_len = self.config['model'].get('past_sequence_length', 50)

                # Use recent data
                recent_data = features.iloc[-past_seq_len:].copy() if len(features) >= past_seq_len else features.copy()

                # Create input tensors
                # Note: In a real implementation, you'd apply the same scaling as during training

                # Create model input dictionary
                model_input = {
                    'past': torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0),
                    'static': torch.zeros((1, 1), dtype=torch.float32),  # Dummy static input
                    'future': torch.zeros((1, self.config['model'].get('forecast_horizon', 12), 1),
                                          dtype=torch.float32)  # Dummy future input
                }

                return model_input
            else:
                # ONNX model - create numpy arrays

                # Get sequence lengths from config
                past_seq_len = self.config['model'].get('past_sequence_length', 50)

                # Use recent data
                recent_data = features.iloc[-past_seq_len:].copy() if len(features) >= past_seq_len else features.copy()

                # Create input arrays
                model_input = {
                    'past': recent_data.values.astype(np.float32).reshape(1, -1, recent_data.shape[1]),
                    'static': np.zeros((1, 1), dtype=np.float32),  # Dummy static input
                    'future': np.zeros((1, self.config['model'].get('forecast_horizon', 12), 1),
                                       dtype=np.float32)  # Dummy future input
                }

                return model_input

        except Exception as e:
            self.error_manager.handle_error(e, 'feature_error', {
                'component': 'feature_creation',
                'instrument': instrument
            })
            return None

    def _create_basic_features(self, data):
        """Create basic features when feature creator is not available"""
        df = data.copy()

        # Technical indicators
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

        # Price changes
        df['close_change'] = df['close'].pct_change()
        df['high_change'] = df['high'].pct_change()
        df['low_change'] = df['low'].pct_change()

        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()

        # Moving average differences
        df['ma_diff'] = df['sma_10'] - df['sma_20']

        # Fill NAs with 0
        df = df.fillna(0)

        # Remove date index for model input
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)

        return df

    def _get_current_prices(self):
        """Get current prices for all instruments"""
        instruments = self.config['data']['instruments']
        prices = {}

        try:
            # Get prices from broker
            return self.broker.get_current_prices()
        except Exception as e:
            self.error_manager.handle_error(e, 'market_error', {
                'component': 'price_fetch'
            })

            # Fallback to last known prices from market data
            for instrument in instruments:
                if (instrument in self.last_data and
                        self.config['data']['timeframes']['low'] in self.last_data[instrument]):
                    data = self.last_data[instrument][self.config['data']['timeframes']['low']]
                    if not data.empty:
                        prices[instrument] = data['close'].iloc[-1]

            return prices

    def get_execution_status(self):
        """Get current execution engine status"""
        return {
            'is_running': self.is_running,
            'last_data_update': self.last_data_update.isoformat() if self.last_data_update else None,
            'last_model_inference': self.last_model_inference.isoformat() if self.last_model_inference else None,
            'is_trading_time': self.trading_hours_manager.is_trading_time(),
            'next_trading_session': self.trading_hours_manager.next_trading_session(),
            'active_positions': len(self.position_manager.get_position_summary().get('positions', [])),
            'max_positions': self.config.get('execution', {}).get('max_open_positions', 2),
            'daily_trade_count': getattr(self.strategy, 'daily_trade_count', 0),
            'max_daily_trades': self.config.get('execution', {}).get('max_daily_trades', 3),
            'last_trade_times': {k: v.isoformat() for k, v in self.last_trade_times.items()}
        }

    def backtest(self, start_date=None, end_date=None, initial_balance=10000.0):
        """
        Run a backtest on historical data

        Parameters:
        - start_date: Start date for backtest (str YYYY-MM-DD)
        - end_date: End date for backtest (str YYYY-MM-DD)
        - initial_balance: Initial account balance

        Returns:
        - Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Set up backtest environment
        backtest_config = self.config['backtest'] if 'backtest' in self.config else {}

        # Use specified dates or from config
        if start_date is None:
            start_date = backtest_config.get('start_date')
        if end_date is None:
            end_date = backtest_config.get('end_date')

        # Collect historical data
        historical_data = self._collect_historical_data(start_date, end_date)

        if not historical_data:
            self.logger.error("Failed to collect historical data for backtest")
            return {"success": False, "error": "No historical data available"}

        # Initialize account state
        account = {
            'balance': initial_balance,
            'equity': initial_balance,
            'positions': {},
            'trades': [],
            'equity_curve': [(start_date, initial_balance)]
        }

        # Process each time step
        time_steps = self._get_backtest_time_steps(historical_data)

        self.logger.info(f"Running backtest with {len(time_steps)} time steps")

        for current_time, data_slice in time_steps:
            # Skip if outside trading hours (if configured)
            if backtest_config.get('respect_trading_hours', True):
                dt = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
                if not self._is_trading_time_backtest(dt):
                    continue

            # Update strategy with current data
            self.strategy.update_data(data_slice)

            # Run model inference
            predictions = self._run_model_inference(data_slice)

            if not predictions:
                continue

            # Generate signals
            signals = self.strategy.generate_signals(predictions, data_slice)

            # Process signals
            for instrument, signal in signals.items():
                if signal.get('valid', False):
                    # Process trade entry
                    self._process_backtest_entry(account, signal, current_time)

            # Process open positions for exits
            self._process_backtest_exits(account, data_slice, current_time)

            # Update equity curve
            account['equity'] = account['balance'] + self._calculate_unrealized_pnl(account, data_slice)
            account['equity_curve'].append((current_time, account['equity']))

        # Close any remaining positions
        final_data_slice = time_steps[-1][1] if time_steps else None
        if final_data_slice:
            self._close_all_backtest_positions(account, final_data_slice, time_steps[-1][0])

        # Calculate performance metrics
        results = self._calculate_backtest_metrics(account)

        # Save backtest results if configured
        if backtest_config.get('save_results', True):
            self._save_backtest_results(results, account)

        self.logger.info(f"Backtest completed with {len(account['trades'])} trades")

        return results

    def _collect_historical_data(self, start_date, end_date):
        """Collect historical data for backtest"""
        instruments = self.config['data']['instruments']
        timeframes = [
            self.config['data']['timeframes']['high'],
            self.config['data']['timeframes']['low']
        ]

        result = {}

        try:
            # Determine data source based on configuration
            if self.config['data']['source'] == 'oanda':
                # Use OandaDataCollector
                from data.collectors.oanda_collector import OandaDataCollector
                collector = OandaDataCollector(self.config)
                historical_data = collector.collect_historical_data(
                    instruments,
                    timeframes,
                    start_date,
                    end_date
                )
                return historical_data
            elif self.config['data']['source'] == 'mt5':
                # Use MT5DataCollector
                from data.collectors.mt5_collector import MT5DataCollector
                collector = MT5DataCollector(self.config)
                historical_data = collector.collect_historical_data(
                    instruments,
                    timeframes,
                    start_date,
                    end_date
                )
                return historical_data
            else:
                # Fallback to broker
                for instrument in instruments:
                    result[instrument] = {}

                    for timeframe in timeframes:
                        data = self.broker.get_historical_candles(
                            instrument,
                            timeframe,
                            start_date,
                            end_date
                        )

                        if data is not None and not data.empty:
                            result[instrument][timeframe] = data

                return result
        except Exception as e:
            self.error_manager.handle_error(e, 'data_error', {
                'component': 'historical_data_collection'
            })
            return {}

    def _get_backtest_time_steps(self, historical_data):
        """Generate time steps for backtest processing"""
        time_steps = []

        # Determine common time points across all data
        common_times = set()
        first_iteration = True

        low_tf = self.config['data']['timeframes']['low']

        for instrument, timeframes in historical_data.items():
            if low_tf in timeframes:
                df = timeframes[low_tf]

                # Format datetime index to string
                if isinstance(df.index, pd.DatetimeIndex):
                    times = set(df.index.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    times = set([str(idx) for idx in df.index])

                if first_iteration:
                    common_times = times
                    first_iteration = False
                else:
                    common_times = common_times.intersection(times)

        # Sort times chronologically
        common_times = sorted(list(common_times))

        # Create time slices
        for time_str in common_times:
            data_slice = {}

            for instrument, timeframes in historical_data.items():
                data_slice[instrument] = {}

                for tf_name, df in timeframes.items():
                    # Filter data up to current time
                    if isinstance(df.index, pd.DatetimeIndex):
                        current_time = pd.to_datetime(time_str)
                        mask = df.index <= current_time
                    else:
                        current_time = time_str
                        mask = df.index <= current_time

                    data_slice[instrument][tf_name] = df[mask].copy()

            time_steps.append((time_str, data_slice))

        return time_steps

    def _is_trading_time_backtest(self, dt):
        """Check if given datetime is within trading hours"""
        if not isinstance(dt, datetime):
            try:
                dt = pd.to_datetime(dt)
            except:
                return True  # Default to allow trading if parsing fails

        # Get day of week
        day = dt.strftime('%A')

        # Check each trading session
        for session in self.config['trading_hours']['sessions']:
            # Check if day is in trading days
            if day in session['days']:
                # Parse session times
                start_time = datetime.strptime(session['start'], '%H:%M').time()
                end_time = datetime.strptime(session['end'], '%H:%M').time()

                # Check if current time is within session
                if start_time <= dt.time() <= end_time:
                    return True

        return False

    def _process_backtest_entry(self, account, signal, current_time):
        """Process a trade entry in backtest"""
        instrument = signal['instrument']
        direction = signal['signal']
        current_price = signal['current_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']

        # Skip if already in a position for this instrument
        if instrument in account['positions']:
            return

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            instrument,
            current_price,
            stop_loss,
            direction
        )

        # Adjust for backtest balance
        account_balance_ratio = account['balance'] / self.config['execution'].get('initial_balance', 10000.0)
        position_size *= account_balance_ratio

        # Check max positions
        if len(account['positions']) >= self.config['execution']['max_open_positions']:
            return

        # Create trade
        trade = {
            'instrument': instrument,
            'direction': direction,
            'entry_price': current_price,
            'entry_time': current_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'exit_price': None,
            'exit_time': None,
            'pnl': 0.0,
            'status': 'open',
            'quality_grade': signal.get('quality_grade', 'B'),
            'signal_strength': signal.get('strength', 0)
        }

        # Store in account
        account['positions'][instrument] = trade

        # Log trade
        self.logger.info(
            f"Backtest entry: {direction} {instrument} at {current_price} (Grade: {trade['quality_grade']}, Strength: {trade['signal_strength']:.1%})")

    def _process_backtest_exits(self, account, data_slice, current_time):
        """Process exits for open positions in backtest"""
        # Process each open position
        for instrument, position in list(account['positions'].items()):
            # Skip if instrument data not available
            if (instrument not in data_slice or
                    self.config['data']['timeframes']['low'] not in data_slice[instrument]):
                continue

            # Get latest price data
            df = data_slice[instrument][self.config['data']['timeframes']['low']]
            if df.empty:
                continue

            latest_row = df.iloc[-1]
            current_price = latest_row['close']

            # Check for stop loss hit
            stop_hit = False
            tp_hit = False

            if position['direction'] == 'buy':
                if latest_row['low'] <= position['stop_loss']:
                    # Stop loss hit
                    stop_hit = True
                    exit_price = position['stop_loss']
                elif latest_row['high'] >= position['take_profit']:
                    # Take profit hit
                    tp_hit = True
                    exit_price = position['take_profit']
            else:  # sell
                if latest_row['high'] >= position['stop_loss']:
                    # Stop loss hit
                    stop_hit = True
                    exit_price = position['stop_loss']
                elif latest_row['low'] <= position['take_profit']:
                    # Take profit hit
                    tp_hit = True
                    exit_price = position['take_profit']

            # Execute exit if conditions met
            if stop_hit or tp_hit:
                # Calculate P&L
                if position['direction'] == 'buy':
                    pnl = (exit_price - position['entry_price']) * position['position_size']
                else:  # sell
                    pnl = (position['entry_price'] - exit_price) * position['position_size']

                # Update position
                position['exit_price'] = exit_price
                position['exit_time'] = current_time
                position['pnl'] = pnl
                position['status'] = 'closed'

                # Add to trade history
                account['trades'].append(position)

                # Update account balance
                account['balance'] += pnl

                # Remove from open positions
                del account['positions'][instrument]

                # Log trade
                reason = "Stop loss" if stop_hit else "Take profit"
                self.logger.info(
                    f"Backtest exit ({reason}): {position['direction']} {instrument} at {exit_price} (P&L: {pnl:.2f}, Grade: {position['quality_grade']})")

    def _close_all_backtest_positions(self, account, data_slice, current_time):
        """Close all open positions at the end of backtest"""
        for instrument, position in list(account['positions'].items()):
            # Get latest price
            if (instrument in data_slice and
                    self.config['data']['timeframes']['low'] in data_slice[instrument]):
                df = data_slice[instrument][self.config['data']['timeframes']['low']]
                if not df.empty:
                    current_price = df['close'].iloc[-1]

                    # Calculate P&L
                    if position['direction'] == 'buy':
                        pnl = (current_price - position['entry_price']) * position['position_size']
                    else:  # sell
                        pnl = (position['entry_price'] - current_price) * position['position_size']

                    # Update position
                    position['exit_price'] = current_price
                    position['exit_time'] = current_time
                    position['pnl'] = pnl
                    position['status'] = 'closed'

                    # Add to trade history
                    account['trades'].append(position)

                    # Update account balance
                    account['balance'] += pnl

                    # Remove from open positions
                    del account['positions'][instrument]

                    # Log trade
                    self.logger.info(
                        f"Backtest final exit: {position['direction']} {instrument} at {current_price} (P&L: {pnl:.2f}, Grade: {position['quality_grade']})")

    def _calculate_unrealized_pnl(self, account, data_slice):
        """Calculate unrealized P&L for open positions"""
        unrealized_pnl = 0.0

        for instrument, position in account['positions'].items():
            # Get current price
            if (instrument in data_slice and
                    self.config['data']['timeframes']['low'] in data_slice[instrument]):
                df = data_slice[instrument][self.config['data']['timeframes']['low']]
                if not df.empty:
                    current_price = df['close'].iloc[-1]

                    # Calculate unrealized P&L
                    if position['direction'] == 'buy':
                        position_pnl = (current_price - position['entry_price']) * position['position_size']
                    else:  # sell
                        position_pnl = (position['entry_price'] - current_price) * position['position_size']

                    unrealized_pnl += position_pnl

        return unrealized_pnl

    def _calculate_backtest_metrics(self, account):
        """Calculate performance metrics from backtest results"""
        trades = account['trades']

        if not trades:
            return {
                'success': True,
                'total_trades': 0,
                'net_profit': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }

        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] <= 0)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Quality metrics
        a_grade_trades = sum(1 for t in trades if t.get('quality_grade') == 'A')
        b_grade_trades = sum(1 for t in trades if t.get('quality_grade') == 'B')
        avg_signal_strength = sum(t.get('signal_strength', 0) for t in trades) / total_trades if total_trades > 0 else 0

        # Calculate drawdown
        equity_curve = account['equity_curve']
        max_drawdown = 0.0

        if len(equity_curve) > 1:
            peak = equity_curve[0][1]

            for _, equity in equity_curve[1:]:
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)

        # Return enhanced metrics
        return {
            'success': True,
            'initial_balance': account['equity_curve'][0][1],
            'final_balance': account['equity'],
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'net_profit': account['equity'] - account['equity_curve'][0][1],
            'net_profit_percent': (account['equity'] / account['equity_curve'][0][1] - 1) * 100,
            'equity_curve': account['equity_curve'],
            'quality_metrics': {
                'a_grade_trades': a_grade_trades,
                'b_grade_trades': b_grade_trades,
                'avg_signal_strength': avg_signal_strength,
                'quality_distribution': {
                    'A': a_grade_trades / total_trades if total_trades > 0 else 0,
                    'B': b_grade_trades / total_trades if total_trades > 0 else 0
                }
            }
        }

    def _save_backtest_results(self, results, account):
        """Save backtest results to file"""
        try:
            # Create output directory
            output_dir = self.config.get('backtest', {}).get('output_dir', 'backtest_results')
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"enhanced_backtest_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            # Prepare data for serialization
            output_data = {
                'metrics': {k: v for k, v in results.items() if k != 'equity_curve'},
                'trades': account['trades'],
                'equity_curve': account['equity_curve'],
                'config_snapshot': self.config
            }

            # Write to file
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)

            self.logger.info(f"Enhanced backtest results saved to {filepath}")

            # Generate plots if matplotlib is available
            try:
                self._generate_backtest_plots(results, account, output_dir, timestamp)
            except Exception as e:
                self.logger.warning(f"Failed to generate backtest plots: {e}")

        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")

    def _generate_backtest_plots(self, results, account, output_dir, timestamp):
        """Generate plots for backtest results"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # Prepare equity curve data
            equity_dates = [pd.to_datetime(time) for time, _ in account['equity_curve']]
            equity_values = [value for _, value in account['equity_curve']]

            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(equity_dates, equity_values, 'b-', linewidth=1)
            plt.title('Enhanced Strategy Equity Curve')
            plt.grid(True, alpha=0.3)
            plt.xlabel('Date')
            plt.ylabel('Equity')

            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(equity_dates) // 10)))
            plt.xticks(rotation=45)

            # Add annotations
            plt.annotate(f"Start: ${equity_values[0]:.2f}",
                         xy=(equity_dates[0], equity_values[0]),
                         xytext=(10, 10),
                         textcoords='offset points')

            plt.annotate(f"End: ${equity_values[-1]:.2f}",
                         xy=(equity_dates[-1], equity_values[-1]),
                         xytext=(-70, 10),
                         textcoords='offset points')

            # Add quality metrics text
            quality_metrics = results.get('quality_metrics', {})
            plt.text(0.02, 0.98,
                     f"A-grade: {quality_metrics.get('a_grade_trades', 0)}\n"
                     f"B-grade: {quality_metrics.get('b_grade_trades', 0)}\n"
                     f"Avg Strength: {quality_metrics.get('avg_signal_strength', 0):.1%}",
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # Save plot
            equity_plot_path = os.path.join(output_dir, f"enhanced_equity_curve_{timestamp}.png")
            plt.savefig(equity_plot_path)
            plt.close()

            self.logger.info(f"Enhanced backtest plots generated in {output_dir}")

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Error generating backtest plots: {e}")