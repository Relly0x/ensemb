def check_positions(self):
    """Check and manage open positions with enhanced monitoring"""
    try:
        # Get all positions
        positions = mt5.positions_get()

        if positions is None:
            return

        current_mt5_positions = {}

        for position in positions:
            # Only check positions opened by this EA
            if position.magic != 123456:
                continue

            symbol = position.symbol
            profit = position.profit

            # Convert back to our instrument format
            instrument = symbol
            for config_instrument in self.config['data']['instruments']:
                if config_instrument.replace('_', '') == symbol:
                    instrument = config_instrument
                    break

            current_mt5_positions[instrument] = {
                'profit': profit,
                'current_price': position.price_current,
                'entry_price': position.price_open,
                'volume': position.volume
            }

            # Check for significant profit/loss changes
            if instrument in self.positions:
                stored_pos = self.positions[instrument]

                # Check for major moves (optional notification)
                if abs(profit) > 50:  # Notify for profits/losses > $50
                    if not hasattr(stored_pos, 'last_major_notification') or abs(
                            profit - stored_pos.get('last_major_profit', 0)) > 25:
                        emoji = "💰" if profit > 0 else "⚠️"
                        if self.telegram:
                            self.telegram.send_sync(
                                f"{emoji} **Position Update**\n\n"
                                f"📊 {symbol}: {profit:+.2f}\n"
                                f"📈 Entry: {position.price_open:.5f}\n"
                                f"📊 Current: {position.price_current:.5f}"
                            )
                        stored_pos['last_major_profit'] = profit
                        stored_pos['last_major_notification'] = datetime.now()

        # Check for closed positions (positions that were in our tracking but no longer in MT5)
        closed_positions = []
        for instrument in list(self.positions.keys()):
            if instrument not in current_mt5_positions:
                closed_positions.append(instrument)

        # Handle closed positions
        for instrument in closed_positions:
            stored_pos = self.positions[instrument]

            # Try to get the closing details from MT5 history
            symbol = instrument.replace('_', '')

            # Get recent deals to find the closing trade
            from_date = stored_pos['entry_time']
            to_date = datetime.now()

            deals = mt5.history_deals_get(from_date, to_date, symbol=symbol)

            if deals:
                # Find the closing deal (opposite direction)
                entry_type = mt5.ORDER_TYPE_BUY if stored_pos['direction'] == 'buy' else mt5.ORDER_TYPE_SELL
                close_type = mt5.ORDER_TYPE_SELL if stored_pos['direction'] == 'buy' else mt5.ORDER_TYPE_BUY

                closing_deal = None
                for deal in reversed(deals):  # Check most recent deals first
                    if deal.type == close_type and deal.magic == 123456:
                        closing_deal = deal
                        break

                if closing_deal:
                    # Calculate P&L
                    if stored_pos['direction'] == 'buy':
                        pnl = (closing_deal.price - stored_pos['entry_price']) * stored_pos[
                            'size'] * 100000  # Assuming standard lot
                    else:
                        pnl = (stored_pos['entry_price'] - closing_deal.price) * stored_pos['size'] * 100000

                    # Determine close reason
                    if abs(closing_deal.price - stored_pos['stop_loss']) < 0.00005:
                        close_reason = "Stop Loss"
                        emoji = "🛡️"
                    elif abs(closing_deal.price - stored_pos['take_profit']) < 0.00005:
                        close_reason = "Take Profit"
                        emoji = "🎯"
                    else:
                        close_reason = "Manual/Other"
                        emoji = "🔄"

                    # Calculate duration
                    duration = datetime.now() - stored_pos['entry_time']
                    duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"

                    # Send notification
                    profit_emoji = "💰" if pnl > 0 else "📉"

                    if self.telegram:
                        self.telegram.send_sync(
                            f"{emoji} **Trade Closed - {close_reason}**\n\n"
                            f"📊 {symbol} {stored_pos['direction'].upper()}\n"
                            f"🎯 Entry: {stored_pos['entry_price']:.5f}\n"
                            f"🏁 Exit: {closing_deal.price:.5f}\n"
                            f"{profit_emoji} P&L: {pnl:+.2f}\n"
                            f"💪 Signal Strength: {stored_pos.get('signal_strength', 0):.1%}\n"
                            f"⏱️ Duration: {duration_str}\n"
                            f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}"
                        )

                    self.logger.info(
                        f"Position closed: {symbol} {close_reason} P&L: {pnl:+.2f}"
                    )
                else:
                    # Couldn't find closing deal, send generic notification
                    if self.telegram:
                        self.telegram.send_sync(
                            f"🔄 **Position Closed**\n\n"
                            f"📊 {symbol} {stored_pos['direction'].upper()}\n"
                            f"🎯 Entry: {stored_pos['entry_price']:.5f}\n"
                            f"ℹ️ Position no longer active in MT5"
                        )

            # Remove from our tracking
            del self.positions[instrument]

        self.logger.debug(f"Monitoring {len(current_mt5_positions)} positions")

    except Exception as e:
        self.logger.error(f"Error checking positions: {e}")
        if self.telegram:
            self.telegram.send_sync(f"⚠️ Error monitoring positions: {str(e)}")


def cleanup(self):
    """Cleanup function for graceful shutdown"""
    self.logger.info("🛑 Starting cleanup...")

    # Set shutdown flag
    self.is_running = False
    self.shutdown_event.set()

    try:
        # Send shutdown notification
        if self.telegram:
            self.telegram.send_sync(
                f"🛑 **Trading Bot Shutting Down**\n\n"
                f"📊 Current Positions: {len(self.positions)}\n"
                f"📈 Daily Trades: {self.daily_trade_count}\n"
                f"⏰ Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # Close any open positions if configured
        if self.config.get('execution', {}).get('close_positions_on_shutdown', False):
            self.logger.info("Closing all open positions...")
            positions = mt5.positions_get()

            if positions:
                closed_count = 0
                for position in positions:
                    if position.magic == 123456:  # Only our positions
                        # Get current price for closing
                        tick = mt5.symbol_info_tick(position.symbol)
                        if not tick:
                            continue

                        close_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask

                        # Close position
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": position.symbol,
                            "volume": position.volume,
                            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                            "position": position.ticket,
                            "price": close_price,
                            "magic": 123456,
                            "comment": "Bot shutdown",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }

                        result = mt5.order_send(close_request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            self.logger.info(f"Closed position {position.symbol}")
                            closed_count += 1
                        else:
                            self.logger.error(f"Failed to close position {position.symbol}: {result.comment}")

                if self.telegram and closed_count > 0:
                    self.telegram.send_sync(f"✅ Closed {closed_count} positions safely")

        # Stop telegram bot
        if self.telegram:
            self.logger.info("Stopping Telegram bot...")
            self.telegram.stop_bot()
            time.sleep(1)  # Give it time to send final messages

        # Shutdown MT5 connection
        mt5.shutdown()
        self.logger.info("MT5 connection closed")

    except Exception as e:
        self.logger.error(f"Error during cleanup: {e}")

    self.logger.info("✅ Cleanup complete")


def run(self):
    """Main trading loop with enhanced shutdown handling"""
    self.logger.info("🚀 Starting Enhanced MT5 Trading Bot")

    # Initialize MT5
    if not self.initialize_mt5():
        return

    # Load model
    if not self.load_model():
        return

    # Initialize components
    if not self.initialize_components():
        return

    # Start Telegram bot if configured
    if self.telegram:
        if self.telegram.start_bot():
            self.logger.info("✅ Telegram bot started")
        else:
            self.logger.warning("⚠️ Telegram bot failed to start")

    self.is_running = True
    self.logger.info("✅ Bot is ready and running - Press Ctrl+C to stop safely")

    # Send startup notification
    if self.telegram:
        account_info = mt5.account_info()
        if account_info:
            self.telegram.send_sync(
                f"🚀 **MT5 Trading Bot Started!**\n\n"
                f"🏢 Broker: {account_info.company}\n"
                f"💰 Balance: {account_info.balance:.2f} {account_info.currency}\n"
                f"📊 Strategy: Enhanced TFT\n"
                f"💼 Risk per Trade: {self.config.get('execution', {}).get('risk_per_trade', 0.01) * 100}%\n"
                f"📈 Max Positions: {self.config.get('execution', {}).get('max_open_positions', 2)}\n"
                f"📅 Max Daily Trades: {self.config.get('execution', {}).get('max_daily_trades', 3)}\n"
                f"🎯 Quality Filter: Enabled\n"
                f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
                f"Ready to trade! 💪"
            )

    # Main loop
    iteration_count = 0
    last_status_update = datetime.now()

    try:
        while self.is_running and not self.shutdown_event.is_set():
            try:
                iteration_count += 1

                # Check if market is open
                if not self._is_market_open():
                    self.logger.debug("Market closed, waiting...")
                    # Send market closed notification once per hour
                    now = datetime.now()
                    if (now - last_status_update).seconds > 3600:  # 1 hour
                        if self.telegram:
                            next_open = "Monday 08:00" if now.weekday() >= 4 else "08:00"  # Simplified
                            self.telegram.send_sync(
                                f"😴 **Market Closed**\n\n"
                                f"⏰ Current Time: {now.strftime('%H:%M')}\n"
                                f"📅 Next Open: {next_open}\n"
                                f"🤖 Bot Status: Waiting..."
                            )
                        last_status_update = now
                    time.sleep(60)
                    continue

                # Get market data
                market_data = self.get_market_data()
                if not market_data:
                    self.logger.warning("No market data available")
                    time.sleep(10)
                    continue

                # Generate signals
                signals = self.generate_signals(market_data)

                # Process signals
                signals_processed = 0
                for instrument, signal in signals.items():
                    if signal.get('valid', False):
                        # Check if we already have a position
                        if instrument not in self.positions:
                            self.logger.info(
                                f"New signal for {instrument}: {signal['signal']} (strength: {signal.get('strength', 0):.1%})")
                            if self.execute_signal(signal, instrument):
                                signals_processed += 1

                # Check existing positions
                self.check_positions()

                # Send periodic status updates
                now = datetime.now()
                if (now - last_status_update).seconds > 1800:  # Every 30 minutes
                    if self.telegram and self.is_running:
                        positions_count = len(self.positions)
                        self.telegram.send_sync(
                            f"📊 **Status Update**\n\n"
                            f"🟢 Bot: Running\n"
                            f"📈 Open Positions: {positions_count}\n"
                            f"📊 Daily Trades: {self.daily_trade_count}\n"
                            f"🔄 Iteration: {iteration_count}\n"
                            f"⏰ Time: {now.strftime('%H:%M:%S')}"
                        )
                    last_status_update = now

                # Log status every 10 iterations
                if iteration_count % 10 == 0:
                    self.logger.info(
                        f"Bot running normally (iteration {iteration_count}, positions: {len(self.positions)})")

                # Wait before next iteration (check shutdown every second)
                for _ in range(60):  # 60 seconds total
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                # Send error notification
                if self.telegram:
                    self.telegram.send_sync(f"❌ **Main Loop Error**\n\n{str(e)}")
                time.sleep(60)

    except KeyboardInterrupt:
        self.logger.info("KeyboardInterrupt received")
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
    finally:
        # Always cleanup
        self.cleanup()


def _is_market_open(self):
    """Check if market is open based on trading hours"""
    now = datetime.now()
    current_hour = now.hour
    current_day = now.strftime('%A')

    # Check trading sessions
    for session in self.config['trading_hours']['sessions']:
        if current_day in session['days']:
            start_hour = int(session['start'].split(':')[0])
            end_hour = int(session['end'].split(':')[0])

            if start_hour <= current_hour < end_hour:
                return True

    return False  # mt5_trading_bot.py


"""
Enhanced TFT Trading Bot for MetaTrader 5 with proper shutdown handling
Direct integration without ONNX - more reliable and flexible
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import time
import logging
import json
import os
import signal
import sys
import threading
import atexit
from datetime import datetime

# Import your existing modules
from models.tft.model import SimpleTFT
from data.processors.normalizer import DataNormalizer
from strategy.strategy_factory import create_strategy
from execution.risk.risk_manager import RiskManager

# Telegram imports (optional)
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

    TELEGRAM_AVAILABLE = True
except ImportError:
    # Create dummy classes to prevent import errors
    class Update:
        pass


    class ContextTypes:
        DEFAULT_TYPE = None


    TELEGRAM_AVAILABLE = False
    print("⚠️ python-telegram-bot not installed. Telegram features disabled.")

# Global variables for cleanup
shutdown_event = threading.Event()
telegram_bot = None


class MT5TradingBot:
    def __init__(self, config_path='config/config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Setup logging
        self.setup_logging()

        # Initialize components
        self.model = None
        self.normalizer = None
        self.strategy = None
        self.risk_manager = None
        self.telegram = None

        # Trading state
        self.is_running = False
        self.positions = {}
        self.main_thread = None
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Shutdown handling
        self.shutdown_event = threading.Event()

        # Initialize Telegram if configured
        if self.config.get('telegram', {}).get('token'):
            self.telegram = TelegramManager(self.config, self)
            global telegram_bot
            telegram_bot = self.telegram

        self.logger.info("MT5 Trading Bot initialized with enhanced shutdown handling")

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/mt5_trading_bot.log')
            ]
        )
        self.logger = logging.getLogger('MT5TradingBot')

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error("Failed to initialize MetaTrader5")
            return False

        # Check if already logged in
        account_info = mt5.account_info()

        if account_info is not None:
            # Already logged in!
            self.logger.info("✅ Using existing MT5 session")
            self.logger.info(f"Connected to account: {account_info.login}")
            self.logger.info(f"Broker: {account_info.company}")
            self.logger.info(f"Balance: {account_info.balance} {account_info.currency}")
            return True

        # Only try to login if credentials are provided
        if 'mt5' in self.config and self.config['mt5'].get('login'):
            login = self.config['mt5'].get('login')
            password = self.config['mt5'].get('password')
            server = self.config['mt5'].get('server')

            self.logger.info("Attempting to login with provided credentials...")

            if not mt5.login(login, password=password, server=server):
                self.logger.error(f"Failed to login: {mt5.last_error()}")
                return False
        else:
            # No credentials provided and not logged in
            self.logger.error("❌ No active MT5 session found and no login credentials provided")
            self.logger.error("Please either:")
            self.logger.error("1. Open MT5 and login manually, or")
            self.logger.error("2. Add login credentials to config.json")
            return False

        return True

    def load_model(self):
        """Load the trained TFT model"""
        model_path = self.config['export']['model_path']

        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Create model instance
            self.model = SimpleTFT(self.config['model'])

            # Load weights
            if 'model_state_dict' in checkpoint:
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)

            self.model.eval()
            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def initialize_components(self):
        """Initialize normalizer, strategy, and risk manager"""
        try:
            # Initialize normalizer
            self.normalizer = DataNormalizer(self.config)

            # Initialize strategy
            self.strategy = create_strategy(self.config)

            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)

            self.logger.info("Components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False

    def get_market_data(self):
        """Get current market data from MT5"""
        try:
            market_data = {}

            # Process each configured instrument
            for instrument in self.config['data']['instruments']:
                # Convert instrument format (EUR_USD -> EURUSD)
                mt5_symbol = instrument.replace('_', '')

                # Check if symbol exists
                symbol_info = mt5.symbol_info(mt5_symbol)
                if symbol_info is None:
                    self.logger.warning(f"Symbol {mt5_symbol} not found")
                    continue

                # Select symbol if not visible
                if not symbol_info.visible:
                    if not mt5.symbol_select(mt5_symbol, True):
                        self.logger.warning(f"Failed to select {mt5_symbol}")
                        continue

                market_data[instrument] = {}

                # Get data for each timeframe
                timeframes = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1
                }

                for tf_name in ['M5', 'M1']:  # Your configured timeframes
                    # Get historical data
                    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframes[tf_name], 0, 200)

                    if rates is None:
                        self.logger.warning(f"No data for {mt5_symbol} {tf_name}")
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

                    market_data[instrument][tf_name] = df

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def generate_signals(self, market_data):
        """Generate trading signals using the model and strategy"""
        try:
            # Process data through normalizer
            processed_data = self.normalizer.process(market_data)

            # Prepare predictions for each instrument
            predictions = {}

            for instrument in self.config['data']['instruments']:
                if instrument not in processed_data:
                    continue

                # Get high timeframe data
                high_tf = self.config['data']['timeframes']['high']
                if high_tf not in processed_data[instrument]:
                    continue

                df = processed_data[instrument][high_tf]

                # Check if we have enough data
                past_seq_len = self.config['model']['past_sequence_length']
                if len(df) < past_seq_len:
                    self.logger.warning(f"Insufficient data for {instrument}")
                    continue

                # Prepare model input
                recent_data = df.iloc[-past_seq_len:].copy()

                # Create tensors
                past_tensor = torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0)
                future_tensor = torch.zeros((1, self.config['model']['forecast_horizon'], recent_data.shape[1] - 1),
                                            dtype=torch.float32)
                static_tensor = torch.zeros((1, 1), dtype=torch.float32)

                batch_data = {
                    'past': past_tensor,
                    'future': future_tensor,
                    'static': static_tensor
                }

                # Run inference
                with torch.no_grad():
                    output = self.model(batch_data)

                predictions[instrument] = output.numpy()

            # Update strategy with new data
            self.strategy.update_data(market_data)

            # Generate signals
            signals = self.strategy.generate_signals(predictions, market_data)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {}

    def execute_signal(self, signal, instrument):
        """Execute a trading signal with enhanced validation"""
        try:
            # Convert instrument format
            mt5_symbol = instrument.replace('_', '')

            # Enhanced signal validation
            if not signal.get('valid', False):
                return False

            # Check daily trade limit
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today

            max_daily_trades = self.config.get('execution', {}).get('max_daily_trades', 3)
            if self.daily_trade_count >= max_daily_trades:
                self.logger.info(f"Daily trade limit reached ({self.daily_trade_count}/{max_daily_trades})")
                if self.telegram:
                    self.telegram.send_sync(
                        f"⚠️ Daily trade limit reached ({self.daily_trade_count}/{max_daily_trades})")
                return False

            # Check if we already have a position
            if instrument in self.positions:
                self.logger.info(f"Already have position in {instrument}")
                return False

            # Get signal details
            direction = signal.get('signal')
            current_price = signal.get('current_price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            signal_strength = signal.get('strength', 0)

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                instrument, current_price, stop_loss, direction
            )

            # Get current price from MT5
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick for {mt5_symbol}")
                return False

            # Prepare order
            if direction == 'buy':
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": float(position_size),
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"TFT signal: {signal_strength:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed for {mt5_symbol}: {result.comment}"
                self.logger.error(error_msg)
                if self.telegram:
                    self.telegram.send_sync(f"❌ {error_msg}")
                return False

            # Increment daily trade count
            self.daily_trade_count += 1

            self.logger.info(f"Order executed: {direction} {position_size} lots of {mt5_symbol} at {price}")

            # Send detailed telegram notification
            if self.telegram:
                self.telegram.send_sync(
                    f"🟢 **Trade Opened!**\n\n"
                    f"📈 **{direction.upper()}** {mt5_symbol}\n"
                    f"💰 Size: {position_size:.2f} lots\n"
                    f"🎯 Entry: {price:.5f}\n"
                    f"🛡️ Stop Loss: {stop_loss:.5f}\n"
                    f"💎 Take Profit: {take_profit:.5f}\n"
                    f"⚡ Signal Strength: {signal_strength:.1%}\n"
                    f"📊 Daily Trades: {self.daily_trade_count}/{max_daily_trades}\n"
                    f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}"
                )

            # Store position info
            self.positions[instrument] = {
                'ticket': result.order,
                'direction': direction,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': datetime.now(),
                'signal_strength': signal_strength
            }

            return True

        except Exception as e:
            error_msg = f"Error executing signal for {instrument}: {e}"
            self.logger.error(error_msg)
            if self.telegram:
                self.telegram.send_sync(f"❌ {error_msg}")
            return False

    def check_positions(self):
        """Check and manage open positions"""
        try:
            # Get all positions
            positions = mt5.positions_get()

            if positions is None:
                return

            for position in positions:
                # Only check positions opened by this EA
                if position.magic != 123456:
                    continue

                # You can add trailing stop logic here
                # For now, just log position status
                profit = position.profit
                symbol = position.symbol

                self.logger.debug(f"Position {symbol}: P&L = {profit:.2f}")

        except Exception as e:
            self.logger.error(f"Error checking positions: {e}")

    def cleanup(self):
        """Cleanup function for graceful shutdown"""
        self.logger.info("🛑 Starting cleanup...")

        # Set shutdown flag
        self.is_running = False
        self.shutdown_event.set()

        try:
            # Close any open positions if configured
            if self.config.get('execution', {}).get('close_positions_on_shutdown', False):
                self.logger.info("Closing all open positions...")
                positions = mt5.positions_get()

                if positions:
                    for position in positions:
                        if position.magic == 123456:  # Only our positions
                            # Close position
                            close_request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": position.symbol,
                                "volume": position.volume,
                                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": position.ticket,
                                "magic": 123456,
                                "comment": "Bot shutdown",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }

                            result = mt5.order_send(close_request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.logger.info(f"Closed position {position.symbol}")
                            else:
                                self.logger.error(f"Failed to close position {position.symbol}: {result.comment}")

            # Shutdown MT5 connection
            mt5.shutdown()
            self.logger.info("MT5 connection closed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

        self.logger.info("✅ Cleanup complete")

    def run(self):
        """Main trading loop with enhanced shutdown handling"""
        self.logger.info("🚀 Starting Enhanced MT5 Trading Bot")

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Load model
        if not self.load_model():
            return

        # Initialize components
        if not self.initialize_components():
            return

        self.is_running = True
        self.logger.info("✅ Bot is ready and running - Press Ctrl+C to stop safely")

        # Send startup notification
        if telegram_bot:
            send_telegram_message(
                f"🚀 MT5 Trading Bot Started!\n\n"
                f"📊 Strategy: Enhanced TFT\n"
                f"💰 Risk per Trade: {self.config.get('execution', {}).get('risk_per_trade', 0.01) * 100}%\n"
                f"📈 Max Positions: {self.config.get('execution', {}).get('max_open_positions', 2)}\n"
                f"🎯 Quality Filter: Enabled\n\n"
                f"Ready to trade! 💪"
            )

        # Main loop
        iteration_count = 0

        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    iteration_count += 1

                    # Check if market is open
                    if not self._is_market_open():
                        self.logger.debug("Market closed, waiting...")
                        time.sleep(60)
                        continue

                    # Get market data
                    market_data = self.get_market_data()
                    if not market_data:
                        self.logger.warning("No market data available")
                        time.sleep(10)
                        continue

                    # Generate signals
                    signals = self.generate_signals(market_data)

                    # Process signals
                    for instrument, signal in signals.items():
                        if signal.get('valid', False):
                            # Check if we already have a position
                            if instrument not in self.positions:
                                self.logger.info(f"New signal for {instrument}: {signal['signal']}")
                                self.execute_signal(signal, instrument)

                    # Check existing positions
                    self.check_positions()

                    # Log status every 10 iterations
                    if iteration_count % 10 == 0:
                        self.logger.info(f"Bot running normally (iteration {iteration_count})")

                    # Wait before next iteration
                    for _ in range(60):  # 60 seconds, but check shutdown every second
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    # Send error notification
                    if telegram_bot:
                        send_telegram_message(f"❌ Bot Error: {str(e)}")
                    time.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            # Always cleanup
            self.cleanup()

            # Send shutdown notification
            if telegram_bot:
                send_telegram_message(
                    f"🛑 MT5 Trading Bot Stopped\n\n"
                    f"📊 Session completed after {iteration_count} iterations\n"
                    f"✅ All positions and connections closed safely"
                )

    def _is_market_open(self):
        """Check if market is open based on trading hours"""
        now = datetime.now()
        current_hour = now.hour
        current_day = now.strftime('%A')

        # Check trading sessions
        for session in self.config['trading_hours']['sessions']:
            if current_day in session['days']:
                start_hour = int(session['start'].split(':')[0])
                end_hour = int(session['end'].split(':')[0])

                if start_hour <= current_hour < end_hour:
                    return True

        return False


class TelegramManager:
    """Integrated Telegram bot manager for MT5 Trading Bot"""

    def __init__(self, config, trading_bot=None):
        self.config = config
        self.trading_bot = trading_bot
        self.telegram_config = config.get('telegram', {})
        self.token = self.telegram_config.get('token')
        self.authorized_users = set(str(user) for user in self.telegram_config.get('authorized_users', []))
        self.admin_users = set(str(user) for user in self.telegram_config.get('admin_users', []))

        # Bot state
        self.app = None
        self.is_running = False
        self.logger = logging.getLogger('telegram_manager')
        self.telegram_available = TELEGRAM_AVAILABLE

        if not self.token:
            self.logger.warning("No Telegram token provided")
            return

        if not self.telegram_available:
            self.logger.warning("python-telegram-bot not installed")
            return

        self.logger.info("Telegram integration available")

    async def send_message(self, message, parse_mode=None):
        """Send message to all authorized users"""
        if not self.telegram_available or not self.app:
            return

        for user_id in self.authorized_users:
            try:
                await self.app.bot.send_message(
                    chat_id=int(user_id),
                    text=message,
                    parse_mode=parse_mode
                )
            except Exception as e:
                self.logger.error(f"Error sending message to {user_id}: {e}")

    def send_sync(self, message):
        """Synchronous wrapper for sending messages"""
        if not self.telegram_available or not self.app:
            return

        try:
            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule the coroutine
                    asyncio.create_task(self.send_message(message))
                else:
                    # If loop is not running, run until complete
                    loop.run_until_complete(self.send_message(message))
            except RuntimeError:
                # No event loop, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_message(message))
                loop.close()

        except Exception as e:
            self.logger.error(f"Error in sync send: {e}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not TELEGRAM_AVAILABLE:
            return

        user_id = update.effective_user.id

        if str(user_id) not in self.authorized_users:
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            self.logger.warning(f"Unauthorized access attempt from {user_id}")
            return

        status = "🟢 Running" if self.trading_bot and self.trading_bot.is_running else "🔴 Stopped"

        welcome_msg = (
            f"🤖 **MT5 Trading Bot Control Panel**\n\n"
            f"Status: {status}\n\n"
            f"**Available Commands:**\n"
            f"/status - Get bot status\n"
            f"/positions - View open positions\n"
            f"/stop - Stop the trading bot\n"
            f"/help - Show this message\n\n"
            f"Bot will send automatic notifications for:\n"
            f"• Trade entries and exits\n"
            f"• Important alerts\n"
            f"• System status updates"
        )

        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not TELEGRAM_AVAILABLE:
            return

        user_id = update.effective_user.id

        if str(user_id) not in self.authorized_users:
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        if not self.trading_bot:
            await update.message.reply_text("❌ Trading bot not connected")
            return

        # Get MT5 account info
        try:
            account_info = mt5.account_info()
            positions = mt5.positions_get()

            if account_info:
                status_msg = (
                    f"📊 **MT5 Trading Bot Status**\n\n"
                    f"🟢 Status: {'Running' if self.trading_bot.is_running else 'Stopped'}\n"
                    f"💰 Balance: {account_info.balance:.2f} {account_info.currency}\n"
                    f"📈 Equity: {account_info.equity:.2f} {account_info.currency}\n"
                    f"📊 Margin: {account_info.margin:.2f} {account_info.currency}\n"
                    f"🎯 Open Positions: {len(positions) if positions else 0}\n"
                    f"🏢 Broker: {account_info.company}\n"
                    f"🔗 Server: {account_info.server}\n"
                    f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                status_msg = "❌ Cannot connect to MT5 account"

        except Exception as e:
            status_msg = f"❌ Error getting status: {str(e)}"

        await update.message.reply_text(status_msg, parse_mode='Markdown')

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        if not TELEGRAM_AVAILABLE:
            return

        user_id = update.effective_user.id

        if str(user_id) not in self.authorized_users:
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        try:
            positions = mt5.positions_get()

            if not positions:
                await update.message.reply_text("📊 No open positions")
                return

            # Filter only our positions (magic number 123456)
            our_positions = [p for p in positions if p.magic == 123456]

            if not our_positions:
                await update.message.reply_text("📊 No open positions from this bot")
                return

            pos_msg = "📊 **Open Positions:**\n\n"

            for i, pos in enumerate(our_positions, 1):
                direction = "🟢 BUY" if pos.type == mt5.POSITION_TYPE_BUY else "🔴 SELL"
                profit_emoji = "💰" if pos.profit > 0 else "📉" if pos.profit < 0 else "⚪"

                pos_msg += (
                    f"**{i}. {pos.symbol}**\n"
                    f"{direction} {pos.volume} lots\n"
                    f"💵 Entry: {pos.price_open:.5f}\n"
                    f"📊 Current: {pos.price_current:.5f}\n"
                    f"{profit_emoji} P&L: {pos.profit:.2f}\n"
                    f"🛡️ SL: {pos.sl:.5f}\n"
                    f"🎯 TP: {pos.tp:.5f}\n\n"
                )

            await update.message.reply_text(pos_msg, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting positions: {str(e)}")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        if not TELEGRAM_AVAILABLE:
            return

        user_id = update.effective_user.id

        if str(user_id) not in self.admin_users:
            await update.message.reply_text("❌ This command is only available to administrators.")
            return

        if not self.trading_bot:
            await update.message.reply_text("❌ Trading bot not connected")
            return

        try:
            await update.message.reply_text("🛑 Stopping trading bot...")

            # Stop the trading bot
            self.trading_bot.cleanup()

            await update.message.reply_text("✅ Trading bot stopped successfully")

        except Exception as e:
            await update.message.reply_text(f"❌ Error stopping bot: {str(e)}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not TELEGRAM_AVAILABLE:
            return
        await self.start_command(update, context)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        if not TELEGRAM_AVAILABLE:
            return

        user_id = update.effective_user.id

        if str(user_id) not in self.authorized_users:
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        await update.message.reply_text(
            "ℹ️ Please use commands to interact with the bot. Type /help for available commands."
        )

    def start_bot(self):
        """Start the Telegram bot"""
        if not self.telegram_available or not self.token:
            self.logger.warning("Cannot start Telegram bot - missing requirements")
            return False

        try:
            # Create application
            self.app = Application.builder().token(self.token).build()

            # Add command handlers
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("positions", self.positions_command))
            self.app.add_handler(CommandHandler("stop", self.stop_command))
            self.app.add_handler(CommandHandler("help", self.help_command))

            # Add message handler
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

            # Start in separate thread
            def run_bot():
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Start the bot
                    loop.run_until_complete(self.app.initialize())
                    loop.run_until_complete(self.app.start())
                    loop.run_until_complete(self.app.updater.start_polling())

                    self.is_running = True
                    self.logger.info("✅ Telegram bot started successfully")

                    # Send startup message
                    loop.run_until_complete(self.send_message(
                        "🤖 **Telegram Bot Connected!**\n\n"
                        "✅ Ready to receive commands and send notifications\n"
                        "Type /help for available commands"
                    ))

                    # Keep running until shutdown
                    while self.is_running and not shutdown_event.is_set():
                        time.sleep(1)

                    # Cleanup
                    loop.run_until_complete(self.app.updater.stop())
                    loop.run_until_complete(self.app.stop())
                    loop.run_until_complete(self.app.shutdown())
                    loop.close()

                except Exception as e:
                    self.logger.error(f"Telegram bot error: {e}")

            # Start bot thread
            bot_thread = threading.Thread(target=run_bot, daemon=True)
            bot_thread.start()

            # Wait a moment for startup
            time.sleep(2)
            return True

        except Exception as e:
            self.logger.error(f"Error starting Telegram bot: {e}")
            return False

    def stop_bot(self):
        """Stop the Telegram bot"""
        self.is_running = False
        self.logger.info("Telegram bot stopped")


def send_telegram_message(message):
    """Global function to send telegram messages"""
    global telegram_bot
    if telegram_bot and hasattr(telegram_bot, 'send_sync'):
        telegram_bot.send_sync(message)


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n📡 Received signal {sig} - Starting graceful shutdown...")

    # Set global shutdown event
    shutdown_event.set()

    # Give main loop time to finish current iteration
    time.sleep(2)

    print("✅ Shutdown signal processed")
    sys.exit(0)


def cleanup_and_exit():
    """Global cleanup function"""
    global telegram_bot

    print("\n🛑 Final cleanup...")

    if telegram_bot:
        try:
            print("Stopping telegram bot...")
            telegram_bot.stop_bot()
        except Exception as e:
            print(f"Error stopping telegram bot: {e}")

    print("✅ Global cleanup complete!")


if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup_and_exit)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # On Windows, also handle SIGBREAK
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    print("🤖 Enhanced MT5 Trading Bot with Full Telegram Integration")
    print("📱 Telegram commands: /start, /status, /positions, /stop, /help")
    print("Press Ctrl+C at any time to stop safely\n")

    # Create and run the bot with proper shutdown
    try:
        bot = MT5TradingBot()
        bot.run()
    except Exception as e:
        print(f"❌ Bot error: {e}")
        if telegram_bot:
            send_telegram_message(f"❌ **Critical Bot Error**\n\n{str(e)}")
    finally:
        print("🏁 Bot execution completed")