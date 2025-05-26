# execution/position/advanced_position_manager.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import time
import json

from execution.position.position_manager import Position
from execution.risk.risk_manager import RiskManager
from utils.error_handling.error_manager import ErrorManager, handle_errors
from strategy.hooks.event_system import EventManager

class AdvancedPositionManager:
    """
    Advanced position management with dynamic adjustment strategies
    """

    def __init__(self, config, broker):
        self.config = config
        self.broker = broker
        self.error_manager = ErrorManager(config)
        self.event_manager = EventManager(config)
        self.risk_manager = RiskManager(config)

        # Set up logging
        self.logger = logging.getLogger('advanced_position_manager')
        self.logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Position tracking
        self.positions = {}  # instrument -> Position
        self.position_lock = threading.Lock()

        # Position history for analytics
        self.position_history = []

        # Load position management strategies
        self.pyramid_enabled = config.get('execution', {}).get('advanced', {}).get('enable_pyramiding', False)
        self.partial_exits_enabled = config.get('execution', {}).get('advanced', {}).get('enable_partial_exits', True)
        self.breakeven_enabled = config.get('execution', {}).get('advanced', {}).get('enable_breakeven', True)

        # Breakeven settings
        self.breakeven_threshold = config.get('execution', {}).get('advanced', {}).get('breakeven_threshold', 0.5)

        # Partial exit settings
        self.partial_exit_levels = config.get('execution', {}).get('advanced', {}).get('partial_exit_levels', [0.25, 0.5, 0.75])
        self.partial_exit_sizes = config.get('execution', {}).get('advanced', {}).get('partial_exit_sizes', [0.25, 0.25, 0.25])

        # Pyramiding settings
        self.pyramid_max_entries = config.get('execution', {}).get('advanced', {}).get('pyramid_max_entries', 3)
        self.pyramid_minimal_distance = config.get('execution', {}).get('advanced', {}).get('pyramid_minimal_distance', 0.01)

        # Load any saved positions
        self._load_positions()

        # Start position management thread
        self.is_running = True
        self.management_thread = threading.Thread(target=self._position_management_loop, daemon=True)
        self.management_thread.start()

        self.logger.info("Advanced position manager initialized")

    @handle_errors(error_type='execution_error')
    def open_position(self, instrument, direction, entry_price, size, stop_loss, take_profit, signal_strength=0.5, tags=None):
        """
        Open a new position or add to existing if pyramiding is enabled

        Parameters:
        - instrument: Trading instrument
        - direction: 'long' or 'short'
        - entry_price: Entry price level
        - size: Position size in lots
        - stop_loss: Stop loss level
        - take_profit: Take profit level
        - signal_strength: Signal confidence (0.0-1.0)
        - tags: Optional tags for position categorization

        Returns:
        - (success, result) tuple
        """
        with self.position_lock:
            # Check if position already exists for this instrument
            if instrument in self.positions:
                existing_position = self.positions[instrument]

                # Check if pyramiding is enabled and directions match
                if self.pyramid_enabled and existing_position.direction == direction:
                    # Check if max entries reached
                    if existing_position.entry_count >= self.pyramid_max_entries:
                        return False, "Maximum pyramid entries reached"

                    # Check if minimum distance is satisfied
                    price_diff_pct = abs(entry_price - existing_position.entry_price) / existing_position.entry_price

                    if price_diff_pct < self.pyramid_minimal_distance:
                        return False, f"Minimum pyramid distance not met ({price_diff_pct:.2%} < {self.pyramid_minimal_distance:.2%})"

                    # Calculate average entry, update size
                    total_value = (existing_position.entry_price * existing_position.size) + (entry_price * size)
                    total_size = existing_position.size + size
                    avg_entry = total_value / total_size if total_size > 0 else entry_price

                    # Execute pyramid entry order
                    order_result = self.broker.place_order(
                        instrument=instrument,
                        direction=direction,
                        size=size,
                        stop_loss=stop_loss,  # Will be adjusted after entry
                        take_profit=take_profit  # Will be adjusted after entry
                    )

                    if order_result['success']:
                        # Update position
                        existing_position.entry_price = avg_entry
                        existing_position.size = total_size
                        existing_position.entry_count += 1
                        existing_position.pyramid_entries.append({
                            'price': entry_price,
                            'size': size,
                            'time': datetime.now().isoformat()
                        })

                        # Update stop loss and take profit
                        self._update_exit_levels(existing_position)

                        # Emit event
                        self.event_manager.emit('trade:pyramid_entry', {
                            'instrument': instrument,
                            'direction': direction,
                            'entry_price': entry_price,
                            'total_size': total_size,
                            'avg_entry': avg_entry,
                            'entry_count': existing_position.entry_count
                        }, source='position_manager')

                        return True, existing_position
                    else:
                        return False, order_result['error']

                elif existing_position.direction != direction:
                    # Close existing position in opposite direction first
                    close_result, _ = self.close_position(instrument, entry_price)

                    if not close_result:
                        return False, "Failed to close existing position in opposite direction"

                    # Proceed to open new position
                else:
                    return False, "Position already exists and pyramiding is disabled"

            # Check if we've reached max open positions
            max_positions = self.config.get('execution', {}).get('max_open_positions', 10)

            if len(self.positions) >= max_positions:
                return False, f"Maximum open positions reached ({max_positions})"

            # Create new position object
            position = Position(
                instrument=instrument,
                direction=direction,
                entry_price=entry_price,
                size=size,
                initial_stop=stop_loss,
                current_stop=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now()
            )

            # Add advanced management fields
            position.entry_count = 1
            position.pyramid_entries = [{
                'price': entry_price,
                'size': size,
                'time': datetime.now().isoformat()
            }]
            position.partial_exits = []
            position.breakeven_triggered = False
            position.signal_strength = signal_strength
            position.tags = tags or []
            position.risk_reward_ratio = self._calculate_risk_reward(entry_price, stop_loss, take_profit, direction)
            position.risk_amount = abs(entry_price - stop_loss) * size

            # Execute order via broker
            order_result = self.broker.place_order(
                instrument=instrument,
                direction=direction,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if order_result['success']:
                # Store position
                self.positions[instrument] = position

                # Emit event
                self.event_manager.emit('trade:entry', {
                    'instrument': instrument,
                    'direction': direction,
                    'entry_price': entry_price,
                    'size': size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': position.risk_reward_ratio
                }, source='position_manager')

                # Save positions to disk
                self._save_positions()

                return True, position
            else:
                return False, order_result['error']

    @handle_errors(error_type='execution_error')
    def close_position(self, instrument, exit_price=None, reason="Manual closure", partial_size=None):
        """
        Close an existing position (fully or partially)

        Parameters:
        - instrument: Trading instrument
        - exit_price: Optional exit price override
        - reason: Reason for closing
        - partial_size: Optional size for partial closure

        Returns:
        - (success, result) tuple
        """
        with self.position_lock:
            # Check if position exists
            if instrument not in self.positions:
                return False, "No position exists for this instrument"

            position = self.positions[instrument]

            # Handle partial closure if specified
            if partial_size is not None and partial_size < position.size:
                # Execute partial close order
                close_result = self.broker.close_partial(
                    instrument=instrument,
                    size=partial_size
                )

                if close_result['success']:
                    # Update position size
                    old_size = position.size
                    position.size -= partial_size

                    # Calculate realized P&L for this exit
                    if position.direction == 'long':
                        partial_pnl = (close_result['price'] - position.entry_price) * partial_size
                    else:
                        partial_pnl = (position.entry_price - close_result['price']) * partial_size

                    # Record partial exit
                    exit_record = {
                        'price': close_result['price'],
                        'size': partial_size,
                        'pnl': partial_pnl,
                        'time': datetime.now().isoformat(),
                        'reason': reason
                    }

                    position.partial_exits.append(exit_record)

                    # Emit event
                    self.event_manager.emit('trade:partial_exit', {
                        'instrument': instrument,
                        'direction': position.direction,
                        'exit_price': close_result['price'],
                        'size': partial_size,
                        'remaining_size': position.size,
                        'pnl': partial_pnl,
                        'reason': reason
                    }, source='position_manager')

                    # Save positions to disk
                    self._save_positions()

                    return True, exit_record
                else:
                    return False, close_result['error']

            # Full position closure
            close_result = self.broker.close_order(instrument)

            if close_result['success']:
                # Update position
                position.exit_price = exit_price or close_result['price']
                position.exit_time = datetime.now()
                position.status = 'closed'

                # Calculate P&L
                if position.direction == 'long':
                    position.pnl = (position.exit_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - position.exit_price) * position.size

                # Add to history
                self.position_history.append(self._position_to_dict(position))

                # Remove from active positions
                del self.positions[instrument]

                # Emit event
                event_type = 'trade:exit'
                if reason == "Stop loss hit":
                    event_type = 'trade:stop_hit'
                elif reason == "Take profit hit":
                    event_type = 'trade:tp_hit'

                self.event_manager.emit(event_type, {
                    'instrument': instrument,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'size': position.size,
                    'pnl': position.pnl,
                    'entry_time': position.entry_time.isoformat(),
                    'exit_time': position.exit_time.isoformat(),
                    'duration': (position.exit_time - position.entry_time).total_seconds() / 60,  # minutes
                    'reason': reason
                }, source='position_manager')

                # Save positions to disk
                self._save_positions()

                return True, position
            else:
                return False, close_result['error']

    @handle_errors(error_type='execution_error')
    def modify_position(self, instrument, stop_loss=None, take_profit=None):
        """
        Modify an existing position's stop loss or take profit

        Parameters:
        - instrument: Trading instrument
        - stop_loss: New stop loss level
        - take_profit: New take profit level

        Returns:
        - (success, result) tuple
        """
        with self.position_lock:
            # Check if position exists
            if instrument not in self.positions:
                return False, "No position exists for this instrument"

            position = self.positions[instrument]

            # Update stop loss if provided
            if stop_loss is not None:
                # Validate stop loss direction
                if position.direction == 'long' and stop_loss >= position.entry_price:
                    return False, "Stop loss must be below entry price for long positions"
                elif position.direction == 'short' and stop_loss <= position.entry_price:
                    return False, "Stop loss must be above entry price for short positions"

                # Update stop loss with broker
                sl_result = self.broker.update_stop_loss(instrument, stop_loss)

                if sl_result['success']:
                    position.current_stop = stop_loss
                else:
                    return False, sl_result['error']

            # Update take profit if provided
            if take_profit is not None:
                # Validate take profit direction
                # Temporal Fusion Transformer Trading Bot - Implementation Guide

## 1. Core Modules

### Configuration System (`config/config.json`)

```json
{
  "data": {
    "source": "oanda",
    "instruments": ["EUR_USD", "GBP_USD", "USD_JPY"],
    "timeframes": {
      "high": "M5",
      "low": "M1"
    },
    "history_period": "3Y"
  },
  "model": {
    "hidden_size": 64,
    "dropout": 0.1,
    "attention_heads": 4,
    "lstm_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 256,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "quantiles": [0.1, 0.5, 0.9]
  },
  "strategy": {
    "trend_filter": true,
    "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
    "confidence_thresholds": {
      "high": 0.8,
      "medium": 0.6,
      "low": 0.4
    }
  },
  "execution": {
    "risk_per_trade": 0.02,
    "max_open_positions": 5,
    "min_lot_size": 0.01,
    "max_lot_size": 1.0,
    "stop_loss": {
      "type": "atr_multiple",
      "value": 1.5
    },
    "take_profit": {
      "type": "risk_reward",
      "value": 2.0
    },
    "trailing_stop": {
      "activation_percent": 0.5,
      "step_size": 0.1
    }
  },
  "trading_hours": {
    "timezone": "Europe/London",
    "sessions": [
      {"start": "08:00", "end": "16:00", "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
    ]
  },
  "export": {
    "auto_export_after_training": true,
    "export_dir": "exported_models",
    "model_path": "models/checkpoints/best_model.pt"
  }
}
