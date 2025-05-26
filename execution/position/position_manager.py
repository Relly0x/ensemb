from datetime import datetime
from .trailing_stop import TrailingStop

class Position:
    def __init__(self, instrument, direction, entry_price, size, stop_loss, take_profit, entry_time):
        self.instrument = instrument
        self.direction = direction  # 'long' or 'short'
        self.entry_price = entry_price
        self.size = size  # in lots
        self.initial_stop = stop_loss
        self.current_stop = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.status = 'open'

class PositionManager:
    def __init__(self, config, broker):
        self.config = config
        self.broker = broker
        self.positions = {}  # instrument -> Position
        self.trailing_stop = TrailingStop(config)
        self.max_positions = config['execution']['max_open_positions']

    def open_position(self, instrument, direction, entry_price, size, stop_loss, take_profit):
        """
        Open a new position
        """
        if len(self.positions) >= self.max_positions:
            return False, "Maximum positions reached"

        if instrument in self.positions:
            return False, "Position already exists for this instrument"

        # Create new position
        position = Position(
            instrument=instrument,
            direction=direction,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )

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
            return True, position
        else:
            return False, order_result['error']

    def close_position(self, instrument, exit_price=None, reason="Manual closure"):
        """
        Close an existing position
        """
        if instrument not in self.positions:
            return False, "No position exists for this instrument"

        position = self.positions[instrument]

        # Execute close order via broker
        close_result = self.broker.close_order(instrument)

        if close_result['success']:
            # Update position
            position.exit_price = exit_price or close_result['price']
            position.exit_time = datetime.now()
            position.status = 'closed'

            # Calculate PnL
            if position.direction == 'long':
                position.pnl = (position.exit_price - position.entry_price) * position.size
            else:
                position.pnl = (position.entry_price - position.exit_price) * position.size

            # Remove from active positions
            del self.positions[instrument]

            return True, position
        else:
            return False, close_result['error']

    def update_positions(self, current_prices):
        """
        Update all positions with current market prices
        - Check for stop loss and take profit hits
        - Update trailing stops
        """
        for instrument, position in list(self.positions.items()):
            if instrument not in current_prices:
                continue

            current_price = current_prices[instrument]

            # Check for stop loss hit
            if position.direction == 'long' and current_price <= position.current_stop:
                self.close_position(instrument, current_price, "Stop loss hit")
                continue

            if position.direction == 'short' and current_price >= position.current_stop:
                self.close_position(instrument, current_price, "Stop loss hit")
                continue

            # Check for take profit hit
            if position.direction == 'long' and current_price >= position.take_profit:
                self.close_position(instrument, current_price, "Take profit hit")
                continue

            if position.direction == 'short' and current_price <= position.take_profit:
                self.close_position(instrument, current_price, "Take profit hit")
                continue

            # Update trailing stop
            new_stop = self.trailing_stop.calculate_stop_level(position, current_price)
            if new_stop != position.current_stop:
                position.current_stop = new_stop
                # Update stop in broker
                self.broker.update_stop_loss(instrument, new_stop)

    def get_position_summary(self):
        """
        Get summary of all open positions
        """
        return {
            'count': len(self.positions),
            'positions': [
                {
                    'instrument': p.instrument,
                    'direction': p.direction,
                    'size': p.size,
                    'entry_price': p.entry_price,
                    'current_stop': p.current_stop,
                    'take_profit': p.take_profit,
                    'entry_time': p.entry_time.isoformat()
                }
                for p in self.positions.values()
            ]
        }
