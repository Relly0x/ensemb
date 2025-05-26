import MetaTrader5 as mt5
import time
import logging
from datetime import datetime


class MT5Broker:
    """MT5 broker implementation for executing trades"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('mt5_broker')
        self.initialized = False
        self.initialize()

    def initialize(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # Login if credentials provided
        if 'login' in self.config.get('mt5', {}):
            login = self.config['mt5']['login']
            password = self.config['mt5']['password']
            server = self.config['mt5'].get('server', '')

            if not mt5.login(login, password, server):
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False

        self.initialized = True
        return True

    def place_order(self, instrument, direction, size, stop_loss, take_profit):
        """Place a new order"""
        if not self.initialized and not self.initialize():
            return {'success': False, 'error': 'MT5 not initialized'}

        # Determine order type
        order_type = mt5.ORDER_TYPE_BUY if direction == 'long' else mt5.ORDER_TYPE_SELL

        # Get current price
        tick = mt5.symbol_info_tick(instrument)
        if tick is None:
            return {'success': False, 'error': f"Failed to get price for {instrument}"}

        price = tick.ask if direction == 'long' else tick.bid

        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": instrument,
            "volume": size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": 12345,  # Magic number for identifying our EA's orders
            "comment": "TFT Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }

        # Send order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f"Order failed with error code: {result.retcode}"
            }

        return {
            'success': True,
            'order_id': result.order,
            'price': price
        }

    def close_order(self, instrument):
        """Close an open position"""
        if not self.initialized and not self.initialize():
            return {'success': False, 'error': 'MT5 not initialized'}

        # Find position
        positions = mt5.positions_get(symbol=instrument)

        if positions is None or len(positions) == 0:
            return {'success': False, 'error': f"No open position for {instrument}"}

        position = positions[0]

        # Determine closing order type (opposite of position)
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

        # Get price
        tick = mt5.symbol_info_tick(instrument)
        price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask

        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": instrument,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "magic": 12345,
            "comment": "Close by TFT Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }

        # Send close order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f"Close order failed with error code: {result.retcode}"
            }

        return {
            'success': True,
            'price': price
        }

    def update_stop_loss(self, instrument, stop_loss):
        """Update stop loss for an open position"""
        if not self.initialized and not self.initialize():
            return {'success': False, 'error': 'MT5 not initialized'}

        # Find position
        positions = mt5.positions_get(symbol=instrument)

        if positions is None or len(positions) == 0:
            return {'success': False, 'error': f"No open position for {instrument}"}

        position = positions[0]

        # Prepare modify request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": instrument,
            "sl": stop_loss,
            "tp": position.tp,
            "position": position.ticket,
            "magic": 12345
        }

        # Send modify request
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f"SL update failed with error code: {result.retcode}"
            }

        return {'success': True}

    def update_take_profit(self, instrument, take_profit):
        """Update take profit for an open position"""
        if not self.initialized and not self.initialize():
            return {'success': False, 'error': 'MT5 not initialized'}

        # Find position
        positions = mt5.positions_get(symbol=instrument)

        if positions is None or len(positions) == 0:
            return {'success': False, 'error': f"No open position for {instrument}"}

        position = positions[0]

        # Prepare modify request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": instrument,
            "sl": position.sl,
            "tp": take_profit,
            "position": position.ticket,
            "magic": 12345
        }

        # Send modify request
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f"TP update failed with error code: {result.retcode}"
            }

        return {'success': True}

    def get_current_prices(self):
        """Get current prices for all instruments"""
        if not self.initialized and not self.initialize():
            return {}

        instruments = self.config['data']['instruments']
        prices = {}

        for instrument in instruments:
            tick = mt5.symbol_info_tick(instrument)
            if tick is not None:
                prices[instrument] = (tick.ask + tick.bid) / 2

        return prices

    def get_account_balance(self):
        """Get account balance"""
        if not self.initialized and not self.initialize():
            return 0

        account_info = mt5.account_info()
        if account_info is None:
            return 0

        return account_info.balance

    def get_open_positions(self):
        """Get all open positions"""
        if not self.initialized and not self.initialize():
            return []

        positions = mt5.positions_get()

        if positions is None or len(positions) == 0:
            return []

        result = []

        for position in positions:
            # Only include positions with our magic number
            if position.magic != 12345:
                continue

            direction = 'long' if position.type == mt5.POSITION_TYPE_BUY else 'short'

            result.append({
                'instrument': position.symbol,
                'direction': direction,
                'entry_price': position.price_open,
                'current_price': position.price_current,
                'size': position.volume,
                'stop_loss': position.sl,
                'take_profit': position.tp,
                'ticket': position.ticket,
                'profit': position.profit
            })

        return result

    def __del__(self):
        """Clean up MT5 connection"""
        mt5.shutdown()