import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RiskManager:
    """
    Risk management system for trading decisions
    - Position sizing
    - Risk per trade management
    - Exposure limits
    - Drawdown protection
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('risk_manager')

        # Load risk settings from config
        self.risk_per_trade = config['execution']['risk_per_trade']
        self.max_open_positions = config['execution']['max_open_positions']
        self.min_lot_size = config['execution']['min_lot_size']
        self.max_lot_size = config['execution']['max_lot_size']

        # Optional advanced settings
        self.max_risk_per_instrument = config.get('risk', {}).get('max_risk_per_instrument', 0.05)
        self.max_daily_drawdown = config.get('risk', {}).get('max_daily_drawdown', 0.05)
        self.max_total_risk = config.get('risk', {}).get('max_total_risk', 0.2)

        # Risk state tracking
        self.daily_pnl = 0.0
        self.daily_starting_balance = None
        self.last_reset_day = None
        self.active_risks = {}  # instrument -> risk_amount

        # Initialize
        self._reset_daily_metrics()

        self.logger.info("Risk manager initialized")

    def _reset_daily_metrics(self):
        """Reset daily risk metrics"""
        today = datetime.now().date()

        if self.last_reset_day != today:
            self.daily_pnl = 0.0
            self.daily_starting_balance = None
            self.last_reset_day = today

    def calculate_position_size(self, instrument, current_price, stop_loss, direction):
        """
        Calculate appropriate position size based on risk parameters

        Parameters:
        - instrument: Trading instrument
        - current_price: Current market price
        - stop_loss: Stop loss level
        - direction: Trade direction ('buy' or 'sell')

        Returns:
        - Position size in lots
        """
        self._reset_daily_metrics()

        # Calculate risk amount in account currency
        account_balance = self._get_account_balance()
        risk_amount = account_balance * self.risk_per_trade

        # Calculate stop loss distance in price terms
        if direction == 'buy':
            if stop_loss >= current_price:
                self.logger.warning(f"Invalid stop loss for buy: {stop_loss} >= {current_price}")
                stop_loss = current_price * 0.99  # Default to 1% below
            stop_distance = abs(current_price - stop_loss)
        else:  # sell
            if stop_loss <= current_price:
                self.logger.warning(f"Invalid stop loss for sell: {stop_loss} <= {current_price}")
                stop_loss = current_price * 1.01  # Default to 1% above
            stop_distance = abs(stop_loss - current_price)

        # Calculate position size
        # For simplicity, we'll assume a direct relationship between price move and P&L
        # In a real implementation, you'd need to account for pip value, leverage, etc.
        position_size = risk_amount / stop_distance

        # Convert to lots (standard lot is typically 100,000 units)
        standard_lot = 100000
        position_size_lots = position_size / standard_lot

        # Apply scaling based on risk score or volatility (optional)
        position_size_lots = self._scale_by_volatility(instrument, position_size_lots)

        # Ensure min/max lot size
        position_size_lots = max(self.min_lot_size, min(self.max_lot_size, position_size_lots))

        # Round to 2 decimal places (0.01 lot precision)
        position_size_lots = round(position_size_lots, 2)

        self.logger.info(
            f"Calculated position size for {instrument}: {position_size_lots} lots (risk: {risk_amount:.2f})")

        return position_size_lots

    def _get_account_balance(self):
        """Get account balance from broker or configuration"""
        # In a real implementation, you'd get this from the broker
        return self.config.get('execution', {}).get('initial_balance', 10000.0)

    def _scale_by_volatility(self, instrument, base_size):
        """Scale position size based on instrument volatility"""
        # This is a placeholder - in a real implementation, you'd calculate
        # actual volatility metrics from historical data
        volatility_factor = 1.0

        try:
            # Get volatility adjustment from config if available
            volatility_adjustments = self.config.get('risk', {}).get('volatility_adjustments', {})
            if instrument in volatility_adjustments:
                volatility_factor = volatility_adjustments[instrument]
        except:
            pass

        return base_size * volatility_factor

    def update_risk_state(self, instrument, risk_amount, position_id=None):
        """
        Update risk state with new position

        Parameters:
        - instrument: Trading instrument
        - risk_amount: Risk amount in account currency
        - position_id: Optional position identifier

        Returns:
        - True if risk accepted, False if rejected
        """
        # Check risk limits
        if self._check_risk_limits(instrument, risk_amount):
            # Risk accepted, update state
            if instrument not in self.active_risks:
                self.active_risks[instrument] = 0.0

            self.active_risks[instrument] += risk_amount

            return True
        else:
            # Risk rejected
            return False

    def _check_risk_limits(self, instrument, risk_amount):
        """Check if new risk amount would exceed limits"""
        account_balance = self._get_account_balance()

        # Check instrument risk limit
        instrument_risk = self.active_risks.get(instrument, 0.0) + risk_amount
        if instrument_risk / account_balance > self.max_risk_per_instrument:
            self.logger.warning(f"Instrument risk limit exceeded for {instrument}")
            return False

        # Check total risk limit
        total_risk = sum(self.active_risks.values()) + risk_amount
        if total_risk / account_balance > self.max_total_risk:
            self.logger.warning("Total risk limit exceeded")
            return False

        return True

    def update_trade_result(self, instrument, pnl, position_id=None):
        """
        Update risk state with closed trade result

        Parameters:
        - instrument: Trading instrument
        - pnl: Realized profit/loss
        - position_id: Optional position identifier
        """
        self._reset_daily_metrics()

        # Update daily P&L
        self.daily_pnl += pnl

        # Remove from active risks
        if instrument in self.active_risks:
            self.active_risks[instrument] = max(0, self.active_risks[instrument] - abs(pnl))

        # Check drawdown protection
        self._check_drawdown_protection()

    def _check_drawdown_protection(self):
        """Check if drawdown protection should be activated"""
        if self.daily_starting_balance is None:
            self.daily_starting_balance = self._get_account_balance()

        # Calculate current drawdown
        current_balance = self._get_account_balance()
        daily_drawdown = (self.daily_starting_balance - current_balance) / self.daily_starting_balance

        # Check against threshold
        if daily_drawdown >= self.max_daily_drawdown:
            self.logger.warning(f"Daily drawdown protection triggered: {daily_drawdown:.2%}")
            # In a real implementation, you might want to:
            # 1. Notify the user
            # 2. Reduce position sizes
            # 3. Temporarily halt trading

    def get_risk_report(self):
        """Get current risk exposure report"""
        account_balance = self._get_account_balance()

        return {
            'active_risks': self.active_risks,
            'total_risk': sum(self.active_risks.values()),
            'total_risk_percent': sum(self.active_risks.values()) / account_balance if account_balance > 0 else 0,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': self.daily_pnl / self.daily_starting_balance if self.daily_starting_balance else 0
        }

    def calculate_kelly_position_size(self, win_rate, avg_win, avg_loss, current_price, stop_loss, direction):
        """
        Calculate position size using Kelly Criterion

        Parameters:
        - win_rate: Historical win rate (0.0-1.0)
        - avg_win: Average win amount
        - avg_loss: Average loss amount (positive number)
        - current_price: Current market price
        - stop_loss: Stop loss level
        - direction: Trade direction ('buy' or 'sell')

        Returns:
        - Position size in lots
        """
        # Kelly formula: f* = (p * b - q) / b
        # where:
        # f* = fraction of bankroll to wager
        # p = probability of winning
        # q = probability of losing (1 - p)
        # b = odds received on wager (how much you win divided by how much you lose)

        if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
            return self.calculate_position_size(current_price, stop_loss, direction)

        odds = avg_win / avg_loss
        kelly_fraction = (win_rate * odds - (1 - win_rate)) / odds

        # Often use half-Kelly for more conservative sizing
        kelly_fraction = max(0, kelly_fraction * 0.5)

        # Calculate risk amount
        account_balance = self._get_account_balance()
        risk_amount = account_balance * kelly_fraction

        # Calculate stop loss distance
        if direction == 'buy':
            stop_distance = current_price - stop_loss
        else:  # sell
            stop_distance = stop_loss - current_price

        # Calculate position size
        standard_lot = 100000
        position_size_lots = (risk_amount / stop_distance) / standard_lot

        # Ensure min/max lot size
        position_size_lots = max(self.min_lot_size, min(self.max_lot_size, position_size_lots))

        # Round to 2 decimal places
        position_size_lots = round(position_size_lots, 2)

        return position_size_lots