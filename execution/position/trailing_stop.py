class TrailingStop:
    def __init__(self, config):
        self.activation_percent = config['execution']['trailing_stop']['activation_percent']
        self.step_size = config['execution']['trailing_stop']['step_size']

    def calculate_stop_level(self, position, current_price):
        """
        Calculate the trailing stop level for a position

        Parameters:
        - position: Position object with entry_price, direction, current_stop
        - current_price: Current market price

        Returns:
        - New stop level
        """
        entry_price = position.entry_price
        direction = position.direction  # 'long' or 'short'
        current_stop = position.current_stop

        # Calculate price movement since entry
        if direction == 'long':
            price_movement = (current_price - entry_price) / entry_price
            # Check if price has moved enough to activate trailing stop
            if price_movement >= self.activation_percent:
                # Calculate new stop level
                potential_stop = current_price * (1 - self.step_size)
                # Only move stop upward
                if potential_stop > current_stop:
                    return potential_stop
        else:  # Short position
            price_movement = (entry_price - current_price) / entry_price
            # Check if price has moved enough to activate trailing stop
            if price_movement >= self.activation_percent:
                # Calculate new stop level
                potential_stop = current_price * (1 + self.step_size)
                # Only move stop downward
                if potential_stop < current_stop:
                    return potential_stop

        # If conditions not met, keep current stop
        return current_stop
