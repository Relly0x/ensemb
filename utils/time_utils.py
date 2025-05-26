# utils/time_utils.py

from datetime import datetime, time, timedelta
import pytz

class TradingHoursManager:
    def __init__(self, config):
        self.config = config
        self.timezone = pytz.timezone(config['trading_hours']['timezone'])
        self.sessions = config['trading_hours']['sessions']

    def is_trading_time(self):
        """
        Check if current time is within trading hours

        Returns:
        - Boolean indicating if it's trading time
        """
        now = datetime.now(self.timezone)
        current_time = now.time()
        current_day = now.strftime('%A')

        for session in self.sessions:
            # Check if today is in trading days
            if current_day in session['days']:
                # Parse session times
                start_time = datetime.strptime(session['start'], '%H:%M').time()
                end_time = datetime.strptime(session['end'], '%H:%M').time()

                # Check if current time is within session
                if start_time <= current_time <= end_time:
                    return True

        return False

    def next_trading_session(self):
        """
        Get the start time of the next trading session

        Returns:
        - Datetime object for the next session start
        """
        now = datetime.now(self.timezone)
        current_time = now.time()
        current_day = now.strftime('%A')

        # Days of the week in order
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_day_idx = days_of_week.index(current_day)

        # Check all possible sessions starting from current day
        for day_offset in range(7):  # Check all 7 days
            check_day_idx = (current_day_idx + day_offset) % 7
            check_day = days_of_week[check_day_idx]

            for session in self.sessions:
                if check_day in session['days']:
                    start_time = datetime.strptime(session['start'], '%H:%M').time()

                    # If same day, check if start time is in the future
                    if day_offset == 0 and start_time <= current_time:
                        continue

                    # Create datetime for the next session start
                    next_date = (now + timedelta(days=day_offset)).date()
                    next_session = datetime.combine(next_date, start_time)
                    next_session = self.timezone.localize(next_session)

                    return next_session

        return None  # Should not happen with valid configuration
