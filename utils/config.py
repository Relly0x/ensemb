# utils/time_utils.py

from datetime import datetime, timedelta
import pytz
from dateutil import parser
import time


class TradingHoursManager:
    """
    Manages trading hours and session times
    """

    def __init__(self, config):
        self.config = config
        self.timezone = pytz.timezone(config['trading_hours']['timezone'])
        self.sessions = config['trading_hours']['sessions']

    def is_trading_time(self):
        """
        Check if current time is within trading hours

        Returns:
        - True if within trading hours, False otherwise
        """
        current_time = datetime.now(self.timezone)
        current_day = current_time.strftime('%A')  # Monday, Tuesday, etc.

        # Check each session
        for session in self.sessions:
            # Check if day matches
            if current_day in session['days']:
                # Parse session times
                start_time = datetime.strptime(session['start'], '%H:%M').time()
                end_time = datetime.strptime(session['end'], '%H:%M').time()

                # Check if current time is within session
                if start_time <= current_time.time() <= end_time:
                    return True

        return False

    def next_trading_session(self):
        """
        Get datetime of next trading session start

        Returns:
        - Datetime of next session start, or None if no sessions configured
        """
        if not self.sessions:
            return None

        current_time = datetime.now(self.timezone)
        current_day = current_time.strftime('%A')
        current_day_idx = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(
            current_day)

        # Check each day forward from current
        for day_offset in range(8):  # Check current day and up to 7 days ahead
            check_day_idx = (current_day_idx + day_offset) % 7
            check_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][check_day_idx]
            check_date = current_time.date() + timedelta(days=day_offset)

            # Find sessions for this day
            for session in self.sessions:
                if check_day in session['days']:
                    # Parse session times
                    start_time = datetime.strptime(session['start'], '%H:%M').time()

                    # Create datetime for session start
                    session_start = datetime.combine(check_date, start_time)
                    session_start = self.timezone.localize(session_start)

                    # If we're checking the current day, make sure the session hasn't started yet
                    if day_offset == 0 and start_time <= current_time.time():
                        continue

                    return session_start

        return None

    def time_until_next_session(self):
        """
        Get time until next trading session

        Returns:
        - Timedelta to next session, or None if no sessions configured
        """
        next_session = self.next_trading_session()

        if next_session:
            current_time = datetime.now(self.timezone)
            return next_session - current_time

        return None


def convert_timezone(dt, from_tz, to_tz):
    """
    Convert datetime from one timezone to another

    Parameters:
    - dt: Datetime object
    - from_tz: Source timezone (string or tzinfo)
    - to_tz: Target timezone (string or tzinfo)

    Returns:
    - Datetime in target timezone
    """
    if isinstance(from_tz, str):
        from_tz = pytz.timezone(from_tz)

    if isinstance(to_tz, str):
        to_tz = pytz.timezone(to_tz)

    # Localize if datetime is naive
    if dt.tzinfo is None:
        dt = from_tz.localize(dt)

    # Convert to target timezone
    return dt.astimezone(to_tz)


def parse_timeframe(timeframe):
    """
    Parse timeframe string to minutes

    Parameters:
    - timeframe: Timeframe string (M1, M5, H1, D1, etc.)

    Returns:
    - Minutes as integer
    """
    if timeframe.startswith('M'):
        return int(timeframe[1:])
    elif timeframe.startswith('H'):
        return int(timeframe[1:]) * 60
    elif timeframe.startswith('D'):
        return int(timeframe[1:]) * 60 * 24
    elif timeframe.startswith('W'):
        return int(timeframe[1:]) * 60 * 24 * 7
    else:
        return 1  # Default to 1 minute