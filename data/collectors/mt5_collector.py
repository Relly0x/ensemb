import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MT5DataCollector:
    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.initialize()

    def initialize(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        self.initialized = True
        return True

    def collect_training_data(self):
        """Collect historical data for all instruments for training"""
        if not self.initialized and not self.initialize():
            return {}

        instruments = self.config['data']['instruments']
        timeframes = [
            self.config['data']['timeframes']['high'],
            self.config['data']['timeframes']['low']
        ]

        result = {}

        for instrument in instruments:
            result[instrument] = {}

            for timeframe in timeframes:
                # Convert timeframe to MT5 format
                mt5_timeframe = self._convert_timeframe(timeframe)

                # Calculate date range
                end_date = datetime.now()
                history_period = self.config['data']['history_period']

                if history_period.endswith('Y'):
                    start_date = end_date - timedelta(days=365 * int(history_period[:-1]))
                elif history_period.endswith('M'):
                    start_date = end_date - timedelta(days=30 * int(history_period[:-1]))
                else:
                    start_date = end_date - timedelta(days=365)  # Default 1 year

                # Request data from MT5
                mt5_data = mt5.copy_rates_range(
                    instrument,
                    mt5_timeframe,
                    start_date,
                    end_date
                )

                if mt5_data is None or len(mt5_data) == 0:
                    print(f"No data for {instrument} {timeframe}")
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(mt5_data)

                # Convert time to datetime
                df['time'] = pd.to_datetime(df['time'], unit='s')

                # Set time as index
                df.set_index('time', inplace=True)

                # Rename columns to match our format
                df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'tick_volume': 'volume'
                }, inplace=True)

                result[instrument][timeframe] = df

        return result

    def _convert_timeframe(self, timeframe):
        """Convert timeframe string to MT5 timeframe constant"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        return timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)

    def __del__(self):
        """Clean up MT5 connection"""
        mt5.shutdown()