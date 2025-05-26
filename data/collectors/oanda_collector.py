import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging


class OandaDataCollector:
    """
    Data collector for OANDA API
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('oanda_collector')

        # OANDA API setup
        self.oanda_config = config.get('oanda', {})
        self.api_key = self.oanda_config.get('api_key')
        self.account_id = self.oanda_config.get('account_id')
        self.environment = self.oanda_config.get('environment', 'practice')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA API key and account ID must be provided in config")

        # Initialize OANDA API
        try:
            import oandapyV20
            import oandapyV20.endpoints.instruments as instruments

            self.api = oandapyV20.API(
                access_token=self.api_key,
                environment=self.environment
            )
            self.instruments_api = instruments

        except ImportError:
            raise ImportError("oandapyV20 library not installed. Install with: pip install oandapyV20")

        self.logger.info("OANDA Data Collector initialized")

    def collect_training_data(self):
        """
        Collect historical data for all instruments for training

        Returns:
        - Dictionary of dataframes by instrument and timeframe
        """
        instruments = self.config['data']['instruments']
        timeframes = [
            self.config['data']['timeframes']['high'],
            self.config['data']['timeframes']['low']
        ]

        result = {}

        for instrument in instruments:
            result[instrument] = {}

            for timeframe in timeframes:
                self.logger.info(f"Collecting data for {instrument} {timeframe}")

                # Get historical data
                data = self._get_historical_data(instrument, timeframe)

                if data is not None and not data.empty:
                    result[instrument][timeframe] = data
                else:
                    self.logger.warning(f"No data received for {instrument} {timeframe}")

        return result

    def _get_historical_data(self, instrument, timeframe, count=5000):
        """
        Get historical candlestick data for an instrument

        Parameters:
        - instrument: Trading instrument (e.g., 'EUR_USD')
        - timeframe: Timeframe (e.g., 'M5', 'H1')
        - count: Number of candles to retrieve

        Returns:
        - DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to OANDA format
            oanda_granularity = self._convert_timeframe(timeframe)

            # Create request
            params = {
                "granularity": oanda_granularity,
                "count": count
            }

            # Make API request
            request = self.instruments_api.InstrumentsCandles(
                instrument=instrument,
                params=params
            )

            response = self.api.request(request)

            # Convert to DataFrame
            candles = response.get('candles', [])

            if not candles:
                self.logger.warning(f"No candles returned for {instrument}")
                return None

            # Process candle data
            data = []
            for candle in candles:
                if candle['complete']:  # Only use complete candles
                    ohlc = candle['mid']  # Use mid prices
                    data.append({
                        'time': candle['time'],
                        'open': float(ohlc['o']),
                        'high': float(ohlc['h']),
                        'low': float(ohlc['l']),
                        'close': float(ohlc['c']),
                        'volume': int(candle['volume'])
                    })

            if not data:
                self.logger.warning(f"No complete candles for {instrument}")
                return None

            # Create DataFrame
            df = pd.DataFrame(data)

            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            # Sort by time
            df.sort_index(inplace=True)

            self.logger.info(f"Collected {len(df)} candles for {instrument} {timeframe}")

            return df

        except Exception as e:
            self.logger.error(f"Error collecting data for {instrument} {timeframe}: {e}")
            return None

    def _convert_timeframe(self, timeframe):
        """
        Convert timeframe string to OANDA granularity

        Parameters:
        - timeframe: Timeframe string (M1, M5, H1, etc.)

        Returns:
        - OANDA granularity string
        """
        timeframe_map = {
            'M1': 'M1',
            'M5': 'M5',
            'M15': 'M15',
            'M30': 'M30',
            'H1': 'H1',
            'H4': 'H4',
            'D1': 'D',
            'W1': 'W',
            'MN1': 'M'
        }

        return timeframe_map.get(timeframe, 'M5')

    def get_current_prices(self):
        """
        Get current prices for all configured instruments

        Returns:
        - Dictionary of current prices by instrument
        """
        instruments = self.config['data']['instruments']
        prices = {}

        try:
            import oandapyV20.endpoints.pricing as pricing

            # Get pricing for all instruments
            params = {
                "instruments": ",".join(instruments)
            }

            request = pricing.PricingInfo(
                accountID=self.account_id,
                params=params
            )

            response = self.api.request(request)

            # Extract prices
            for price_info in response.get('prices', []):
                instrument = price_info['instrument']

                # Use mid price
                ask = float(price_info['asks'][0]['price'])
                bid = float(price_info['bids'][0]['price'])
                mid_price = (ask + bid) / 2

                prices[instrument] = mid_price

        except Exception as e:
            self.logger.error(f"Error getting current prices: {e}")

        return prices

    def get_candles(self, instrument, timeframe, count=200):
        """
        Get recent candlestick data

        Parameters:
        - instrument: Trading instrument
        - timeframe: Timeframe
        - count: Number of candles

        Returns:
        - DataFrame with OHLCV data
        """
        return self._get_historical_data(instrument, timeframe, count)

    def get_historical_candles(self, instrument, timeframe, start_date, end_date):
        """
        Get historical data for a date range

        Parameters:
        - instrument: Trading instrument
        - timeframe: Timeframe
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)

        Returns:
        - DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to OANDA format
            oanda_granularity = self._convert_timeframe(timeframe)

            # Format dates for OANDA API
            start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

            # Create request
            params = {
                "granularity": oanda_granularity,
                "from": start_dt,
                "to": end_dt
            }

            # Make API request
            request = self.instruments_api.InstrumentsCandles(
                instrument=instrument,
                params=params
            )

            response = self.api.request(request)

            # Convert to DataFrame
            candles = response.get('candles', [])

            if not candles:
                return None

            # Process candle data
            data = []
            for candle in candles:
                if candle['complete']:
                    ohlc = candle['mid']
                    data.append({
                        'time': candle['time'],
                        'open': float(ohlc['o']),
                        'high': float(ohlc['h']),
                        'low': float(ohlc['l']),
                        'close': float(ohlc['c']),
                        'volume': int(candle['volume'])
                    })

            if not data:
                return None

            # Create DataFrame
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error collecting historical data: {e}")
            return None