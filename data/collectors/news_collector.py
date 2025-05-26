# data/collectors/news_collector.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import logging
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class NewsCollector:
    """
    Collect and analyze financial news for trading insights
    """

    def __init__(self, config):
        self.config = config
        self.news_config = config.get('news', {})
        self.api_key = self.news_config.get('api_key')

        # Set up logging
        self.logger = logging.getLogger('news_collector')
        self.logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Initialize sentiment analyzer
        self.sentiment_model = self.news_config.get('sentiment_model', 'vader')

        if self.sentiment_model == 'vader':
            # Download VADER lexicon if not already downloaded
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')

            self.analyzer = SentimentIntensityAnalyzer()

        elif self.sentiment_model == 'finbert':
            # Use FinBERT model
            try:
                model_name = "ProsusAI/finbert"
                self.analyzer = pipeline("sentiment-analysis", model=model_name)
            except Exception as e:
                self.logger.error(f"Error loading FinBERT model: {e}")
                # Fallback to VADER
                self.sentiment_model = 'vader'
                self.analyzer = SentimentIntensityAnalyzer()

        # Cache for news data
        self.news_cache = {}
        self.cache_dir = self.news_config.get('cache_dir', 'cache/news')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load cached data if available
        self._load_cache()

        self.logger.info(f"News collector initialized with {self.sentiment_model} sentiment model")

    def collect_news(self, instruments=None, lookback_days=1):
        """
        Collect news for specified instruments

        Parameters:
        - instruments: List of instruments to collect news for
        - lookback_days: Number of days to look back

        Returns:
        - Dictionary of news data by instrument
        """
        # Use configured instruments if none specified
        if instruments is None:
            instruments = self.config.get('data', {}).get('instruments', [])

        # Convert instrument format if needed (e.g., EUR_USD -> EURUSD)
        formatted_instruments = [instr.replace('_', '') for instr in instruments]

        # Prepare date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # Check cache first
        cache_key = f"{','.join(formatted_instruments)}_{from_date}_{to_date}"

        if cache_key in self.news_cache and not self.news_config.get('force_refresh', False):
            self.logger.info(f"Using cached news data for {cache_key}")
            return self.news_cache[cache_key]

        # Collect news using configured source
        news_source = self.news_config.get('source', 'newsapi')

        if news_source == 'newsapi':
            news_data = self._collect_from_newsapi(formatted_instruments, from_date, to_date)
        elif news_source == 'fxstreet':
            news_data = self._collect_from_fxstreet(formatted_instruments, from_date, to_date)
        elif news_source == 'forexfactory':
            news_data = self._collect_from_forexfactory(formatted_instruments, from_date, to_date)
        else:
            # Default to sample data if source not recognized
            news_data = self._generate_sample_news(formatted_instruments, from_date, to_date)

        # Analyze sentiment
        news_data = self._analyze_sentiment(news_data)

        # Cache results
        self.news_cache[cache_key] = news_data
        self._save_cache()

        return news_data

    def _collect_from_newsapi(self, instruments, from_date, to_date):
        """Collect news from News API"""
        if not self.api_key:
            self.logger.warning("News API key not configured, using sample data")
            return self._generate_sample_news(instruments, from_date, to_date)

        # News data by instrument
        news_data = {instr: [] for instr in instruments}

        try:
            base_url = "https://newsapi.org/v2/everything"

            # Use currency names for better results
            currency_map = {
                'EUR': 'Euro',
                'USD': 'Dollar',
                'GBP': 'Pound',
                'JPY': 'Yen',
                'AUD': 'Australian Dollar',
                'NZD': 'New Zealand Dollar',
                'CAD': 'Canadian Dollar',
                'CHF': 'Swiss Franc'
            }

            # Generate search queries for each instrument
            for instrument in instruments:
                # Extract currencies from instrument
                if len(instrument) == 6:
                    base_currency = instrument[:3]
                    quote_currency = instrument[3:]

                    # Use currency names if available
                    base_name = currency_map.get(base_currency, base_currency)
                    quote_name = currency_map.get(quote_currency, quote_currency)

                    # Create search query
                    query = f"({base_name} AND {quote_name}) OR {instrument}"
                else:
                    # Use instrument as is
                    query = instrument

                # Add forex keyword for more relevant results
                query = f"{query} AND (forex OR currency OR exchange rate)"

                # Call News API
                params = {
                    'q': query,
                    'from': from_date,
                    'to': to_date,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'apiKey': self.api_key
                }

                response = requests.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()

                    # Process articles
                    for article in data.get('articles', []):
                        news_item = {
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'instrument': instrument
                        }

                        news_data[instrument].append(news_item)

                    self.logger.info(f"Collected {len(data.get('articles', []))} news items for {instrument}")

                else:
                    self.logger.error(f"Error fetching news from News API: {response.status_code} - {response.text}")

                # Respect API rate limits
                time.sleep(1)

        except Exception as e:
            self.logger.error(f"Error collecting news from News API: {e}")

        return news_data

    def _collect_from_fxstreet(self, instruments, from_date, to_date):
        """Collect news from FXStreet (requires web scraping)"""
        # This would require implementing a web scraper for FXStreet
        # For demonstration, we'll use sample data
        self.logger.warning("FXStreet scraping not implemented, using sample data")
        return self._generate_sample_news(instruments, from_date, to_date)

    def _collect_from_forexfactory(self, instruments, from_date, to_date):
        """Collect news from Forex Factory (requires web scraping)"""
        # This would require implementing a web scraper for Forex Factory
        # For demonstration, we'll use sample data
        self.logger.warning("Forex Factory scraping not implemented, using sample data")
        return self._generate_sample_news(instruments, from_date, to_date)

    def _generate_sample_news(self, instruments, from_date, to_date):
        """Generate sample news data for testing"""
        # Convert dates to datetime objects
        start_date = datetime.fromisoformat(from_date)
        end_date = datetime.fromisoformat(to_date)

        # Calculate number of days
        days = (end_date - start_date).days + 1

        # News data by instrument
        news_data = {instr: [] for instr in instruments}

        # Sample news templates
        positive_templates = [
            "{base} strengthens against {quote} as economic data exceeds expectations",
            "{base} rallies on positive {base_country} economic outlook",
            "Bullish trend continues for {pair} amid market optimism",
            "{base_country} central bank hints at rate hike, boosting {base}",
            "Analysts predict {pair} will continue to rise in coming weeks"
        ]

        negative_templates = [
            "{base} weakens against {quote} following disappointing economic data",
            "{base} declines as {base_country} inflation concerns rise",
            "Bearish sentiment dominates {pair} trading session",
            "{base_country} political uncertainty weighs on {base}",
            "Analysts forecast continued downward pressure on {pair}"
        ]

        neutral_templates = [
            "{pair} trades in narrow range ahead of key data release",
            "Market participants await central bank decision on {pair}",
            "Mixed signals for {pair} as traders assess economic indicators",
            "{base_country} and {quote_country} data show conflicting trends for {pair}",
            "Volatility expected to increase for {pair} in coming sessions"
        ]

        # Currency information
        currency_info = {
            'EUR': {'country': 'European', 'name': 'Euro'},
            'USD': {'country': 'US', 'name': 'US Dollar'},
            'GBP': {'country': 'UK', 'name': 'British Pound'},
            'JPY': {'country': 'Japanese', 'name': 'Japanese Yen'},
            'AUD': {'country': 'Australian', 'name': 'Australian Dollar'},
            'NZD': {'country': 'New Zealand', 'name': 'New Zealand Dollar'},
            'CAD': {'country': 'Canadian', 'name': 'Canadian Dollar'},
            'CHF': {'country': 'Swiss', 'name': 'Swiss Franc'}
        }

        # Generate random news for each instrument
        for instrument in instruments:
            # Extract currencies
            base_currency = instrument[:3]
            quote_currency = instrument[3:] if len(instrument) >= 6 else "USD"

            # Get currency info
            base_info = currency_info.get(base_currency, {'country': base_currency, 'name': base_currency})
            quote_info = currency_info.get(quote_currency, {'country': quote_currency, 'name': quote_currency})

            # Format pair name
            pair = f"{base_currency}/{quote_currency}"

            # Generate random news items
            num_news = np.random.randint(3, 10)  # Random number of news items

            for _ in range(num_news):
                # Random date within the range
                days_offset = np.random.randint(0, max(1, days))
                news_date = start_date + timedelta(days=days_offset)

                # Random time
                hour = np.random.randint(0, 24)
                minute = np.random.randint(0, 60)
                news_date = news_date.replace(hour=hour, minute=minute)

                # Random sentiment
                sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.4, 0.2])

                # Select template based on sentiment
                if sentiment == 'positive':
                    template = np.random.choice(positive_templates)
                elif sentiment == 'negative':
                    template = np.random.choice(negative_templates)
                else:
                    template = np.random.choice(neutral_templates)

                # Format title
                title = template.format(
                    base=base_info['name'],
                    quote=quote_info['name'],
                    pair=pair,
                    base_country=base_info['country'],
                    quote_country=quote_info['country']
                )

                # Generate realistic content
                content = f"{title}. Market analysts at several major banks have revised their forecasts for the {pair} pair following recent economic developments. "
                content += f"The {base_info['country']} economy has shown signs of {'strengthening' if sentiment == 'positive' else 'weakening' if sentiment == 'negative' else 'mixed performance'}, "
                content += f"while {quote_info['country']} data has {'underperformed expectations' if sentiment == 'positive' else 'exceeded forecasts' if sentiment == 'negative' else 'met expectations'}. "
                content += f"Technical indicators suggest the pair may {'continue its upward trend' if sentiment == 'positive' else 'extend recent losses' if sentiment == 'negative' else 'trade sideways in the near term'}."

                # Create news item
                news_item = {
                    'title': title,
                    'description': f"Summary of recent {pair} market activity and analyst forecasts.",
                    'content': content,
                    'source': np.random.choice(['Market News Int', 'Forex Daily', 'Currency Focus', 'Trading Insights']),
                    'url': f"https://example.com/news/{instrument.lower()}/{news_date.strftime('%Y%m%d%H%M')}",
                    'published_at': news_date.isoformat(),
                    'instrument': instrument
                }

                news_data[instrument].append(news_item)

        return news_data

    def _analyze_sentiment(self, news_data):
        """Analyze sentiment of news articles"""
        # Process each instrument
        for instrument, articles in news_data.items():
            for article in articles:
                # Combine title and content for analysis
                text = f"{article.get('title', '')} {article.get('content', '')}"

                # Skip empty text
                if not text.strip():
                    article['sentiment'] = {
                        'score': 0,
                        'label': 'neutral',
                        'confidence': 0
                    }
                    continue

                # Analyze using selected model
                if self.sentiment_model == 'vader':
                    # VADER sentiment analysis
                    sentiment = self.analyzer.polarity_scores(text)

                    # Determine sentiment label based on compound score
                    if sentiment['compound'] >= 0.05:
                        label = 'positive'
                    elif sentiment['compound'] <= -0.05:
                        label = 'negative'
                    else:
                        label = 'neutral'

                    # Store results
                    article['sentiment'] = {
                        'score': sentiment['compound'],
                        'label': label,
                        'confidence': abs(sentiment['compound']),
                        'details': {
                            'positive': sentiment['pos'],
                            'negative': sentiment['neg'],
                            'neutral': sentiment['neu']
                        }
                    }

                elif self.sentiment_model == 'finbert':
                    try:
                        # FinBERT sentiment analysis (limit text length for performance)
                        max_length = 512
                        if len(text) > max_length:
                            text = text[:max_length]

                        # Run inference
                        result = self.analyzer(text)[0]

                        # Store results
                        article['sentiment'] = {
                            'score': result['score'] if result['label'] != 'neutral' else 0,
                            'label': result['label'],
                            'confidence': result['score']
                        }

                    except Exception as e:
                        self.logger.error(f"Error running FinBERT sentiment analysis: {e}")

                        # Fallback to neutral sentiment
                        article['sentiment'] = {
                            'score': 0,
                            'label': 'neutral',
                            'confidence': 0
                        }

        return news_data

    def get_sentiment_scores(self, instruments=None, lookback_days=1):
        """
        Get aggregated sentiment scores for instruments

        Parameters:
        - instruments: List of instruments to get sentiment for
        - lookback_days: Number of days to look back

        Returns:
        - Dictionary of sentiment scores by instrument
        """
        # Collect news data
        news_data = self.collect_news(instruments, lookback_days)

        # Aggregate sentiment by instrument
        sentiment_scores = {}

        for instrument, articles in news_data.items():
            if not articles:
                sentiment_scores[instrument] = {
                    'score': 0,
                    'label': 'neutral',
                    'confidence': 0,
                    'article_count': 0
                }
                continue

            # Extract sentiment scores
            scores = [article.get('sentiment', {}).get('score', 0) for article in articles]

            # Weight more recent articles higher
            weights = np.linspace(0.5, 1.0, len(scores))
            weighted_scores = np.array(scores) * weights

            # Calculate weighted average
            avg_score = np.average(weighted_scores) if scores else 0

            # Determine sentiment label
            if avg_score >= 0.05:
                label = 'positive'
            elif avg_score <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            # Calculate confidence as standard deviation of scores
            confidence = 1.0 - min(1.0, np.std(scores)) if len(scores) > 1 else 0.5

            # Store results
            sentiment_scores[instrument] = {
                'score': avg_score,
                'label': label,
                'confidence': confidence,
                'article_count': len(articles)
            }

        return sentiment_scores

    def _load_cache(self):
        """Load news cache from disk"""
        cache_file = os.path.join(self.cache_dir, 'news_cache.json')

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.news_cache = json.load(f)

                self.logger.info(f"Loaded {len(self.news_cache)} news cache entries")

            except Exception as e:
                self.logger.error(f"Error loading news cache: {e}")
                self.news_cache = {}

    def _save_cache(self):
        """Save news cache to disk"""
        cache_file = os.path.join(self.cache_dir, 'news_cache.json')

        try:
            with open(cache_file, 'w') as f:
                json.dump(self.news_cache, f)

            self.logger.info(f"Saved {len(self.news_cache)} news cache entries")

        except Exception as e:
            self.logger.error(f"Error saving news cache: {e}")

    def generate_news_report(self, instruments=None, lookback_days=1):
        """
        Generate a human-readable news report

        Parameters:
        - instruments: List of instruments to include
        - lookback_days: Number of days to look back

        Returns:
        - Formatted report text
        """
        # Collect news data
        news_data = self.collect_news(instruments, lookback_days)

        # Get sentiment scores
        sentiment_scores = self.get_sentiment_scores(instruments, lookback_days)

        # Generate report
        report = []
        report.append("# Financial News Sentiment Report")
        report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Add sentiment summary
        report.append("## Sentiment Summary")

        for instrument, sentiment in sentiment_scores.items():
            sentiment_emoji = "🟢" if sentiment['label'] == 'positive' else "🔴" if sentiment['label'] == 'negative' else "⚪"
            report.append(f"{sentiment_emoji} **{instrument}**: {sentiment['label'].capitalize()} " +
                        f"(Score: {sentiment['score']:.2f}, Confidence: {sentiment['confidence']:.2f}, " +
                        f"Articles: {sentiment['article_count']})")

        # Add recent news by instrument
        report.append("\n## Recent News by Instrument")

        for instrument, articles in news_data.items():
            if not articles:
                continue

            report.append(f"\n### {instrument}")

            # Sort by publication date (newest first)
            sorted_articles = sorted(articles, key=lambda x: x.get('published_at', ''), reverse=True)

            for i, article in enumerate(sorted_articles[:5]):  # Show up to 5 most recent articles
                # Format publication date
                pub_date = article.get('published_at', '')
                try:
                    pub_date = datetime.fromisoformat(pub_date).strftime('%Y-%m-%d %H:%M')
                except:
                    pass

                # Get sentiment emoji
                sentiment = article.get('sentiment', {})
                sentiment_label = sentiment.get('label', 'neutral')
                sentiment_emoji = "🟢" if sentiment_label == 'positive' else "🔴" if sentiment_label == 'negative' else "⚪"

                # Add article
                report.append(f"\n{i+1}. {sentiment_emoji} **{article.get('title', 'No title')}**")
                report.append(f"   *{pub_date} - {article.get('source', 'Unknown Source')}*")
                report.append(f"   {article.get('description', '')}")
                report.append(f"   Sentiment: {sentiment_label.capitalize()} (Score: {sentiment.get('score', 0):.2f})")
                report.append(f"   [Read more]({article.get('url', '#')})")

        # Add trading implications
        report.append("\n## Trading Implications")

        for instrument, sentiment in sentiment_scores.items():
            if sentiment['article_count'] == 0:
                continue

            report.append(f"\n### {instrument}")

            # Generate trading implications based on sentiment
            if sentiment['label'] == 'positive':
                if sentiment['confidence'] > 0.7:
                    report.append("* Strong positive sentiment suggests potential bullish momentum")
                    report.append("* Consider long positions with appropriate risk management")
                    report.append("* Monitor key resistance levels for potential breakouts")
                else:
                    report.append("* Moderately positive sentiment, but with limited conviction")
                    report.append("* Consider selective long entries on technical confirmation")
                    report.append("* Keep position sizes conservative given mixed signals")
            elif sentiment['label'] == 'negative':
                if sentiment['confidence'] > 0.7:
                    report.append("* Strong negative sentiment suggests potential bearish momentum")
                    report.append("* Consider short positions with appropriate risk management")
                    report.append("* Monitor key support levels for potential breakdowns")
                else:
                    report.append("* Moderately negative sentiment, but with limited conviction")
                    report.append("* Consider selective short entries on technical confirmation")
                    report.append("* Keep position sizes conservative given mixed signals")
            else:
                report.append("* Neutral sentiment suggests range-bound trading conditions")
                report.append("* Consider range trading strategies or reduced position sizing")
                report.append("* Wait for clearer directional signals before establishing positions")

        return "\n".join(report)
