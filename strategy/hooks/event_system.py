import time
import logging
import threading
import json
import os
from queue import Queue
from datetime import datetime
from functools import wraps


class EventManager:
    """
    Event system for trading bot hooks and notifications
    Implements a publisher-subscriber pattern to decouple components
    """

    def __init__(self, config):
        self.config = config
        self.subscribers = {}  # event_type -> list of callbacks
        self.event_log = []
        self.max_log_size = config.get('event_system', {}).get('max_log_size', 1000)
        self.event_lock = threading.Lock()
        self.queue = Queue()
        self.running = False
        self.event_thread = None

        # Setup logger
        self.logger = logging.getLogger('event_system')

        # Create log directory if needed
        self.log_dir = config.get('event_system', {}).get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # Start event processing thread
        self.start()

        self.logger.info("Event system initialized")

    def start(self):
        """Start the event processing thread"""
        if self.running:
            return

        self.running = True
        self.event_thread = threading.Thread(target=self._process_events, daemon=True)
        self.event_thread.start()

    def stop(self):
        """Stop the event processing thread"""
        self.running = False
        if self.event_thread:
            self.event_thread.join(timeout=1.0)
            self.event_thread = None

    def subscribe(self, event_type, callback):
        """
        Subscribe a callback function to an event type

        Parameters:
        - event_type: String identifier for the event
        - callback: Function to call when event occurs

        Returns:
        - Subscription ID
        """
        with self.event_lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []

            subscription_id = f"{event_type}:{len(self.subscribers[event_type])}"
            self.subscribers[event_type].append((subscription_id, callback))

            self.logger.debug(f"Added subscription {subscription_id} to {event_type}")

            return subscription_id

    def unsubscribe(self, subscription_id):
        """
        Unsubscribe from an event using subscription ID

        Parameters:
        - subscription_id: ID returned from subscribe()

        Returns:
        - True if successful, False otherwise
        """
        if not subscription_id or ':' not in subscription_id:
            return False

        event_type, _ = subscription_id.split(':', 1)

        with self.event_lock:
            if event_type in self.subscribers:
                self.subscribers[event_type] = [
                    (sub_id, callback)
                    for sub_id, callback in self.subscribers[event_type]
                    if sub_id != subscription_id
                ]
                self.logger.debug(f"Removed subscription {subscription_id}")
                return True

        return False

    def emit(self, event_type, data=None, source=None):
        """
        Emit an event to all subscribers

        Parameters:
        - event_type: String identifier for the event
        - data: Optional data to pass to subscribers
        - source: Optional string identifying the event source

        Returns:
        - Number of subscribers notified
        """
        timestamp = time.time()

        # Format the event
        event = {
            'type': event_type,
            'data': data,
            'source': source,
            'timestamp': timestamp,
            'time': datetime.fromtimestamp(timestamp).isoformat()
        }

        # Add to processing queue
        self.queue.put(event)

        return self._count_subscribers(event_type)

    def _process_events(self):
        """Process events from the queue (runs in a separate thread)"""
        while self.running:
            try:
                if self.queue.empty():
                    # Sleep for a short time if queue is empty
                    time.sleep(0.01)
                    continue

                event = self.queue.get()

                # Log the event
                self._log_event(event)

                # Notify subscribers
                event_type = event['type']

                with self.event_lock:
                    # Notify specific event subscribers
                    if event_type in self.subscribers:
                        for _, callback in self.subscribers[event_type]:
                            try:
                                callback(event)
                            except Exception as e:
                                self.logger.error(f"Error in event callback: {e}")

                    # Notify wildcard subscribers
                    if '*' in self.subscribers:
                        for _, callback in self.subscribers['*']:
                            try:
                                callback(event)
                            except Exception as e:
                                self.logger.error(f"Error in wildcard callback: {e}")

                # Mark as completed
                self.queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                time.sleep(0.1)  # Prevent high CPU usage in case of repeated errors

    def _log_event(self, event):
        """Add event to log and trim if needed"""
        with self.event_lock:
            # Add to in-memory log
            self.event_log.append(event)

            # Trim log if needed
            if len(self.event_log) > self.max_log_size:
                self.event_log = self.event_log[-self.max_log_size:]

            # Log to file if configured
            if self.config.get('event_system', {}).get('log_to_file', False):
                try:
                    log_file = os.path.join(self.log_dir, f"events_{datetime.now().strftime('%Y%m%d')}.log")
                    with open(log_file, 'a') as f:
                        f.write(json.dumps(event) + '\n')
                except Exception as e:
                    self.logger.error(f"Error writing event to log file: {e}")

    def _count_subscribers(self, event_type):
        """Count the number of subscribers for an event type"""
        with self.event_lock:
            specific = len(self.subscribers.get(event_type, []))
            wildcard = len(self.subscribers.get('*', []))
            return specific + wildcard

    def get_recent_events(self, limit=100, event_type=None):
        """
        Get recent events from the log

        Parameters:
        - limit: Maximum number of events to return
        - event_type: Optional filter by event type

        Returns:
        - List of recent events
        """
        with self.event_lock:
            if event_type:
                filtered = [e for e in self.event_log if e['type'] == event_type]
                return filtered[-limit:]
            else:
                return self.event_log[-limit:]


# Decorator for event handlers
def event_handler(event_manager, event_type):
    """
    Decorator for event handler methods

    Usage:
    @event_handler(event_manager, 'trade:entry')
    def handle_trade_entry(self, event):
        # Handle the event
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, event):
            return func(self, event)

        # Register the handler
        event_manager.subscribe(event_type, lambda event: wrapper(self, event))

        return wrapper

    return decorator