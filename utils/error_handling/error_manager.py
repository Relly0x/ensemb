import logging
import traceback
import sys
import os
import time
import json
from datetime import datetime
from functools import wraps
import threading


class ErrorManager:
    """
    Comprehensive error management system for the trading bot

    Features:
    - Error logging and classification
    - Recovery strategies
    - Notification system integration
    - Error rate limiting
    """

    def __init__(self, config):
        self.config = config
        self.error_log = []
        self.max_log_size = config.get('error_handling', {}).get('max_log_size', 1000)
        self.error_lock = threading.Lock()

        # Error counts by type (for rate limiting)
        self.error_counts = {}
        self.error_thresholds = config.get('error_handling', {}).get('thresholds', {
            'critical': 3,  # 3 critical errors trigger emergency stop
            'network': 10,  # 10 network errors trigger reconnection strategy
            'data': 5,  # 5 data errors trigger data refresh
            'execution': 5  # 5 execution errors trigger retry with backoff
        })

        # Setup logger
        self.logger = logging.getLogger('error_manager')

        # Create error log directory if needed
        self.log_dir = config.get('error_handling', {}).get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # Event manager reference (to be set later)
        self.event_manager = None

        self.logger.info("Error manager initialized")

    def set_event_manager(self, event_manager):
        """Set reference to event manager for notifications"""
        self.event_manager = event_manager

    def handle_error(self, error, error_type='general', context=None, notify=True):
        """
        Process and handle an error

        Parameters:
        - error: The exception or error string
        - error_type: Classification of error
        - context: Additional context information
        - notify: Whether to emit event notification

        Returns:
        - Error log entry
        """
        # Format the error
        error_info = self._format_error(error, error_type, context)

        # Log the error
        self._log_error(error_info)

        # Update error count
        self._increment_error_count(error_type)

        # Check if threshold exceeded
        self._check_thresholds(error_type)

        # Notify if requested and event manager is available
        if notify and self.event_manager:
            self.event_manager.emit('system:error', error_info, source='error_manager')

        return error_info

    def _format_error(self, error, error_type, context):
        """Format error information"""
        # Get exception info
        if isinstance(error, Exception):
            error_message = str(error)
            error_traceback = traceback.format_exception(
                type(error), error, error.__traceback__)
        else:
            error_message = str(error)
            error_traceback = traceback.format_stack()

        # Create error info
        timestamp = time.time()
        return {
            'type': error_type,
            'message': error_message,
            'traceback': error_traceback,
            'context': context or {},
            'timestamp': timestamp,
            'time': datetime.fromtimestamp(timestamp).isoformat()
        }

    def _log_error(self, error_info):
        """Add error to log and log file"""
        with self.error_lock:
            # Add to in-memory log
            self.error_log.append(error_info)

            # Trim log if needed
            if len(self.error_log) > self.max_log_size:
                self.error_log = self.error_log[-self.max_log_size:]

            # Log to console
            self.logger.error(f"Error ({error_info['type']}): {error_info['message']}")

            # Log to file
            try:
                log_file = os.path.join(self.log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        'time': error_info['time'],
                        'type': error_info['type'],
                        'message': error_info['message'],
                        'context': error_info['context']
                    }) + '\n')

                    # Write traceback separately for better readability
                    if isinstance(error_info['traceback'], list):
                        f.write('Traceback:\n' + ''.join(error_info['traceback']) + '\n')
                    else:
                        f.write(f"Traceback: {error_info['traceback']}\n")

                    f.write('-' * 80 + '\n')
            except Exception as e:
                self.logger.error(f"Error writing to log file: {e}")

    def _increment_error_count(self, error_type):
        """Update error count for threshold checking"""
        with self.error_lock:
            # Initialize if not exists
            if error_type not in self.error_counts:
                self.error_counts[error_type] = {
                    'count': 0,
                    'window_start': time.time()
                }

            # Update count
            self.error_counts[error_type]['count'] += 1

            # Reset window if needed (default 1 hour window)
            window_size = self.config.get('error_handling', {}).get('window_size', 3600)
            current_time = time.time()
            if current_time - self.error_counts[error_type]['window_start'] > window_size:
                self.error_counts[error_type] = {
                    'count': 1,  # Count the current error
                    'window_start': current_time
                }

    def _check_thresholds(self, error_type):
        """Check if error thresholds have been exceeded"""
        with self.error_lock:
            if error_type not in self.error_counts or error_type not in self.error_thresholds:
                return False

            count = self.error_counts[error_type]['count']
            threshold = self.error_thresholds[error_type]

            if count >= threshold:
                # Execute recovery strategy
                self._execute_recovery_strategy(error_type, count)
                return True

            return False

    def _execute_recovery_strategy(self, error_type, count):
        """Execute appropriate recovery strategy based on error type"""
        self.logger.warning(f"Error threshold exceeded for {error_type} ({count} errors). Executing recovery strategy.")

        # Notify about threshold breach
        if self.event_manager:
            self.event_manager.emit('system:error_threshold', {
                'error_type': error_type,
                'count': count,
                'threshold': self.error_thresholds.get(error_type)
            }, source='error_manager')

        # Execute specific recovery strategies
        if error_type == 'critical':
            self._handle_critical_errors()
        elif error_type == 'network':
            self._handle_network_errors()
        elif error_type == 'data':
            self._handle_data_errors()
        elif error_type == 'execution':
            self._handle_execution_errors()
        else:
            # Generic recovery for other error types
            self.logger.info(f"No specific recovery strategy for {error_type} errors.")

    def _handle_critical_errors(self):
        """Handle critical errors - emergency stop"""
        self.logger.critical("CRITICAL ERROR THRESHOLD EXCEEDED - EMERGENCY STOP")

        # Emit emergency stop event
        if self.event_manager:
            self.event_manager.emit('system:emergency_stop', {
                'reason': 'Critical error threshold exceeded',
                'error_count': self.error_counts.get('critical', {}).get('count', 0)
            }, source='error_manager')

        # Log the emergency stop
        try:
            with open(os.path.join(self.log_dir, 'emergency_stop.log'), 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Emergency stop due to critical errors\n")
                f.write(f"Error count: {self.error_counts.get('critical', {}).get('count', 0)}\n")
                f.write('-' * 80 + '\n')
        except Exception as e:
            self.logger.error(f"Error writing emergency stop log: {e}")

    def _handle_network_errors(self):
        """Handle network errors - reconnection strategy"""
        self.logger.warning("Network error threshold exceeded - attempting reconnection")

        # Emit reconnect event
        if self.event_manager:
            self.event_manager.emit('system:reconnect', {
                'reason': 'Network error threshold exceeded',
                'error_count': self.error_counts.get('network', {}).get('count', 0)
            }, source='error_manager')

    def _handle_data_errors(self):
        """Handle data errors - refresh data"""
        self.logger.warning("Data error threshold exceeded - refreshing data")

        # Emit data refresh event
        if self.event_manager:
            self.event_manager.emit('system:refresh_data', {
                'reason': 'Data error threshold exceeded',
                'error_count': self.error_counts.get('data', {}).get('count', 0)
            }, source='error_manager')

    def _handle_execution_errors(self):
        """Handle execution errors - retry with backoff"""
        self.logger.warning("Execution error threshold exceeded - retry with backoff")

        # Emit retry event
        if self.event_manager:
            self.event_manager.emit('system:retry_execution', {
                'reason': 'Execution error threshold exceeded',
                'error_count': self.error_counts.get('execution', {}).get('count', 0)
            }, source='error_manager')

    def get_recent_errors(self, limit=100, error_type=None):
        """
        Get recent errors from the log

        Parameters:
        - limit: Maximum number of errors to return
        - error_type: Optional filter by error type

        Returns:
        - List of recent errors
        """
        with self.error_lock:
            if error_type:
                filtered = [e for e in self.error_log if e['type'] == error_type]
                return filtered[-limit:]
            else:
                return self.error_log[-limit:]

    def get_error_stats(self):
        """Get error statistics"""
        with self.error_lock:
            stats = {
                'total_errors': len(self.error_log),
                'counts_by_type': {},
                'thresholds': self.error_thresholds
            }

            # Count errors by type
            for error in self.error_log:
                error_type = error['type']
                if error_type not in stats['counts_by_type']:
                    stats['counts_by_type'][error_type] = 0
                stats['counts_by_type'][error_type] += 1

            return stats


# Error handling decorator
def handle_errors(error_type='general', notify=True, reraise=False, max_retries=0, retry_delay=1.0):
    """
    Decorator for automatic error handling

    Parameters:
    - error_type: Classification of error
    - notify: Whether to emit event notification
    - reraise: Whether to re-raise the exception after handling
    - max_retries: Maximum number of retry attempts (0 = no retries)
    - retry_delay: Delay between retries (in seconds)

    Usage:
    @handle_errors(error_type='network', max_retries=3)
    def fetch_data(self):
        # Function that might raise network errors
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find error manager instance
            error_manager = None
            for arg in args:
                if hasattr(arg, 'error_manager'):
                    error_manager = arg.error_manager
                    break

            if error_manager is None:
                # Fall back to less precise method if no error manager found
                for arg in args:
                    if isinstance(arg, object) and hasattr(arg, '__dict__'):
                        for attr_name, attr_value in arg.__dict__.items():
                            if isinstance(attr_value, ErrorManager):
                                error_manager = attr_value
                                break
                        if error_manager:
                            break

            if error_manager is None:
                # If no error manager found, just try to execute the function
                return func(*args, **kwargs)

            # Try to execute with retries
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'args': str(args),
                        'retries': retries
                    }

                    # Handle the error
                    error_manager.handle_error(e, error_type, context, notify)

                    # Check if we should retry
                    if retries < max_retries:
                        retries += 1
                        time.sleep(retry_delay * (2 ** (retries - 1)))  # Exponential backoff
                        continue

                    # Re-raise if requested
                    if reraise:
                        raise

                    # Return default values based on function's return annotation
                    return_type = func.__annotations__.get('return')
                    if return_type == bool:
                        return False
                    elif return_type == int:
                        return 0
                    elif return_type == float:
                        return 0.0
                    elif return_type == str:
                        return ""
                    elif return_type == list:
                        return []
                    elif return_type == dict:
                        return {}
                    elif return_type == tuple:
                        return tuple()
                    else:
                        return None

        return wrapper

    return decorator