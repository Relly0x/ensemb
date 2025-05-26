import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from sklearn.metrics import confusion_matrix


class PerformanceAnalyzer:
    """
    Analyze trading performance and generate reports
    """

    def __init__(self, trades=None, equity_curve=None, config=None):
        """
        Initialize performance analyzer

        Parameters:
        - trades: List of trade dictionaries
        - equity_curve: List of (timestamp, equity) tuples
        - config: Configuration dictionary
        """
        self.trades = trades or []
        self.equity_curve = equity_curve or []
        self.config = config or {}

        # Cache for metrics
        self.metrics_cache = None
        self.trade_analysis_cache = None

    def load_trades(self, trades):
        """Load trade history"""
        self.trades = trades
        self.metrics_cache = None
        self.trade_analysis_cache = None

    def load_equity_curve(self, equity_curve):
        """Load equity curve data"""
        self.equity_curve = equity_curve
        self.metrics_cache = None

    def load_from_file(self, file_path):
        """Load performance data from a file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if 'trades' in data:
                self.trades = data['trades']

            if 'equity_curve' in data:
                self.equity_curve = data['equity_curve']

            if 'config' in data:
                self.config = data['config']

            self.metrics_cache = None
            self.trade_analysis_cache = None

            return True
        except Exception as e:
            print(f"Error loading performance data: {e}")
            return False

    def calculate_metrics(self, recalculate=False):
        """Calculate performance metrics"""
        if self.metrics_cache is not None and not recalculate:
            return self.metrics_cache

        metrics = {}

        if not self.trades:
            # No trades to analyze
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_profit': 0.0,
                'average_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0
            }
            self.metrics_cache = metrics
            return metrics

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in self.trades if t.get('pnl', 0) <= 0)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        total_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        total_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Average profit/loss
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

        # Risk-reward ratio
        risk_reward_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')

        # Calculate drawdown
        max_drawdown = 0.0
        if self.equity_curve:
            # Convert equity curve to DataFrame for easier analysis
            try:
                if isinstance(self.equity_curve[0][0], str):
                    # Convert timestamp strings to datetime objects
                    df = pd.DataFrame([
                        (pd.to_datetime(timestamp), equity)
                        for timestamp, equity in self.equity_curve
                    ], columns=['timestamp', 'equity'])
                else:
                    df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])

                # Calculate running maximum
                df['peak'] = df['equity'].cummax()

                # Calculate drawdown
                df['drawdown'] = (df['peak'] - df['equity']) / df['peak']

                # Get maximum drawdown
                max_drawdown = df['drawdown'].max()

                # Calculate returns for Sharpe ratio
                df['return'] = df['equity'].pct_change()

                # Calculate Sharpe ratio (annualized)
                returns_mean = df['return'].mean()
                returns_std = df['return'].std()
                sharpe_ratio = np.sqrt(252) * returns_mean / returns_std if returns_std > 0 else 0

                # Calculate total return
                initial_equity = df['equity'].iloc[0]
                final_equity = df['equity'].iloc[-1]
                total_return = (final_equity / initial_equity - 1) * 100
            except Exception as e:
                print(f"Error calculating equity curve metrics: {e}")
                max_drawdown = 0.0
                sharpe_ratio = 0.0
                total_return = 0.0
        else:
            sharpe_ratio = 0.0
            total_return = 0.0

        # Calculate average trade duration
        durations = []
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                try:
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
                    durations.append(duration)
                except:
                    pass

        avg_duration = np.mean(durations) if durations else 0

        # Store metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_profit': avg_profit,
            'average_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'average_duration': avg_duration
        }

        self.metrics_cache = metrics
        return metrics

    def analyze_trades_by_instrument(self, recalculate=False):
        """Analyze trades grouped by instrument"""
        if self.trade_analysis_cache is not None and not recalculate:
            return self.trade_analysis_cache

        if not self.trades:
            return {}

        # Group trades by instrument
        instruments = {}

        for trade in self.trades:
            instrument = trade.get('instrument')
            if not instrument:
                continue

            if instrument not in instruments:
                instruments[instrument] = []

            instruments[instrument].append(trade)

        # Calculate metrics for each instrument
        instrument_metrics = {}

        for instrument, trades in instruments.items():
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losing_trades = sum(1 for t in trades if t.get('pnl', 0) <= 0)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            total_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
            total_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            net_profit = sum(t.get('pnl', 0) for t in trades)

            instrument_metrics[instrument] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'net_profit': net_profit
            }

        self.trade_analysis_cache = instrument_metrics
        return instrument_metrics

    def analyze_trades_by_time(self):
        """Analyze trades by time of day/week"""
        if not self.trades:
            return {}

        # Convert trade times to datetime
        time_analysis = {
            'hour_of_day': {},
            'day_of_week': {}
        }

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for trade in self.trades:
            if 'entry_time' not in trade:
                continue

            try:
                entry_time = pd.to_datetime(trade['entry_time'])

                # Hour of day
                hour = entry_time.hour
                if hour not in time_analysis['hour_of_day']:
                    time_analysis['hour_of_day'][hour] = {
                        'total': 0,
                        'wins': 0,
                        'losses': 0,
                        'profit': 0
                    }

                time_analysis['hour_of_day'][hour]['total'] += 1

                if trade.get('pnl', 0) > 0:
                    time_analysis['hour_of_day'][hour]['wins'] += 1
                else:
                    time_analysis['hour_of_day'][hour]['losses'] += 1

                time_analysis['hour_of_day'][hour]['profit'] += trade.get('pnl', 0)

                # Day of week
                day_of_week = days[entry_time.weekday()]
                if day_of_week not in time_analysis['day_of_week']:
                    time_analysis['day_of_week'][day_of_week] = {
                        'total': 0,
                        'wins': 0,
                        'losses': 0,
                        'profit': 0
                    }

                time_analysis['day_of_week'][day_of_week]['total'] += 1

                if trade.get('pnl', 0) > 0:
                    time_analysis['day_of_week'][day_of_week]['wins'] += 1
                else:
                    time_analysis['day_of_week'][day_of_week]['losses'] += 1

                time_analysis['day_of_week'][day_of_week]['profit'] += trade.get('pnl', 0)

            except:
                continue

        # Calculate win rates
        for hour, stats in time_analysis['hour_of_day'].items():
            stats['win_rate'] = stats['wins'] / stats['total'] if stats['total'] > 0 else 0

        for day, stats in time_analysis['day_of_week'].items():
            stats['win_rate'] = stats['wins'] / stats['total'] if stats['total'] > 0 else 0

        return time_analysis

    def analyze_drawdowns(self):
        """Analyze drawdown periods"""
        if not self.equity_curve:
            return {
                'max_drawdown': 0,
                'drawdown_periods': []
            }

        try:
            # Convert equity curve to DataFrame
            if isinstance(self.equity_curve[0][0], str):
                # Convert timestamp strings to datetime objects
                df = pd.DataFrame([
                    (pd.to_datetime(timestamp), equity)
                    for timestamp, equity in self.equity_curve
                ], columns=['timestamp', 'equity'])
            else:
                df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])

            # Calculate running maximum
            df['peak'] = df['equity'].cummax()

            # Calculate drawdown
            df['drawdown'] = (df['peak'] - df['equity']) / df['peak']

            # Identify drawdown periods
            df['in_drawdown'] = df['drawdown'] > 0

            # Find start and end of drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0

            for i, row in df.iterrows():
                if row['in_drawdown'] and not in_drawdown:
                    # Start of drawdown period
                    in_drawdown = True
                    start_idx = i
                elif not row['in_drawdown'] and in_drawdown:
                    # End of drawdown period
                    in_drawdown = False

                    # Calculate drawdown metrics
                    period_data = df.loc[start_idx:i - 1]
                    max_drawdown = period_data['drawdown'].max()
                    duration = (period_data.iloc[-1]['timestamp'] - period_data.iloc[0][
                        'timestamp']).total_seconds() / 86400  # in days

                    drawdown_periods.append({
                        'start': period_data.iloc[0]['timestamp'].isoformat(),
                        'end': period_data.iloc[-1]['timestamp'].isoformat(),
                        'max_drawdown': max_drawdown,
                        'duration_days': duration
                    })

            # Check if still in drawdown at the end
            if in_drawdown:
                period_data = df.loc[start_idx:]
                max_drawdown = period_data['drawdown'].max()
                duration = (period_data.iloc[-1]['timestamp'] - period_data.iloc[0][
                    'timestamp']).total_seconds() / 86400  # in days

                drawdown_periods.append({
                    'start': period_data.iloc[0]['timestamp'].isoformat(),
                    'end': 'ongoing',
                    'max_drawdown': max_drawdown,
                    'duration_days': duration
                })

            # Sort by max drawdown
            drawdown_periods.sort(key=lambda x: x['max_drawdown'], reverse=True)

            return {
                'max_drawdown': df['drawdown'].max(),
                'drawdown_periods': drawdown_periods[:5]  # Top 5 drawdown periods
            }
        except Exception as e:
            print(f"Error analyzing drawdowns: {e}")
            return {
                'max_drawdown': 0,
                'drawdown_periods': []
            }

    def generate_performance_report(self, output_file=None):
        """Generate a comprehensive performance report"""
        metrics = self.calculate_metrics()
        instrument_analysis = self.analyze_trades_by_instrument()
        time_analysis = self.analyze_trades_by_time()
        drawdown_analysis = self.analyze_drawdowns()

        report = []
        report.append("# Trading Performance Report")
        report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overall metrics
        report.append("## Overall Performance Metrics")
        report.append(f"- Total Trades: {metrics['total_trades']}")
        report.append(f"- Winning Trades: {metrics['winning_trades']} ({metrics['win_rate'] * 100:.1f}%)")
        report.append(f"- Losing Trades: {metrics['losing_trades']}")
        report.append(f"- Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"- Average Profit: {metrics['average_profit']:.2f}")
        report.append(f"- Average Loss: {metrics['average_loss']:.2f}")
        report.append(f"- Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
        report.append(f"- Maximum Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
        report.append(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"- Total Return: {metrics['total_return']:.2f}%")
        report.append(f"- Average Trade Duration: {metrics['average_duration']:.1f} minutes")

        # Performance by instrument
        report.append("\n## Performance by Instrument")

        # Sort instruments by net profit
        sorted_instruments = sorted(
            instrument_analysis.items(),
            key=lambda x: x[1]['net_profit'],
            reverse=True
        )

        for instrument, stats in sorted_instruments:
            report.append(f"\n### {instrument}")
            report.append(f"- Total Trades: {stats['total_trades']}")
            report.append(f"- Winning Trades: {stats['winning_trades']} ({stats['win_rate'] * 100:.1f}%)")
            report.append(f"- Losing Trades: {stats['losing_trades']}")
            report.append(f"- Profit Factor: {stats['profit_factor']:.2f}")
            report.append(f"- Net Profit: {stats['net_profit']:.2f}")

        # Performance by time
        report.append("\n## Performance by Time")

        # Day of week
        report.append("\n### Day of Week")
        for day, stats in time_analysis['day_of_week'].items():
            report.append(
                f"- {day}: {stats['win_rate'] * 100:.1f}% win rate ({stats['wins']}/{stats['total']}), Net Profit: {stats['profit']:.2f}")

        # Hour of day
        report.append("\n### Hour of Day")
        for hour in sorted(time_analysis['hour_of_day'].keys()):
            stats = time_analysis['hour_of_day'][hour]
            report.append(
                f"- {hour:02d}:00: {stats['win_rate'] * 100:.1f}% win rate ({stats['wins']}/{stats['total']}), Net Profit: {stats['profit']:.2f}")

        # Drawdown analysis
        report.append("\n## Drawdown Analysis")
        report.append(f"- Maximum Drawdown: {drawdown_analysis['max_drawdown'] * 100:.2f}%")

        report.append("\n### Top Drawdown Periods")
        for i, period in enumerate(drawdown_analysis['drawdown_periods']):
            report.append(
                f"- Period {i + 1}: {period['max_drawdown'] * 100:.2f}% ({period['start']} to {period['end']}), Duration: {period['duration_days']:.1f} days")

        # Combine report
        report_text = "\n".join(report)

        # Write to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"Performance report saved to {output_file}")
            except Exception as e:
                print(f"Error saving performance report: {e}")

        return report_text

    def plot_equity_curve(self, save_path=None):
        """Plot equity curve"""
        if not self.equity_curve:
            return None

        try:
            # Convert equity curve to DataFrame
            if isinstance(self.equity_curve[0][0], str):
                # Convert timestamp strings to datetime objects
                df = pd.DataFrame([
                    (pd.to_datetime(timestamp), equity)
                    for timestamp, equity in self.equity_curve
                ], columns=['timestamp', 'equity'])
            else:
                df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])

            # Set up plot
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['equity'], 'b-', linewidth=1)
            plt.title('Equity Curve')
            plt.grid(True, alpha=0.3)
            plt.xlabel('Date')
            plt.ylabel('Equity')

            # Format x-axis
            plt.gcf().autofmt_xdate()

            # Add annotations
            plt.annotate(f"Start: ${df['equity'].iloc[0]:.2f}",
                         xy=(df['timestamp'].iloc[0], df['equity'].iloc[0]),
                         xytext=(10, 10),
                         textcoords='offset points')

            plt.annotate(f"End: ${df['equity'].iloc[-1]:.2f}",
                         xy=(df['timestamp'].iloc[-1], df['equity'].iloc[-1]),
                         xytext=(-70, 10),
                         textcoords='offset points')

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path)
                print(f"Equity curve saved to {save_path}")

            # Return figure for display
            fig = plt.gcf()
            plt.close()
            return fig
        except Exception as e:
            print(f"Error plotting equity curve: {e}")
            return None

    def plot_drawdown(self, save_path=None):
        """Plot drawdown over time"""
        if not self.equity_curve:
            return None

        try:
            # Convert equity curve to DataFrame
            if isinstance(self.equity_curve[0][0], str):
                # Convert timestamp strings to datetime objects
                df = pd.DataFrame([
                    (pd.to_datetime(timestamp), equity)
                    for timestamp, equity in self.equity_curve
                ], columns=['timestamp', 'equity'])
            else:
                df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])

            # Calculate drawdown
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100  # as percentage

            # Set up plot
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['drawdown'], 'r-', linewidth=1)
            plt.title('Drawdown Over Time')
            plt.grid(True, alpha=0.3)
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')

            # Invert y-axis for better visualization
            plt.gca().invert_yaxis()

            # Format x-axis
            plt.gcf().autofmt_xdate()

            # Add max drawdown annotation
            max_dd_idx = df['drawdown'].idxmax()
            max_dd = df.loc[max_dd_idx, 'drawdown']
            max_dd_time = df.loc[max_dd_idx, 'timestamp']

            plt.annotate(f"Max Drawdown: {max_dd:.2f}%",
                         xy=(max_dd_time, max_dd),
                         xytext=(10, -20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path)
                print(f"Drawdown plot saved to {save_path}")

            # Return figure for display
            fig = plt.gcf()
            plt.close()
            return fig
        except Exception as e:
            print(f"Error plotting drawdown: {e}")
            return None

    def plot_win_loss_distribution(self, save_path=None):
        """Plot distribution of winning and losing trades"""
        if not self.trades:
            return None

        try:
            # Extract P&L values
            pnl_values = [t.get('pnl', 0) for t in self.trades]

            # Create DataFrame
            df = pd.DataFrame({'pnl': pnl_values})

            # Set up plot
            plt.figure(figsize=(10, 6))

            # Create histogram with KDE
            sns.histplot(data=df, x='pnl', kde=True, color='skyblue')

            # Add vertical line at zero
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)

            # Plot settings
            plt.title('P&L Distribution')
            plt.xlabel('Profit/Loss')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            # Annotations
            win_count = sum(1 for x in pnl_values if x > 0)
            loss_count = sum(1 for x in pnl_values if x <= 0)

            plt.annotate(f"Winning Trades: {win_count}",
                         xy=(0.05, 0.95),
                         xycoords='axes fraction',
                         ha='left',
                         va='top')

            plt.annotate(f"Losing Trades: {loss_count}",
                         xy=(0.05, 0.90),
                         xycoords='axes fraction',
                         ha='left',
                         va='top')

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path)
                print(f"P&L distribution plot saved to {save_path}")

            # Return figure for display
            fig = plt.gcf()
            plt.close()
            return fig
        except Exception as e:
            print(f"Error plotting win/loss distribution: {e}")
            return None