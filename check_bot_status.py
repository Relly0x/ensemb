# check_bot_status.py

"""
Comprehensive checker to see which Python bot is running and its version
"""

import os
import sys
import psutil
import MetaTrader5 as mt5
import json
from datetime import datetime
import subprocess


def check_running_python_processes():
    """Check all running Python processes"""
    print("🔍 Checking Running Python Processes")
    print("=" * 50)

    python_processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1:
                    script_name = cmdline[1] if len(cmdline) > 1 else "Unknown"

                    # Check if it's a trading bot
                    is_trading_bot = any(keyword in ' '.join(cmdline).lower() for keyword in
                                         ['mt5', 'trading', 'bot', 'tft', 'telegram'])

                    if is_trading_bot or 'trading' in script_name.lower() or 'bot' in script_name.lower():
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'script': script_name,
                            'cmdline': ' '.join(cmdline),
                            'start_time': datetime.fromtimestamp(proc.info['create_time']),
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'is_bot': is_trading_bot
                        })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if python_processes:
        print(f"Found {len(python_processes)} potential trading bot processes:")
        for i, proc in enumerate(python_processes, 1):
            status = "🤖 TRADING BOT" if proc['is_bot'] else "🐍 Python Process"
            print(f"\n{i}. {status}")
            print(f"   PID: {proc['pid']}")
            print(f"   Script: {os.path.basename(proc['script'])}")
            print(f"   Started: {proc['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   CPU: {proc['cpu_percent']:.1f}%")
            print(f"   Memory: {proc['memory_mb']:.1f} MB")
            print(f"   Command: {proc['cmdline'][:100]}...")
    else:
        print("❌ No trading bot processes found")

    return python_processes


def check_mt5_connection():
    """Check MT5 connection and bot activity"""
    print("\n🔌 Checking MT5 Connection")
    print("=" * 50)

    if not mt5.initialize():
        print("❌ Cannot connect to MT5")
        return None

    try:
        # Account info
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Connected to MT5")
            print(f"   Account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Company: {account_info.company}")
            print(f"   Balance: {account_info.balance:.2f} {account_info.currency}")

        # Check for bot positions (magic number 123456)
        positions = mt5.positions_get()
        if positions:
            bot_positions = [p for p in positions if p.magic == 123456]
            print(f"\n📊 Total Positions: {len(positions)}")
            print(f"🤖 Bot Positions: {len(bot_positions)}")

            if bot_positions:
                print("\nBot Position Details:")
                for i, pos in enumerate(bot_positions, 1):
                    direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                    print(f"   {i}. {pos.symbol} {direction} {pos.volume} lots")
                    print(f"      Entry: {pos.price_open:.5f} | Current: {pos.price_current:.5f}")
                    print(f"      P&L: {pos.profit:+.2f} | Comment: {pos.comment}")
        else:
            print("📊 No open positions")

        # Check recent bot trades
        from datetime import timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        deals = mt5.history_deals_get(start_time, end_time)
        if deals:
            bot_deals = [d for d in deals if d.magic == 123456]
            print(f"\n📈 Bot Trades (Last 24h): {len(bot_deals)}")

            if bot_deals:
                print("Recent Bot Trades:")
                for deal in bot_deals[-3:]:  # Show last 3
                    deal_time = datetime.fromtimestamp(deal.time)
                    deal_type = "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL"
                    print(f"   {deal.symbol} {deal_type} {deal.volume} at {deal.price:.5f}")
                    print(f"   Time: {deal_time.strftime('%Y-%m-%d %H:%M:%S')} | P&L: {deal.profit:+.2f}")

        return account_info

    except Exception as e:
        print(f"❌ Error checking MT5: {e}")
        return None
    finally:
        mt5.shutdown()


def check_bot_file_versions():
    """Check modification times of bot files to see which version is running"""
    print("\n📁 Checking Bot File Versions")
    print("=" * 50)

    # Common bot file names
    bot_files = [
        'mt5_trading_bot.py',
        'integrated_mt5_telegram_bot.py',
        'run_integrated_bot.py',
        'main.py'
    ]

    found_files = []

    for filename in bot_files:
        if os.path.exists(filename):
            stat = os.stat(filename)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            size_kb = stat.st_size / 1024

            found_files.append({
                'filename': filename,
                'modified': mod_time,
                'size_kb': size_kb
            })

    if found_files:
        # Sort by modification time (newest first)
        found_files.sort(key=lambda x: x['modified'], reverse=True)

        print("Bot files found (newest first):")
        for i, file_info in enumerate(found_files, 1):
            age = datetime.now() - file_info['modified']
            age_str = f"{age.days}d {age.seconds // 3600}h ago" if age.days > 0 else f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"

            indicator = "🟢 NEWEST" if i == 1 else f"📁 File {i}"
            print(f"   {indicator}: {file_info['filename']}")
            print(f"      Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')} ({age_str})")
            print(f"      Size: {file_info['size_kb']:.1f} KB")
    else:
        print("❌ No bot files found in current directory")

    return found_files


def check_config_version():
    """Check config file to see what's configured"""
    print("\n⚙️ Checking Configuration")
    print("=" * 50)

    config_files = ['config/config.json', 'config.json']

    for config_path in config_files:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                print(f"✅ Config found: {config_path}")

                # Check key settings
                if 'telegram' in config:
                    token = config['telegram'].get('token', 'Not set')
                    users = config['telegram'].get('authorized_users', [])
                    print(f"   Telegram: {'✅ Configured' if token != 'Not set' else '❌ No token'}")
                    print(f"   Authorized Users: {len(users)}")

                if 'execution' in config:
                    max_trades = config['execution'].get('max_daily_trades', 'Not set')
                    risk = config['execution'].get('risk_per_trade', 'Not set')
                    print(f"   Max Daily Trades: {max_trades}")
                    print(f"   Risk per Trade: {risk}")

                if 'data' in config:
                    instruments = config['data'].get('instruments', [])
                    print(f"   Instruments: {instruments}")

                # Check modification time
                stat = os.stat(config_path)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                age = datetime.now() - mod_time
                age_str = f"{age.days}d {age.seconds // 3600}h ago" if age.days > 0 else f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"
                print(f"   Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_str})")

                return config

            except Exception as e:
                print(f"❌ Error reading {config_path}: {e}")

    print("❌ No config file found")
    return None


def check_log_files():
    """Check recent log entries to see bot activity"""
    print("\n📋 Checking Log Files")
    print("=" * 50)

    log_dirs = ['logs/', './']
    log_files = []

    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith('.log') and (
                        'trading' in filename.lower() or 'bot' in filename.lower() or 'mt5' in filename.lower()):
                    log_path = os.path.join(log_dir, filename)
                    log_files.append(log_path)

    if log_files:
        # Sort by modification time
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        print(f"Found {len(log_files)} log files:")

        for log_file in log_files[:3]:  # Show top 3 most recent
            try:
                stat = os.stat(log_file)
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                age = datetime.now() - mod_time
                age_str = f"{age.days}d {age.seconds // 3600}h ago" if age.days > 0 else f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"

                print(f"\n   📋 {os.path.basename(log_file)}")
                print(f"      Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_str})")

                # Show last few lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("      Recent entries:")
                        for line in lines[-3:]:
                            print(f"        {line.strip()}")

            except Exception as e:
                print(f"      ❌ Error reading log: {e}")
    else:
        print("❌ No log files found")


def get_current_working_directory_info():
    """Show current directory and Python version"""
    print("\n🐍 Environment Information")
    print("=" * 50)

    print(f"Current Directory: {os.getcwd()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("📦 Running in Virtual Environment")
    else:
        print("🌐 Running in System Python")


def main():
    """Main checker function"""
    print("🔍 Python Trading Bot Status Checker")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check environment
    get_current_working_directory_info()

    # Check running processes
    processes = check_running_python_processes()

    # Check MT5 connection
    mt5_info = check_mt5_connection()

    # Check file versions
    files = check_bot_file_versions()

    # Check config
    config = check_config_version()

    # Check logs
    check_log_files()

    # Summary
    print("\n🎯 SUMMARY")
    print("=" * 50)

    if processes:
        active_bots = [p for p in processes if p['is_bot']]
        if active_bots:
            newest_bot = max(active_bots, key=lambda x: x['start_time'])
            print(f"✅ Active Trading Bot: {os.path.basename(newest_bot['script'])}")
            print(f"   PID: {newest_bot['pid']}")
            print(f"   Running since: {newest_bot['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⚠️ Python processes found but no clear trading bots")
    else:
        print("❌ No trading bot processes detected")

    if mt5_info:
        print(f"✅ MT5 Connected: {mt5_info.server}")
    else:
        print("❌ MT5 Connection Failed")

    if files:
        newest_file = files[0]
        print(f"📁 Newest Bot File: {newest_file['filename']}")
        print(f"   Modified: {newest_file['modified'].strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n💡 Tips:")
    print("- If multiple Python processes are running, you might have duplicate bots")
    print("- Check the newest modified file to see which version should be running")
    print("- Use Telegram commands (/status) if your bot has Telegram integration")
    print("- Check log files for detailed bot activity")


if __name__ == "__main__":
    main()