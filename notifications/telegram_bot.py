# telegram_bot_fixed.py

import asyncio
import logging
import json
import os
import time
import threading
from datetime import datetime
import MetaTrader5 as mt5

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        CallbackQueryHandler,
        ContextTypes,
        MessageHandler,
        filters
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ python-telegram-bot not installed. Install with: pip install python-telegram-bot")


class SimpleTelegramBot:
    """
    Simplified Telegram bot that actually works
    """

    def __init__(self, config, trading_bot=None):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot library is not installed")

        self.config = config
        self.trading_bot = trading_bot
        self.telegram_config = config.get('telegram', {})
        self.token = self.telegram_config.get('token')

        if not self.token:
            raise ValueError("Telegram bot token not found in config")

        self.authorized_users = set(str(user) for user in self.telegram_config.get('authorized_users', []))
        self.admin_users = set(str(user) for user in self.telegram_config.get('admin_users', []))

        # Set up logging
        self.logger = logging.getLogger('telegram_bot')

        # Bot state
        self.app = None
        self.is_running = False
        self.loop = None
        self.bot_thread = None

        self.logger.info("Telegram bot initialized")

    def _is_authorized(self, user_id):
        """Check if user is authorized"""
        return str(user_id) in self.authorized_users

    def _is_admin(self, user_id):
        """Check if user is admin"""
        return str(user_id) in self.admin_users

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        self.logger.info(f"Received /start from user {user_id} ({username})")

        if not self._is_authorized(user_id):
            await update.message.reply_text(
                "❌ You are not authorized to use this bot.\n"
                f"Your User ID: {user_id}\n"
                "Please contact the administrator to request access."
            )
            self.logger.warning(f"Unauthorized access attempt from {user_id} ({username})")
            return

        # Get bot status
        bot_status = "🟢 Running" if self.trading_bot and getattr(self.trading_bot, 'is_running',
                                                                 False) else "🔴 Stopped"

        welcome_msg = (
            f"🤖 **MT5 Trading Bot Control Panel**\n\n"
            f"Status: {bot_status}\n"
            f"User: {username} ({user_id})\n\n"
            f"**Available Commands:**\n"
            f"/status - Get bot status\n"
            f"/positions - View open positions\n"
            f"/help - Show this message\n\n"
            f"Bot will send automatic notifications for:\n"
            f"• Trade entries and exits\n"
            f"• Important alerts\n"
            f"• System status updates"
        )

        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("📈 Positions", callback_data="positions")
            ],
            [
                InlineKeyboardButton("ℹ️ Help", callback_data="help")
            ]
        ]

        # Add admin controls if user is admin
        if self._is_admin(user_id):
            admin_row = [
                InlineKeyboardButton("▶️ Start Bot", callback_data="start_bot"),
                InlineKeyboardButton("⏹️ Stop Bot", callback_data="stop_bot")
            ]
            keyboard.insert(-1, admin_row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_msg, parse_mode='Markdown', reply_markup=reply_markup)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = update.effective_user.id

        if not self._is_authorized(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        # Get MT5 account info
        try:
            # Initialize MT5 if not already done
            if not mt5.initialize():
                status_msg = "❌ Cannot connect to MT5"
            else:
                account_info = mt5.account_info()
                positions = mt5.positions_get()

                if account_info:
                    # Get bot status
                    bot_running = self.trading_bot and getattr(self.trading_bot, 'is_running', False)

                    status_msg = (
                        f"📊 **MT5 Trading Bot Status**\n\n"
                        f"🤖 Bot: {'🟢 Running' if bot_running else '🔴 Stopped'}\n"
                        f"💰 Balance: {account_info.balance:.2f} {account_info.currency}\n"
                        f"📈 Equity: {account_info.equity:.2f} {account_info.currency}\n"
                        f"📊 Margin: {account_info.margin:.2f} {account_info.currency}\n"
                        f"📍 Free Margin: {account_info.margin_free:.2f} {account_info.currency}\n"
                        f"🎯 Open Positions: {len(positions) if positions else 0}\n"
                        f"🏢 Broker: {account_info.company}\n"
                        f"🔗 Server: {account_info.server}\n"
                        f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    status_msg = "❌ Cannot get MT5 account info"

        except Exception as e:
            status_msg = f"❌ Error getting status: {str(e)}"

        # Create refresh button
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="status")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(status_msg, parse_mode='Markdown', reply_markup=reply_markup)

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        user_id = update.effective_user.id

        if not self._is_authorized(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        try:
            if not mt5.initialize():
                await update.message.reply_text("❌ Cannot connect to MT5")
                return

            positions = mt5.positions_get()

            if not positions:
                await update.message.reply_text("📊 No open positions")
                return

            # Filter only bot positions (magic number 123456)
            bot_positions = [p for p in positions if p.magic == 123456]

            if not bot_positions:
                await update.message.reply_text("📊 No open positions from this bot")
                return

            pos_msg = "📊 **Open Positions:**\n\n"

            for i, pos in enumerate(bot_positions, 1):
                direction = "🟢 BUY" if pos.type == mt5.POSITION_TYPE_BUY else "🔴 SELL"
                profit_emoji = "💰" if pos.profit > 0 else "📉" if pos.profit < 0 else "⚪"

                pos_msg += (
                    f"**{i}. {pos.symbol}**\n"
                    f"{direction} {pos.volume} lots\n"
                    f"💵 Entry: {pos.price_open:.5f}\n"
                    f"📊 Current: {pos.price_current:.5f}\n"
                    f"{profit_emoji} P&L: {pos.profit:.2f}\n"
                    f"🛡️ SL: {pos.sl:.5f}\n"
                    f"🎯 TP: {pos.tp:.5f}\n\n"
                )

            # Create refresh button
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="positions")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(pos_msg, parse_mode='Markdown', reply_markup=reply_markup)

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting positions: {str(e)}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        return await self.start_command(update, context)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        user_id = query.from_user.id

        if not self._is_authorized(user_id):
            await query.answer("❌ You are not authorized to use this bot.")
            return

        # Acknowledge the query
        await query.answer()

        callback_data = query.data

        if callback_data == "status":
            # Create a fake update for status command
            fake_update = update
            fake_update.message = query.message
            await self.status_command(fake_update, context)

        elif callback_data == "positions":
            # Create a fake update for positions command
            fake_update = update
            fake_update.message = query.message
            await self.positions_command(fake_update, context)

        elif callback_data == "help":
            # Create a fake update for help command
            fake_update = update
            fake_update.message = query.message
            await self.help_command(fake_update, context)

        elif callback_data == "start_bot":
            if not self._is_admin(user_id):
                await query.edit_message_text("❌ This action is only available to administrators.")
                return

            try:
                if self.trading_bot and hasattr(self.trading_bot, 'start_live_trading'):
                    # Start the trading bot
                    if not getattr(self.trading_bot, 'is_running', False):
                        # Start bot in separate thread
                        bot_thread = threading.Thread(target=self.trading_bot.run, daemon=True)
                        bot_thread.start()
                        await query.edit_message_text("✅ Trading bot started successfully!")
                    else:
                        await query.edit_message_text("ℹ️ Trading bot is already running.")
                else:
                    await query.edit_message_text("❌ Trading bot not available.")

            except Exception as e:
                await query.edit_message_text(f"❌ Error starting bot: {str(e)}")

        elif callback_data == "stop_bot":
            if not self._is_admin(user_id):
                await query.edit_message_text("❌ This action is only available to administrators.")
                return

            # Add confirmation
            keyboard = [
                [
                    InlineKeyboardButton("✅ Yes, stop it", callback_data="confirm_stop"),
                    InlineKeyboardButton("❌ Cancel", callback_data="cancel_stop")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "⚠️ Are you sure you want to stop the trading bot?",
                reply_markup=reply_markup
            )

        elif callback_data == "confirm_stop":
            if not self._is_admin(user_id):
                await query.edit_message_text("❌ This action is only available to administrators.")
                return

            try:
                if self.trading_bot and hasattr(self.trading_bot, 'cleanup'):
                    self.trading_bot.cleanup()
                    await query.edit_message_text("✅ Trading bot stopped successfully!")
                else:
                    await query.edit_message_text("❌ Trading bot not available.")

            except Exception as e:
                await query.edit_message_text(f"❌ Error stopping bot: {str(e)}")

        elif callback_data == "cancel_stop":
            await query.edit_message_text("✅ Operation cancelled.")

        else:
            await query.edit_message_text("❌ Unknown command.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        user_id = update.effective_user.id

        if not self._is_authorized(user_id):
            await update.message.reply_text("❌ You are not authorized to use this bot.")
            return

        message = update.message.text.lower()

        if "status" in message:
            await self.status_command(update, context)
        elif "position" in message:
            await self.positions_command(update, context)
        elif "help" in message:
            await self.help_command(update, context)
        else:
            await update.message.reply_text(
                "ℹ️ I don't understand that command. Type /help for available commands."
            )

    def send_sync_message(self, message):
        """Send message synchronously"""
        if not self.is_running or not self.app:
            return

        async def send_to_users():
            for user_id in self.authorized_users:
                try:
                    await self.app.bot.send_message(
                        chat_id=int(user_id),
                        text=message,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    self.logger.error(f"Error sending message to {user_id}: {e}")

        # Schedule the message sending
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(send_to_users(), self.loop)

    def start_bot(self):
        """Start the Telegram bot"""
        if self.is_running:
            self.logger.info("Telegram bot already running")
            return True

        def run_bot():
            try:
                # Create new event loop for this thread
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

                # Create application
                self.app = Application.builder().token(self.token).build()

                # Add handlers
                self.app.add_handler(CommandHandler("start", self.start_command))
                self.app.add_handler(CommandHandler("status", self.status_command))
                self.app.add_handler(CommandHandler("positions", self.positions_command))
                self.app.add_handler(CommandHandler("help", self.help_command))
                self.app.add_handler(CallbackQueryHandler(self.handle_callback))
                self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

                # Start the bot
                self.is_running = True
                self.logger.info("✅ Telegram bot started successfully")

                # Send startup message
                asyncio.run_coroutine_threadsafe(
                    self._send_startup_message(),
                    self.loop
                )

                # Run the bot
                self.app.run_polling(drop_pending_updates=True)

            except Exception as e:
                self.logger.error(f"Error running Telegram bot: {e}")
                self.is_running = False

        # Start bot in separate thread
        self.bot_thread = threading.Thread(target=run_bot, daemon=True)
        self.bot_thread.start()

        # Wait a moment for startup
        time.sleep(2)
        return self.is_running

    async def _send_startup_message(self):
        """Send startup notification"""
        message = (
            "🤖 **Trading Bot Connected!**\n\n"
            "✅ Telegram interface is now active\n"
            "📱 Type /start to access the control panel\n"
            "🔔 You'll receive notifications for all trades"
        )

        for user_id in self.authorized_users:
            try:
                await self.app.bot.send_message(
                    chat_id=int(user_id),
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                self.logger.error(f"Error sending startup message to {user_id}: {e}")

    def stop_bot(self):
        """Stop the Telegram bot"""
        if not self.is_running:
            return

        self.is_running = False

        if self.app:
            try:
                self.app.stop()
            except:
                pass

        self.logger.info("Telegram bot stopped")


# Test function
def test_telegram_bot():
    """Test the Telegram bot"""

    # Test config
    test_config = {
        "telegram": {
            "token": "7539840895:AAGRLwp6LnABgOCYkg1S-50FftFOOi4WMhk",
            "authorized_users": ["362813632"],
            "admin_users": ["362813632"]
        }
    }

    print("🧪 Testing Telegram bot...")

    try:
        # Create bot
        bot = SimpleTelegramBot(test_config)

        # Start bot
        success = bot.start_bot()

        if success:
            print("✅ Telegram bot started successfully!")
            print("📱 Try sending /start to your bot")
            print("⏹️ Press Ctrl+C to stop")

            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                bot.stop_bot()
                print("✅ Bot stopped")
        else:
            print("❌ Failed to start Telegram bot")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_telegram_bot()