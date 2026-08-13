"""
Telegram Bot Tool - Connect OpenMythos Agent to Telegram
Allows the agent to interact with users via Telegram messenger
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import uuid
from datetime import datetime

from . import BaseTool, ToolResult, TOOLS_REGISTRY
from ..terminal import style as _style

# Optional dependency
try:
    from telegram import Update, Bot, InlineKeyboardMarkup, InlineKeyboardButton
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
    from telegram.constants import ParseMode
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False


@dataclass
class TelegramMessage:
    """Represents a Telegram message"""
    message_id: int
    chat_id: int
    text: str
    user_id: int
    username: Optional[str]
    timestamp: datetime
    message_type: str = "text"  # text, photo, document, etc.
    file_id: Optional[str] = None
    file_path: Optional[str] = None
    reply_to: Optional[int] = None


class TelegramBotTool(BaseTool):
    """
    Telegram Bot integration for OpenMythos Agent
    Enables the agent to receive and send messages via Telegram
    """

    name = "telegram"
    description = "Connect the agent to Telegram messenger. Send messages, receive updates, interact with users in real-time."

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "send_message", "send_photo", "send_document", "get_updates", "set_webhook", "delete_webhook"],
                "description": "Action to perform"
            },
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID (for sending messages)"
            },
            "text": {
                "type": "string",
                "description": "Message text to send (supports Markdown/HTML)"
            },
            "photo_path": {
                "type": "string",
                "description": "Local path to photo file (for send_photo)"
            },
            "document_path": {
                "type": "string",
                "description": "Local path to document file (for send_document)"
            },
            "photo_url": {
                "type": "string",
                "description": "URL of photo (alternative to photo_path)"
            },
            "parse_mode": {
                "type": "string",
                "enum": ["HTML", "MarkdownV2", "Markdown"],
                "description": "Parse mode for message formatting",
                "default": "HTML"
            },
            "reply_markup": {
                "type": "object",
                "description": "Inline keyboard markup for interactive buttons"
            },
            "webhook_url": {
                "type": "string",
                "description": "Webhook URL (for set_webhook)"
            },
            "allowed_updates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Types of updates to receive"
            }
        },
        "required": ["action"]
    }

    def __init__(
        self,
        token: Optional[str] = None,
        allowed_user_ids: Optional[List[int]] = None,
        agent_callback = None,
        polling: bool = True,
        webhook_url: Optional[str] = None,
        config_path: str = "./data/telegram"
    ):
        """
        Initialize Telegram Bot tool

        Args:
            token: Telegram bot token (from @BotFather)
            allowed_user_ids: List of allowed Telegram user IDs (None = all users)
            agent_callback: Callback function to process incoming messages
                           Should be async func(update: Update, context: ContextTypes.DEFAULT_TYPE)
            polling: Use polling mode (True) or webhook (False)
            webhook_url: Webhook URL if using webhook mode
            config_path: Path to store bot data and state
        """
        if not _HAS_TELEGRAM:
            raise ImportError(
                "Telegram bot dependencies not installed. "
                "Run: pip install python-telegram-bot --upgrade"
            )

        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.allowed_user_ids = set(allowed_user_ids) if allowed_user_ids else None
        self.agent_callback = agent_callback
        self.polling = polling
        self.webhook_url = webhook_url
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)

        # State
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self.is_running = False
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger("TelegramBot")

        # Statistics
        self.messages_received = 0
        self.messages_sent = 0
        self.start_time: Optional[datetime] = None

    def execute(
        self,
        action: str,
        chat_id: Optional[str] = None,
        text: Optional[str] = None,
        photo_path: Optional[str] = None,
        document_path: Optional[str] = None,
        photo_url: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
        allowed_updates: Optional[List[str]] = None
    ) -> ToolResult:
        """
        Execute Telegram bot action

        Note: Most actions are async but this is sync wrapper
        """
        try:
            if not self.token:
                return ToolResult(
                    success=False,
                    output="",
                    error="Telegram bot token required. Set telegram.bot_token in config.yaml or TELEGRAM_BOT_TOKEN env var."
                )

            # Lazy initialization
            if not self.bot:
                self._init_bot()

            if action == "start":
                return self._start_bot()
            elif action == "stop":
                return self._stop_bot()
            elif action == "send_message":
                return self._send_message_sync(chat_id, text, parse_mode, reply_markup)
            elif action == "send_photo":
                return self._send_photo_sync(chat_id, photo_path, photo_url, text, parse_mode)
            elif action == "send_document":
                return self._send_document_sync(chat_id, document_path, text, parse_mode)
            elif action == "get_updates":
                return self._get_updates_sync()
            elif action == "set_webhook":
                return self._set_webhook_sync(webhook_url, allowed_updates)
            elif action == "delete_webhook":
                return self._delete_webhook_sync()
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            self.logger.error(f"Telegram action '{action}' failed: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Telegram error: {str(e)}"
            )

    def _init_bot(self):
        """Initialize bot instance"""
        self.bot = Bot(token=self.token)
        self.logger.info("Telegram bot initialized")

    def _start_bot(self) -> ToolResult:
        """Start bot polling/webhook"""
        if self.is_running:
            return ToolResult(success=True, output="Bot already running")

        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Build application
        self.application = Application.builder().token(self.token).build()

        # Register handlers
        self._register_handlers()

        # Start in background
        if self.polling:
            loop.create_task(self._start_polling())
        else:
            if self.webhook_url:
                loop.create_task(self._start_webhook())
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error="Webhook URL required for webhook mode"
                )

        self.is_running = True
        self.start_time = datetime.now()

        return ToolResult(
            success=True,
            output=f"Telegram bot started in {'polling' if self.polling else 'webhook'} mode",
            metadata={"mode": "polling" if self.polling else "webhook"}
        )

    async def _start_polling(self):
        """Start polling for updates"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES
        )
        self.logger.info("Polling started")
        await self.application.updater.idle()  # Runs forever

    async def _start_webhook(self):
        """Start webhook server"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_webhook(
            url=self.webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
        self.logger.info(f"Webhook started at {self.webhook_url}")

    def _register_handlers(self):
        """Register message and command handlers"""
        # Command handlers
        from telegram.ext import CommandHandler

        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("status", self._cmd_status))

        # Message handler for text
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        # Photo handler
        self.application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))

        # Document handler
        self.application.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))

        # Callback query handler (for inline keyboards)
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        welcome = f"Hello {user.first_name}! I'm an AI agent powered by OpenMythos. Send me any task or question, and I'll think deeply and help you. Use /help for commands."
        await update.message.reply_text(welcome)

        self.messages_received += 1
        self.logger.info(f"User {user.id} started conversation")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """**Available Commands:**
/start - Start conversation
/help - Show this help
/status - Check agent status

Just send me a message with your task or question. I'll use deep reasoning to help you!"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status = f"""🤖 **Agent Status**

• Uptime: {(datetime.now() - self.start_time).seconds // 3600}h {(datetime.now() - self.start_time).seconds % 3600 // 60}m
• Messages received: {self.messages_received}
• Messages sent: {self.messages_sent}
• Status: {'🟢 Running' if self.is_running else '🔴 Stopped'}

_Updated: {datetime.now().strftime('%H:%M:%S')}_"""

        if self.agent_callback:
            status += "\n• Agent callback: ✅ Connected"
        else:
            status += "\n• Agent callback: ⚠️ Not set (messages will be queued)"

        await update.message.reply_text(status, parse_mode=ParseMode.MARKDOWN)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text message"""
        user = update.effective_user
        chat_id = update.effective_chat.id
        message_text = update.message.text

        # Check if user is allowed
        if self.allowed_user_ids and user.id not in self.allowed_user_ids:
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return

        self.messages_received += 1
        self.logger.info(f"📨 Message from {user.username or user.id}: {message_text[:50]}...")

        # Build message object
        tg_msg = TelegramMessage(
            message_id=update.message.message_id,
            chat_id=chat_id,
            text=message_text,
            user_id=user.id,
            username=user.username,
            timestamp=update.message.date,
            message_type="text",
            reply_to=update.message.reply_to_message.message_id if update.message.reply_to_message else None
        )

        # Process with agent (if callback set)
        if self.agent_callback:
            try:
                # This is where the Agent processes the message
                response = await self.agent_callback(tg_msg, context)
                if response:
                    await update.message.reply_text(response, parse_mode=ParseMode.HTML)
                    self.messages_sent += 1
            except Exception as e:
                self.logger.error(f"Agent callback error: {e}")
                await update.message.reply_text(
                    f"❌ Agent processing error: {str(e)[:200]}",
                    parse_mode=ParseMode.HTML
                )
        else:
            # Queue for later processing
            await self.message_queue.put(tg_msg)
            await update.message.reply_text("✅ Message queued for processing (no agent callback set).")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photo"""
        user = update.effective_user
        chat_id = update.effective_chat.id

        if self.allowed_user_ids and user.id not in self.allowed_user_ids:
            await update.message.reply_text("⛔ Not authorized.")
            return

        # Get largest photo
        photo = update.message.photo[-1]
        file_id = photo.file_id

        # Download photo
        file = await self.bot.get_file(file_id)
        download_path = self.config_path / f"photo_{uuid.uuid4().hex[:8]}.jpg"
        await file.download_to_drive(str(download_path))

        self.messages_received += 1

        tg_msg = TelegramMessage(
            message_id=update.message.message_id,
            chat_id=chat_id,
            text=update.message.caption or "",
            user_id=user.id,
            username=user.username,
            timestamp=update.message.date,
            message_type="photo",
            file_id=file_id,
            file_path=str(download_path),
            reply_to=update.message.reply_to_message.message_id if update.message.reply_to_message else None
        )

        if self.agent_callback:
            try:
                response = await self.agent_callback(tg_msg, context)
                if response:
                    await update.message.reply_text(response, parse_mode=ParseMode.HTML)
                    self.messages_sent += 1
            except Exception as e:
                self.logger.error(f"Photo callback error: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)[:200]}")
        else:
            await self.message_queue.put(tg_msg)
            await update.message.reply_text("Photo received and queued.")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming document"""
        user = update.effective_user
        chat_id = update.effective_chat.id

        if self.allowed_user_ids and user.id not in self.allowed_user_ids:
            await update.message.reply_text("⛔ Not authorized.")
            return

        doc = update.message.document
        file_id = doc.file_id
        filename = doc.file_name

        # Download
        file = await self.bot.get_file(file_id)
        download_path = self.config_path / f"doc_{uuid.uuid4().hex[:8]}_{filename}"
        await file.download_to_drive(str(download_path))

        self.messages_received += 1

        tg_msg = TelegramMessage(
            message_id=update.message.message_id,
            chat_id=chat_id,
            text=update.message.caption or "",
            user_id=user.id,
            username=user.username,
            timestamp=update.message.date,
            message_type="document",
            file_id=file_id,
            file_path=str(download_path),
            reply_to=update.message.reply_to_message.message_id if update.message.reply_to_message else None
        )

        if self.agent_callback:
            try:
                response = await self.agent_callback(tg_msg, context)
                if response:
                    await update.message.reply_text(response, parse_mode=ParseMode.HTML)
                    self.messages_sent += 1
            except Exception as e:
                self.logger.error(f"Document callback error: {e}")
                await update.message.reply_text(f"❌ Error: {str(e)[:200]}")
        else:
            await self.message_queue.put(tg_msg)
            await update.message.reply_text(f"Document '{filename}' received and queued.")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses"""
        query = update.callback_query
        await query.answer()

        data = query.data
        user = query.from_user

        self.logger.info(f"Callback from {user.id}: {data}")

        if self.agent_callback:
            try:
                response = await self.agent_callback({
                    'type': 'callback',
                    'data': data,
                    'user_id': user.id,
                    'chat_id': query.message.chat_id,
                    'message_id': query.message.message_id
                }, context)
                if response:
                    await query.edit_message_text(text=response, parse_mode=ParseMode.HTML)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    def _send_message_sync(
        self,
        chat_id: Optional[str],
        text: str,
        parse_mode: str = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Synchronous wrapper for send_message"""
        if not chat_id:
            return ToolResult(success=False, output="", error="chat_id required")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def send():
            try:
                msg = await self.bot.send_message(
                    chat_id=int(chat_id),
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=self._parse_reply_markup(reply_markup) if reply_markup else None
                )
                self.messages_sent += 1
                return ToolResult(
                    success=True,
                    output=f"Message sent to {chat_id}",
                    metadata={"message_id": msg.message_id, "chat_id": chat_id}
                )
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return loop.run_until_complete(send())

    def _send_photo_sync(
        self,
        chat_id: Optional[str],
        photo_path: Optional[str],
        photo_url: Optional[str],
        caption: Optional[str],
        parse_mode: str = "HTML"
    ) -> ToolResult:
        """Synchronous wrapper for send_photo"""
        if not chat_id:
            return ToolResult(success=False, output="", error="chat_id required")
        if not photo_path and not photo_url:
            return ToolResult(success=False, output="", error="photo_path or photo_url required")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def send():
            try:
                if photo_path and Path(photo_path).exists():
                    with open(photo_path, 'rb') as f:
                        msg = await self.bot.send_photo(
                            chat_id=int(chat_id),
                            photo=f,
                            caption=caption,
                            parse_mode=parse_mode
                        )
                elif photo_url:
                    msg = await self.bot.send_photo(
                        chat_id=int(chat_id),
                        photo=photo_url,
                        caption=caption,
                        parse_mode=parse_mode
                    )
                else:
                    return ToolResult(success=False, output="", error="No valid photo source")

                self.messages_sent += 1
                return ToolResult(
                    success=True,
                    output=f"Photo sent to {chat_id}",
                    metadata={"message_id": msg.message_id}
                )
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return loop.run_until_complete(send())

    def _send_document_sync(
        self,
        chat_id: Optional[str],
        document_path: Optional[str],
        caption: Optional[str],
        parse_mode: str = "HTML"
    ) -> ToolResult:
        """Synchronous wrapper for send_document"""
        if not chat_id or not document_path:
            return ToolResult(success=False, output="", error="chat_id and document_path required")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def send():
            try:
                path = Path(document_path)
                if not path.exists():
                    return ToolResult(success=False, output="", error=f"File not found: {document_path}")

                with open(document_path, 'rb') as f:
                    msg = await self.bot.send_document(
                        chat_id=int(chat_id),
                        document=f,
                        caption=caption,
                        parse_mode=parse_mode
                    )

                self.messages_sent += 1
                return ToolResult(
                    success=True,
                    output=f"Document sent to {chat_id}",
                    metadata={"message_id": msg.message_id, "filename": path.name}
                )
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return loop.run_until_complete(send())

    def _stop_bot(self) -> ToolResult:
        """Stop bot"""
        if not self.is_running:
            return ToolResult(success=True, output="Bot not running")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def stop():
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            self.is_running = False
            self.logger.info("Bot stopped")

        loop.run_until_complete(stop())

        return ToolResult(success=True, output="Bot stopped")

    def _get_updates_sync(self, limit: int = 10) -> ToolResult:
        """Get recent updates (for debugging)"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def get():
            if not self.bot:
                self._init_bot()
            updates = await self.bot.get_updates(limit=limit)
            result = []
            for u in updates:
                result.append({
                    'update_id': u.update_id,
                    'message': {
                        'text': u.message.text if u.message else None,
                        'chat_id': u.message.chat_id if u.message else None,
                        'user': u.message.from_user.first_name if u.message else None
                    } if u.message else None
                })
            return ToolResult(success=True, output=str(result), metadata={'updates': result})

        return loop.run_until_complete(get())

    def _set_webhook_sync(self, url: Optional[str], allowed_updates: Optional[List[str]] = None):
        """Set webhook URL"""
        if not url:
            return ToolResult(success=False, output="", error="webhook_url required")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def set_hook():
            result = await self.bot.set_webhook(
                url=url,
                allowed_updates=allowed_updates or ['message', 'callback_query']
            )
            if result:
                self.webhook_url = url
                return ToolResult(success=True, output=f"Webhook set to {url}")
            else:
                return ToolResult(success=False, output="", error="Webhook setup failed")

        return loop.run_until_complete(set_hook())

    def _delete_webhook_sync(self) -> ToolResult:
        """Delete webhook (switch to polling)"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def delete():
            result = await self.bot.delete_webhook()
            if result:
                self.webhook_url = None
                return ToolResult(success=True, output="Webhook deleted")
            else:
                return ToolResult(success=False, output="", error="Failed to delete webhook")

        return loop.run_until_complete(delete())

    def _parse_reply_markup(self, markup_dict: Optional[Dict[str, Any]]) -> Optional[InlineKeyboardMarkup]:
        """Parse reply markup dict to InlineKeyboardMarkup"""
        if not markup_dict:
            return None

        # Simple parser for button matrix
        # Expected format: {"keyboard": [[{"text": "...", "callback_data": "..."}]]}
        keyboard = []
        for row in markup_dict.get('keyboard', []):
            buttons = []
            for btn in row:
                if 'callback_data' in btn:
                    buttons.append(
                        InlineKeyboardButton(
                            text=btn['text'],
                            callback_data=btn['callback_data']
                        )
                    )
                elif 'url' in btn:
                    buttons.append(
                        InlineKeyboardButton(
                            text=btn['text'],
                            url=btn['url']
                        )
                    )
            if buttons:
                keyboard.append(buttons)

        return InlineKeyboardMarkup(keyboard) if keyboard else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get bot statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        return {
            'is_running': self.is_running,
            'mode': 'polling' if self.polling else 'webhook',
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'uptime_seconds': uptime,
            'queue_size': self.message_queue.qsize() if hasattr(self.message_queue, 'qsize') else 0
        }


# Factory function for easy creation
def create_telegram_bot(
    token: str,
    agent_callback = None,
    allowed_users: List[int] = None,
    polling: bool = True,
    webhook_url: str = None,
    config_path: str = "./data/telegram"
) -> TelegramBotTool:
    """Create and configure Telegram bot tool"""
    bot = TelegramBotTool(
        token=token,
        allowed_user_ids=allowed_users,
        agent_callback=agent_callback,
        polling=polling,
        webhook_url=webhook_url,
        config_path=config_path
    )
    return bot


# Register the tool
if _HAS_TELEGRAM:
    TOOLS_REGISTRY.register(TelegramBotTool())
else:
    print(_style("  telegram tool not registered (python-telegram-bot not installed)", "2"))
