# 🤖 OpenMythos Agent - Telegram Integration Guide

Connect your AI agent to Telegram for instant messaging access!

---

## 🚀 Quick Start

### 1. Get a Telegram Bot Token

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Follow instructions to create your bot
4. Save the token (format: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

### 2. Install Dependencies

```bash
cd agent_project
pip install python-telegram-bot --upgrade
```

### 3. Configure

Edit `config.yaml`:

```yaml
tools:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN_HERE"
    allowed_user_ids: []  # Optional: whitelist of Telegram user IDs
    polling: true  # Use polling (easier) or set false for webhook
    webhook_url: ""  # If webhook mode: "https://your-domain.com/webhook"
```

Or set environment variable:

```bash
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
```

### 4. Run

```bash
# Simple launcher (uses your bot token)
python start_telegram.py

# Or full agent with Telegram enabled
python -m agent_project
```

---

## 📱 Features

### Real-time Chat
- Instant messaging via Telegram
- Support for text, photos, and documents
- Async response handling

### Interactive Buttons
Send inline keyboards for user interaction:

```python
from agent_project.tools import TOOLS_REGISTRY

bot_tool = TOOLS_REGISTRY.get('telegram')

bot_tool.execute(
    action="send_message",
    chat_id="123456",
    text="Choose an option:",
    reply_markup={
        "keyboard": [
            [
                {"text": "✅ Yes", "callback_data": "yes"},
                {"text": "❌ No", "callback_data": "no"}
            ]
        ]
    }
)
```

### File Support
- Send photos (`.jpg`, `.png`, etc.)
- Send documents (`.pdf`, `.txt`, `.py`, etc.)
- Agent receives downloaded files locally

### User Whitelisting
Restrict bot access:

```yaml
telegram:
  allowed_user_ids: [12345678, 87654321]  # Only these users can use bot
```

---

## 🔧 Advanced Usage

### Custom Agent Callback

Integrate your OpenMythosAgent:

```python
import asyncio
from agent_project.tools import create_telegram_bot
from agent_project.agent import OpenMythosAgent
from agent_project.config import load_config

# Load agent
config = load_config("config.yaml")
agent = OpenMythosAgent(config)

async def telegram_callback(update, context):
    """Process Telegram messages with the agent"""
    text = update.message.text
    user = update.effective_user
    
    # Run agent on this task
    result = agent.run(text)
    
    # Format response for Telegram (Markdown supported)
    output = result.get('observations', [{}])[-1].get('output', 'No output')
    return f"**Result:**\n\n{output[:4000]}"

# Create bot
bot = create_telegram_bot(
    token=config.telegram.bot_token,
    agent_callback=telegram_callback,
    polling=True
)

bot.execute(action="start")
```

### Webhook Mode (Production)

For production deployment:

```yaml
telegram:
  polling: false
  webhook_url: "https://your-domain.com/webhook"
```

Create a simple web server to receive updates:

```python
from telegram.ext import Application

# Get the Application from bot
app = bot.application  # (need to modify TelegramBotTool to expose this)

# Use with FastAPI/Flask
@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = Update.de_json(await request.json(), bot.bot)
    await app.process_update(update)
    return {"status": "ok"}
```

---

## 🔒 Security

### User Whitelisting
Prevent unauthorized access:

```python
bot = create_telegram_bot(
    token=token,
    allowed_users=[123456, 789012],  # Only these Telegram user IDs
    agent_callback=callback
)
```

### Rate Limiting
Consider adding rate limiting in your callback to prevent abuse.

### File Restrictions
Downloaded files are stored in `config_path` (default: `./data/telegram`). Ensure this directory is secure.

---

## 📊 Monitoring

Get bot statistics:

```python
stats = bot.get_statistics()
print(f"Running: {stats['is_running']}")
print(f"Messages received: {stats['messages_received']}")
print(f"Messages sent: {stats['messages_sent']}")
print(f"Queue size: {stats['queue_size']}")
```

---

## 🐛 Troubleshooting

### "python-telegram-bot not installed"
```bash
pip install python-telegram-bot --upgrade
```

### "Bot token invalid"
- Double-check token from @BotFather
- No spaces or extra characters

### Bot not responding
- Check logs for errors
- Ensure agent_callback is set and not throwing exceptions
- Verify bot has permission to send messages

### Webhook fails
- Use polling mode for easier setup
- If using webhook, ensure HTTPS URL is valid
- Check with `/setwebhook` command in @BotFather

---

## 📁 File Structure

```
agent_project/
├── config.yaml              # Main config (add telegram.bot_token)
├── start_telegram.py        # Quick launcher
├── data/
│   └── telegram/            # Bot data, downloaded files
├── logs/
│   └── agent.log
└── agent_project/
    ├── tools/
    │   ├── telegram_bot.py  # Telegram bot implementation
    │   └── ...
```

---

## 🎯 Example Use Cases

1. **Personal AI Assistant**: Chat with your agent anywhere via Telegram
2. **Team Productivity**: Share agent with team (use `allowed_user_ids` for access control)
3. **Monitoring Bot**: Get alerts from your agent in Telegram
4. **Task Submission**: Submit tasks to agent from mobile
5. **Interactive Menus**: Use inline keyboards for guided workflows

---

## ⚡ Performance Notes

- **Polling mode**: Checks for updates every 0.5-1 second. Suitable for small deployments.
- **Webhook mode**: Push-based, lower latency, suitable for production/high traffic
- **Async**: All operations are non-blocking; agent runs in background
- **Memory**: Downloaded files are cleaned up periodically (not automatic - implement cleanup if needed)

---

## 🛡️ Best Practices

1. **Token Security**: Never commit bot token to git. Use environment variables or `.env` file.
2. **User Whitelisting**: Always use `allowed_user_ids` in production to prevent unauthorized access.
3. **Error Handling**: Your callback should handle exceptions gracefully.
4. **Rate Limits**: Implement rate limiting in your callback for production use.
5. **File Cleanup**: Regularly clean `./data/telegram` directory to avoid disk fill.

---

## 🤝 Support

Issues: https://github.com/your-repo/issues
