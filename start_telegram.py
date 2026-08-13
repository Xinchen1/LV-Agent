#!/usr/bin/env python3
"""
Telegram bot example for OpenMythos Agent.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'agent_project'))

from agent_project.agent import OpenMythosAgent
from agent_project.config import load_config
from agent_project.logging_module import setup_logging, get_logger
from agent_project.tools import create_telegram_bot

# Configure logging using the centralized configuration
# Note: This will be updated after config is loaded
logger = get_logger(__name__)


def main():
    config_path = Path(__file__).parent / 'config.json'
    if not config_path.exists():
        config_path = Path(__file__).parent / 'config.yaml'
    
    config = load_config(str(config_path))
    
    # Setup logging based on config
    setup_logging(config.logging)
    
    # Re-get logger now that logging is configured
    logger = get_logger(__name__)
    
    logger.info("Starting OpenMythos Telegram Bot")
    
    telegram_cfg = config.tools.telegram
    token = telegram_cfg.get('bot_token') or os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("Telegram bot token not set in config.yaml and TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    allowed_user_ids = telegram_cfg.get('allowed_user_ids') or []
    webhook_url = telegram_cfg.get('webhook_url') or None
    data_path = telegram_cfg.get('config_path') or './data/telegram'

    # Create agent instance
    agent = OpenMythosAgent(config)

    # Create Telegram bot
    bot = create_telegram_bot(
        token=token,
        agent=agent,
        allowed_user_ids=allowed_user_ids,
        webhook_url=webhook_url,
        data_path=data_path,
    )

    # Run the bot
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(f"Unexpected error in Telegram bot: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
