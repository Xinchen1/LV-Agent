#!/usr/bin/env python3
"""
Example script for running the OpenMythos Agent with Telegram integration.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'agent_project'))

from agent_project.agent import OpenMythosAgent
from agent_project.config import load_config
from agent_project.logging_module import setup_logging, get_logger
from agent_project.tools import TelegramBotTool, TOOLS_REGISTRY

# Configure logger (will be reconfigured after logging setup)
logger = get_logger(__name__)


def run_task_with_agent(task: str, agent: OpenMythosAgent) -> Dict[str, Any]:
    """
    Run a task with the provided agent instance.

    This is a placeholder - you'd use your actual agent instance.
    """
    # Example: Load config and agent (you'd want to reuse a single agent instance)
    try:
        config = load_config("config.yaml")
        agent = OpenMythosAgent(config)
        
        # Run the task
        result = agent.run(task)
        
        # Format response
        success = result.get('success', False)
        reward = result.get('final_reward', 0)
        outer_loops = result.get('outer_loops', 0)
        
        return {
            "success": success,
            "reward": reward,
            "outer_loops": outer_loops,
            "result": result
        }
    except Exception as e:
        logger.exception(f"Error running task: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    print("OpenMythos Agent - Telegram Bot")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config("config.yaml")
    except Exception as e:
        print(f"��❌ Config error: {e}")
        return
    
    # Setup logging based on config
    setup_logging(config.logging)
    
    # Re-get logger now that logging is configured
    logger = get_logger(__name__)
    
    # Check Telegram config
    telegram_config = config.tools.telegram if hasattr(config.tools, 'telegram') else {}
    if not telegram_config.get('bot_token'):
        logger.error("Telegram bot token not set!")
        logger.error("   Set telegram.bot_token in config.yaml or TELEGRAM_BOT_TOKEN env var")
        return

    # Create agent instance
    try:
        agent = OpenMythosAgent(config)
        logger.info("OpenMythos Agent initialized successfully")
    except Exception as e:
        logger.exception(f"Failed to initialize OpenMythos Agent: {e}")
        return

    # Create Telegram bot tool
    try:
        telegram_tool = TelegramBotTool(
            bot_token=telegram_config.get('bot_token') or os.getenv('TELEGRAM_BOT_TOKEN'),
            allowed_user_ids=telegram_config.get('allowed_user_ids', []),
            data_path=telegram_config.get('data_path', './data/telegram')
        )
        logger.info("Telegram bot tool initialized successfully")
    except Exception as e:
        logger.exception(f"Failed to initialize Telegram bot tool: {e}")
        return

    # Add tool to registry
    TOOLS_REGISTRY.register(telegram_tool)
    logger.info("Telegram bot tool registered")

    # Here you would typically start the bot in polling mode or webhook mode
    # For this example, we'll just show that everything is set up correctly
    logger.info("Setup complete. Ready to run Telegram bot.")
    print("��✅ Setup complete. Check logs for details.")
    
    # Example of how to run the bot (uncomment to use):
    # try:
    #     telegram_tool.start_polling()
    #     logger.info("Telegram bot started in polling mode")
    # except Exception as e:
    #     logger.exception(f"Error starting Telegram bot: {e}")


if __name__ == '__main__':
    main()
