#!/usr/bin/env python3
"""
trader.py

- Runs the TradingBot every STEP_INTERVAL_MINUTES (default=20).

Usage:
  python trader.py
"""

import schedule
import time
import os

from botlib.environment import (
    STEP_INTERVAL_MINUTES
)


from botlib.tradebot import TradingBot

def main():
    os.system('cls' if os.name == 'nt' else 'clear')  # optional: clear terminal
    bot = TradingBot()
    bot.logger.info("Starting live/paper trading bot...")

    # Run once now immediately:
    bot.run_cycle()

    # Then schedule every STEP_INTERVAL_MINUTES
    schedule.every(STEP_INTERVAL_MINUTES).minutes.do(bot.run_cycle)
    bot.logger.info(f"Bot scheduled to run every {STEP_INTERVAL_MINUTES} minutes.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        bot.logger.info("Stopped manually (KeyboardInterrupt).")
    except Exception as e:
        bot.logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
