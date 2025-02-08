#!/usr/bin/env python3
"""
tradebot.py
Main runner for live/paper trading using the TradingBot class.
Schedules an hourly run, just like the old monolith code's main().
"""

import schedule
import time
import os

from botlib.tradebot import TradingBot


def main():
    os.system('cls')            # clear terminal Windows
    bot = TradingBot()
    bot.logger.info("Starting live/paper trading bot...")

    # Run once now:
    bot.run_cycle()

    # Then schedule e.g. hourly
    schedule.every().hour.do(bot.run_cycle)
    bot.logger.info("Bot scheduled to run hourly.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        bot.logger.info("Stopped manually.")
    except Exception as e:
        bot.logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
