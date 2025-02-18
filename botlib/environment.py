#!/usr/bin/env python3
"""
Environment variables, global flags, and logging setup.

- Loads .env variables (dotenv)
- Sets global PAPER_TRADING, keys, etc.
- Creates directories
- Configures a basic logger
"""

import os
import logging
import nltk
from dotenv import load_dotenv

# VADER for text sentiment
nltk.download("vader_lexicon", quiet=True)

# Load .env
load_dotenv()

# Global environment flags and API keys
PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "deeptrade/0.1")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MIXTRAL_MODEL_PATH = os.getenv("MIXTRAL_MODEL_PATH", "./models/mixtral")
DEEPSEEK_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH", "./models/deepseek-v3")
ADVANCED_MODEL_PATH = os.getenv("ADVANCED_MODEL_PATH", "models/advanced_lstm_model.keras")
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "models/rl_DQNAgent.weights.h5")
STEP_INTERVAL_MINUTES = int(os.getenv("STEP_INTERVAL_MINUTES", 20))
NUM_FUTURE_STEPS = int(os.getenv("NUM_FUTURE_STEPS", 10))

# Make sure directories exist
os.makedirs("debug", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("input_cache", exist_ok=True)

# Cache file constants
GPT_CACHE_FILE = os.path.join("input_cache", "gpt_cache.json")

# Basic logger configuration
def get_logger(name="TradingBotLogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
