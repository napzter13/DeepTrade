# DeepTrade

**DeepTrade** is a Python-based prototype for quantitative cryptocurrency trading and research.  
It combines historical backtesting, multi-timeframe LSTM modeling, GPT-based sentiment analysis,  
and optional offline Reinforcement Learning (DQN).  

> **Disclaimer**: This project is **in active development** and might contain bugs.  
> There is no guarantee of profitability or correctness. Use for learning and experimentation.

---

## Overview

### Major Components

1. **`backtester.py`**  
   - Simulates historical trading from a specified start date to end date.  
   - Generates scenario-based CSV logs (e.g., `backtest_result_local_gpt_1.csv`) to observe how different signals would have performed.  
   - Builds a supervised learning dataset in `training_data.csv` for an advanced multi-input LSTM model (after a +3h future price offset).  
   - Logs RL transitions into `rl_transitions.csv` for optional offline RL training.

2. **`clean_all_outputs.py`**  
   - Simple script that deletes all files from the `debug`, `input_cache`, and `output` folders.  
   - Useful for quickly resetting/cleaning intermediate artifacts.

3. **`fitter.py`**  
   - Main training script for two parts:  
     1. **Multi-input LSTM** from `training_data.csv`.  
     2. **Offline DQN** training from `rl_transitions.csv`.  
   - Supports command-line arguments to control epochs, batch size, etc.  
   - Splits data into train/validation/test sets, trains the LSTM, evaluates test loss, and optionally trains a DQN agent offline.

4. **`trader.py`**  
   - Live or paper-trading script.  
   - Instantiates a `TradingBot` that runs every hour using `schedule`.  
   - The `TradingBot` fetches live data (or sim data), applies the advanced LSTM or GPT-based logic, and places trades (paper or real).

### Additional Important Directories and Files

- **`botlib/`**  
  A local Python package directory with submodules for:
  - `tradebot.py`: The `TradingBot` class and all logic for data fetching, aggregator, RL state-building, and position management.
  - `environment.py`: Shared environment variables and utilities (like `get_logger`).
  - `datafetchers.py`: Functions to retrieve data from Binance, news APIs, Google Trends, Santiment, etc.
  - `input_preprocessing.py`: Contains the `ModelScaler` class and a helper to scale all 5 inputs for the multi-timeframe LSTM.
  - `indicators.py`: Technical indicators (RSI, MACD, Bollinger, etc.).
  - `models.py`:  
    - Loads local HuggingFace language models (or falls back to OpenAI) for sentiment/guidance.  
    - Builds and loads the advanced multi-timeframe LSTM model.  
  - `rl.py`: A simple DQNAgent for RL-based action selection and offline training memory.

- **`models/advanced_lstm_model.keras`** (generated at runtime)  
  - The saved multi-timeframe LSTM model (SavedModel or H5).  

- **`output/`**  
  - Folder to store all CSV outputs (backtest logs, `training_data.csv`, RL transitions, etc.).

- **`input_cache/`**  
  - Cache for downloaded data (e.g. klines, news, etc.) to speed up repeated runs.

- **`debug/`**  
  - Contains various logs or debugging info from GPT calls, aggregator prompts, etc.

---

## Quick Start

1. **Install Dependencies**  
   - Python 3.8+ recommended  
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   (Project libraries like `tensorflow`, `transformers`, `schedule`, `openai`, `binance`, `sklearn` etc. are assumed.)

2. **Prepare Environment**  
   - Copy or rename `.env.example` to `.env` (if provided) and fill with your keys (e.g., BINANCE_API_KEY, NEWS_API_KEY, etc.).  
   - Or define environment variables directly for any external APIs.

3. **Run the Backtester**  
   ```bash
   python backtester.py
   ```
   - Iterates historically (default 30 days) hour by hour.  
   - Writes scenario CSVs to `output/` and accumulates training data in `output/training_data.csv`.

4. **Train the LSTM Model**  
   ```bash
   python fitter.py --csv output/training_data.csv --model_out models/advanced_lstm_model.keras --epochs 600
   ```
   - Reads from `training_data.csv` and trains a multi-input LSTM.  
   - You can also train an offline DQN if you have `output/rl_transitions.csv`.

5. **Clean Output**  
   ```bash
   python clean_all_outputs.py
   ```
   - Wipes the `debug`, `input_cache`, and `output` folders of files.  
   - Use carefully—this removes logs and any CSV data.

6. **Run Live/Paper Trading**  
   ```bash
   python trader.py
   ```
   - Instantiates `TradingBot`, does an immediate run, then schedules hourly runs.  
   - If PAPER_TRADING is set (in environment.py or `.env`), it simulates trades locally.  
   - Otherwise, it attempts real trades on your configured exchange.

---

## Project Status

- **Alpha**: Core code is functional but experimental.  
- **No guaranteed performance**: This is a research/learning tool.  
- Many aspects (model architecture, signals, RL logic) are still being tested.  
- Contributions, issues, and suggestions are welcome!

---

## Contributing

1. Fork the repo & create a feature branch.  
2. Make your changes, add tests if possible.  
3. Submit a Pull Request and describe your changes in detail.

---

## License

This project does not yet have a formal license.  
**Use at your own risk**, for non-commercial or internal research purposes.  

For questions, contact the maintainer or open an Issue on GitHub.

---

### Acknowledgments

- [Binance Python API](https://github.com/binance/binance-spot-api-docs)  
- [OpenAI / GPT APIs](https://platform.openai.com/docs/introduction)  
- [Transformers (HuggingFace)](https://github.com/huggingface/transformers)  
- [TensorFlow / Keras](https://github.com/tensorflow/tensorflow)  

and the broader open-source ecosystem.

