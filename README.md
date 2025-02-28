# DeepTrade

DeepTrade is a Python-based prototype for quantitative cryptocurrency (*now Bitcoin (BTC) - Solana (SOL) to come*) trading and research. It combines **20-minute interval historical backtesting**, **multi‐output (10‐step) LSTM** modeling, GPT-based sentiment analysis, and an **offline single‐step Reinforcement Learning (DQN)** pipeline.

---

## Multi-Timeframe, Multi‐Output LSTM Model Architecture

<div align="center"><center><img src="models/advanced_lstm_model_architecture.png" alt="Multi-Timeframe LSTM-Transformer Model Architecture" width="300"></center></div>

This advanced multi-input deep learning model combines **residual LSTM layers**, **Transformer-style self-attention**, and **dense feed-forward** networks. The model takes multiple time-series inputs (e.g., 5m, 15m, 1h, plus Google Trends and Santiment data) alongside sentiment signals, technical indicators, and market context.

- The model produces **10 outputs** (`y_1..y_10`), each corresponding to a future horizon at +1 step through +10 steps (each step is 20 minutes).  
- We skip rows where all 10 future steps are not available.  

By training on these multi-step forecasts, the model can capture short-term price trajectories in 20‐minute increments.

---

## Single‐Step Deep Q-Network (DQN) Reinforcement Learning

After the neural network model generates 10 predicted returns (`y_1..y_10`), an **RL agent** (DQN) uses these predicted multi-horizon signals (and possibly other features) to decide a **LONG, SHORT, or HOLD** action at **every single 20‐minute bar**.

1. **Single-step reward**: The RL agent collects a reward each step (20 minutes) based on immediate price changes (or PnL).  
2. **Replay buffer**: Stores past transitions (state, action, reward, next_state) to enable offline training.  
3. **DQN**:
   - Feed-forward network approximates Q-values for each action (LONG, SHORT, HOLD).
   - Epsilon-greedy policy for exploration vs. exploitation.
   - Periodic target network updates for stability.

**Goal**: Maximize cumulative reward (e.g., % returns) over many 20‐minute trading steps.

---

## Project Status

- **Alpha:** Core code is functional but experimental.
- **No guaranteed performance:** This is a research/learning tool.
- **Many aspects** (model architecture, signals, RL logic) are still being tested.
- **Contributions, issues, and suggestions are welcome!**

> **Disclaimer:** This project is in active development and might contain bugs. There is no guarantee of profitability or correctness. Use for learning and experimentation.
---

## Overview

### Major Components

1. **`backtester.py`**
   - **Runs historical backtests in 20-minute increments** (configurable via `STEP_INTERVAL_MINUTES`).
   - For each bar, it fetches data, updates the bot, and logs scenario-based CSV outputs (like `backtest_result_local_gpt_1.csv`).
   - Builds **multi-output training data** (`training_data.csv`) with columns `y_1..y_10`, each offset by 1..10 steps (20 minutes per step).
   - Also logs RL transitions (single-step reward) in `rl_transitions.csv`.

2. **`fitter.py`**
   - Trains a **multi-output LSTM** from `training_data.csv`.
   - Optionally trains a **DQN** (offline) using `rl_transitions.csv`.
   - Implements a rolling, time-ordered approach—no random shuffling—so data remains chronological.


3. **`trader.py`**
   - Executes **live** trading using the LSTM model and optional GPT-based sentiment signals, every 20 minutes.
   - If PAPER_TRADING is `True`, simulates trades locally. Otherwise, attempts real trades on the configured exchange.

4. **`merge_training_data.py`**
   - Merges multiple `training_data.csv` files (from different backtests) into one for more comprehensive training.

5. **`feature_selection.py`**
   - Employs Correlation Analysis, PCA, RFE, and SHAP to measure feature importance.
   - Outputs plots and rankings in a specified folder for easy inspection and pruning of unnecessary features.

---

## Additional Important Directories and Files

- **`.env.copy`**  
  Example environment configuration. Rename to `.env` and fill in your API keys (Binance, Santiment, etc.).

- **`botlib/`**  
  - **`tradebot.py`** — Core `TradingBot` class: handles data fetching, aggregator logic, RL state building.  
  - **`environment.py`** — Shared environment variables and logging.  
  - **`datafetchers.py`** — **Now caches per minute** instead of per hour, fetching from Binance, news, Google Trends, Santiment, etc.  
  - **`input_preprocessing.py`** — Scales inputs for the multi-output LSTM. Also merges all LSTM outputs for RL if needed.  
  - **`indicators.py`** — TA indicators (RSI, MACD, Bollinger, etc.).  
  - **`models.py`** — LSTM model construction and loading.  
  - **`rl.py`** — Single-step DQN agent logic (stores transitions, trains offline).  
  - **`training_data_handler.py`** — Handles real-time data logging from `tradebot.py`. Collects neural network (LSTM) and reinforcement learning (RL) sample data, ensuring consistent formatting and time-aligned rolling-window samples for efficient retraining.

- **`models/`** — Stores trained models, e.g. `advanced_lstm_model.keras`.  

- **`output/`** — Contains CSV outputs from backtests (`*_gpt_1.csv`, etc.) and final signals.  

- **`input_cache/`** — Caches downloaded data (klines, news) by minute. Speeds up repeated runs.  

- **`debug/`** — Debugging logs or GPT prompts.  

- **`training_data/`** — CSV files for training.  
  - `training_data.csv` with multi‐output columns `y_1..y_{10}`.  
  - `rl_transitions.csv` for offline DQN.  

---

## Setup and Installation

### Prerequisites

- Python 3.6 (or 3.7+)
- Conda package manager
- (Optional) NVIDIA GPU + CUDA for faster training

### Steps

1. **Clone the Repo**

   ```bash
   git clone https://github.com/napzter13/DeepTrade.git
   cd DeepTrade
   ```

2. **Set Up the Conda Environment:**

   Update Conda and create a new environment:

   ```bash
   conda update -n base -c defaults conda
   conda update -n base --all
   conda create --name deeptrade_env python=3.6
   conda activate deeptrade_env
   ```

3. **Install Dependencies:**

   Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys:**

   - Rename `.env.copy` to `.env`.
   - Populate the `.env` file with your API keys and other necessary configurations.

## Usage

1. **Backtesting:**

   Run the backtester to simulate historical trading:

   ```bash
   python backtester.py
   ```

   - Iterates historically in 20-minute steps (default).
   - Builds training_data.csv with columns:
   - [timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta_63, arr_ctx_11, y_1 .. y_{10}].
   - Also writes RL transitions in rl_transitions.csv using single-step reward.

2. **Training the Model:**

   After generating `training_data.csv` through backtesting, train the LSTM model:

   ```bash
   python fitter.py
   ```

   - LSTM: Trains the multi‐output LSTM that predicts 10 future returns.
   - RL: If rl_transitions.csv is present, trains an offline DQN that uses those transitions.
   - Uses chronological splits (`train_ratio`, `val_ratio`) without shuffling, preserving time order.
   - Saves the LSTM model to models/advanced_lstm_model.keras (and DQN weights to rl_DQNAgent.weights.h5).


3. **Live Trading:**

   Once the model is trained and saved as `lstm_model.h5`, start live trading:

   ```bash
   python trader.py
   ```

   - Runs the TradingBot logic in 20-minute intervals by default.
   - Loads the multi‐output LSTM model, obtains predicted returns, and passes them to the RL agent to produce an action (LONG/SHORT/HOLD).
   - If PAPER_TRADING=True, only simulates trades. Otherwise, attempts real trades via the Binance API.

## Contributing

1. Fork the repo & create a feature branch.
2. Make your changes, add tests if possible.
3. Submit a Pull Request and describe your changes in detail.

## License

This project does not yet have a formal license.
Use at your own risk, for non-commercial or internal research purposes.

For questions, open an Issue on GitHub.

## Acknowledgments

- [Binance Python API](https://github.com/binance/binance-spot-api-docs)  
- [OpenAI / GPT APIs](https://platform.openai.com/docs/introduction)  
- [Transformers (HuggingFace)](https://github.com/huggingface/transformers)  
- [TensorFlow / Keras](https://github.com/tensorflow/tensorflow)  
- and the broader open-source ecosystem.

