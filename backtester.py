#!/usr/bin/env python3
"""
backtester.py

- Iterates from 2024-01-01 up to 2025-01-01, stepping 1 hour at a time.
- For each hour:
   1) We set the HistoricalTradingBot to that 'current_time'.
   2) We run the normal TradingBot logic => produce last_log_data (including signals).
   3) We store those signals in scenario-based CSV logs.
   4) We also buffer that iteration's multi-input feature arrays for building the training_data
      after +3h price is known. (For advanced LSTM model.)

At the end:
- We have 4 scenario-based CSVs (local_gpt_1, local_gpt_2, final_signal).
- We also produce "training_data.csv" with columns:
  [timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta_63, arr_ctx_11, y]
  where y is the future price %change after +3h, clamped to [-1,1].
"""

import os
import csv
import json
import datetime
import sys
import signal
import math
import threading

from botlib.tradebot import TradingBot
from botlib.environment import PAPER_TRADING
from botlib.environment import get_logger

# --------------------------------------------------------------------------------
# You can adjust these if you want a different exact time range:
START_TIME = datetime.datetime(2025, 1, 1)
END_TIME   = datetime.datetime(2025, 2, 14)
# --------------------------------------------------------------------------------

# Create needed directories
os.makedirs("input_cache", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("debug", exist_ok=True)
os.makedirs("training_data", exist_ok=True)

# We'll have scenario-based CSVs plus final:
CSV_GPT1  = os.path.join("output", "backtest_result_local_gpt_1.csv")
CSV_GPT2  = os.path.join("output", "backtest_result_local_gpt_2.csv")
CSV_FINAL = os.path.join("output", "backtest_result_final_signal.csv")

# training_data
TRAINING_DATA_FILE = os.path.join("training_data", "training_data_2025_2_14.csv")
RL_TRANSITIONS_FILE = os.path.join("training_data", "rl_transitions_2025_2_14.csv")

DO_USE_REAL_GPT = False
DO_USE_MODEL_PRED = True
BLOCK_SANTIMENT_FETCHING = True

# Global concurrency variable:
CONCURRENT_THREADS = 2
# We'll use a lock to prevent file write collisions:
file_write_lock = threading.Lock()


###############################################################################
# HistoricalTradingBot
###############################################################################
class HistoricalTradingBot(TradingBot):
    """
    A TradingBot subclass that doesn't store extra local historical_data.
    It overrides data fetch but calls `super()` to rely on
    caching from datafetchers.py if needed.
    """
    def __init__(self):
        super().__init__()
        # Force PAPER_TRADING
        self.simulated_balances = {"BTC": 0.0, "EUR": 10000.0}
        self.current_position = None
        self.last_iteration_data = {}
        self.logger = get_logger("HistoricalTradingBot")

        # We'll keep track of the previous iteration's RL state & action
        self.prev_rl_state_vec = None
        self.prev_action = None

    def set_current_datetime(self, dt: datetime.datetime):
        """
        The backtester calls this each iteration to indicate
        the hour we are simulating.
        """
        self.current_datetime = dt

    # Overridden fetch => pass self.current_datetime
    def fetch_klines(self, symbol="BTCEUR", interval=None, limit=60, end_dt:datetime.datetime=None):
        return super().fetch_klines(symbol, interval, limit, self.current_datetime)

    def fetch_price_at_hour(self, symbol="BTCEUR", dt: datetime.datetime = None):
        return super().fetch_price_at_hour(symbol, dt or self.current_datetime)

    def fetch_order_book(self, symbol="BTCEUR", limit=20, dt=None):
        return super().fetch_order_book(symbol, limit, dt or self.current_datetime)

    def fetch_news_data(self, days=1, end_dt=None):
        return super().fetch_news_data(days, end_dt or self.current_datetime)

    def fetch_google_trends(self, end_dt=None):
        return super().fetch_google_trends(end_dt or self.current_datetime)

    def fetch_santiment_data(self, end_dt=None, only_look_in_cache=False):
        return super().fetch_santiment_data(
            end_dt or self.current_datetime,
            BLOCK_SANTIMENT_FETCHING
        )

    def fetch_reddit_sentiment(self, dt=None):
        return super().fetch_reddit_sentiment(dt or self.current_datetime)
    
    def get_aggregated_signal(
        self,
        current_utc_time,
        current_price,
        klines_5m,
        klines_15m,
        klines_1h,
        news_articles,
        ob_data,
        google_trend,
        balances,
        santiment_data,
        reddit_sent,
        use_real_gpt,
        use_model_pred
    ):
        """
        Passes self.current_datetime into the aggregator for time context.
        """
        return super().get_aggregated_signal(
            self.current_datetime.astimezone(datetime.timezone.utc),
            current_price,
            klines_5m,
            klines_15m,
            klines_1h,
            news_articles,
            ob_data,
            google_trend,
            balances,
            santiment_data,
            reddit_sent,
            use_real_gpt,
            DO_USE_MODEL_PRED
        )
    
    def get_local_model_assessment(
        self,
        gpt_cache,
        tokenizer,
        model,
        model_type,
        type,
        prompt,
        temperature,
        timestamp,
        use_real_gpt
    ):
        timestamp = self.current_datetime
        
        return super().get_local_model_assessment(
            gpt_cache,
            tokenizer,
            model,
            model_type,
            type,
            prompt,
            temperature,
            timestamp,
            use_real_gpt = DO_USE_REAL_GPT
        )

    def run_cycle(self):
        """
        The normal TradingBot cycle. We'll store last_log_data => last_iteration_data
        for the backtester to use.
        Also copy RL state/action so the backtester can store them offline.
        """
        super().run_cycle()

        if hasattr(self, "last_log_data"):
            self.last_iteration_data = self.last_log_data.copy()

            # RL states for offline usage:
            self.last_iteration_data["old_rl_state"] = self.prev_rl_state_vec
            self.last_iteration_data["action"] = self.last_log_data.get("action", "HOLD")

            # Build new RL state for the next iteration
            final_signal = self.last_log_data.get("final_signal", 0.0)
            atr_percent  = self.last_log_data.get("atr_percent", 0.0)
            balances     = self.last_log_data.get("balances", {"BTC":0,"EUR":0})
            price        = self.last_log_data.get("current_price", 0.0)

            new_state_vec = None
            if price > 0:
                new_state_vec = self.build_rl_state(final_signal, atr_percent, balances, price)

            self.last_iteration_data["new_rl_state"] = new_state_vec
            self.last_iteration_data["reward"] = self.last_log_data.get("reward", 0.0)

            self.prev_rl_state_vec = new_state_vec
            self.prev_action = self.last_iteration_data["action"]
        else:
            self.last_iteration_data = {}


###############################################################################
# Backtester
###############################################################################
class Backtester:
    def __init__(self, start_time=None, end_time=None):
        self.logger = get_logger("Backtester")

        # RL Transitions
        with file_write_lock:
            if not os.path.exists(RL_TRANSITIONS_FILE):
                with open(RL_TRANSITIONS_FILE, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["old_state","action","reward","new_state","done"])

        # Our specialized historical bot
        self.bot = HistoricalTradingBot()
        self.bot.logger = self.logger

        if not PAPER_TRADING:
            self.logger.warning("Forcing PAPER_TRADING=True in backtester!")

        # If start_time/end_time are provided, use them; otherwise fallback
        self.start_time = start_time if start_time else START_TIME
        self.end_time   = end_time   if end_time   else END_TIME

        # Floor to the hour
        self.current_time = self.start_time.replace(minute=0, second=0, microsecond=0)

        # CSV init
        self.init_csv_scenarios()
        self.init_csv_training()

        # Buffer for training
        self.feature_buffer = {}

        # 3 local scenarios + final
        self.scenarios = {
            "local_gpt_1":  {"equity": 10000.0, "position": "NONE", "entry_price": None},
            "local_gpt_2":  {"equity": 10000.0, "position": "NONE", "entry_price": None},
            "final_signal": {"equity": 10000.0, "position": "NONE", "entry_price": None},
        }

        # We'll keep track of the old iteration data so we can store transitions
        self.prev_iteration_data = None

    def init_csv_scenarios(self):
        headers = [
            "timestamp","price","google_trend","reddit_sent",
            "santim_social_volume","news_sent","pred_y","action","equity"
        ]
        with file_write_lock:
            # If the file doesn't exist, write header. Otherwise, append
            for fn in [CSV_GPT1, CSV_GPT2, CSV_FINAL]:
                file_exists = os.path.exists(fn)
                mode = "a" if file_exists else "w"
                with open(fn, mode, newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(headers)

    def init_csv_training(self):
        """
        training_data.csv => columns:
          [timestamp, arr_5m, arr_15m, arr_1h,
           arr_google_trend, arr_santiment,
           arr_ta_63, arr_ctx_11,
           y]
        """
        with file_write_lock:
            if not os.path.exists(TRAINING_DATA_FILE):
                os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
                with open(TRAINING_DATA_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "timestamp", "arr_5m", "arr_15m", "arr_1h",
                        "arr_google_trend", "arr_santiment",
                        "arr_ta_63", "arr_ctx_11",
                        "y"
                    ])

    def run_backtest(self):
        self.logger.info(
            f"=== Starting Backtest from {self.start_time} to {self.end_time}, step=1h ==="
        )
        while self.current_time < self.end_time:
            self.logger.info(f"[Backtest] hour => {self.current_time}")
            self.bot.set_current_datetime(self.current_time)
            self.bot.run_cycle()  # normal TradingBot logic
            iteration_data = self.bot.last_iteration_data.copy()

            if iteration_data:
                # LSTM training data
                self.handle_training_buffer(iteration_data)
                # Scenario-based CSV logs
                self.handle_scenarios(iteration_data)

                # RL transitions
                old_state_vec = iteration_data.get("old_rl_state", None)
                next_state_vec= iteration_data.get("new_rl_state", None)
                reward = iteration_data.get("reward", 0.0)
                action = iteration_data.get("action", "HOLD")
                done = False

                if old_state_vec is not None and next_state_vec is not None:
                    with file_write_lock:
                        with open(RL_TRANSITIONS_FILE, "a", encoding="utf-8", newline="") as f:
                            w = csv.writer(f)
                            w.writerow([
                                json.dumps(old_state_vec.tolist()),
                                action,
                                reward,
                                json.dumps(next_state_vec.tolist()),
                                int(done)
                            ])

            self.current_time += datetime.timedelta(hours=1)
            self.prev_iteration_data = iteration_data

        self.logger.info("=== Backtest complete ===")

    def handle_training_buffer(self, iteration_data):
        """
        Store each iteration's data => after +3h, we get the actual future price,
        then we log that row to training_data.csv for supervised LSTM fitting.
        """
        now_ts = self.current_time.replace(minute=0, second=0, microsecond=0)
        now_str = now_ts.strftime("%Y-%m-%d %H:%M")

        rowdict = {
            "arr_5m": json.dumps(iteration_data.get("arr_5m", [])),
            "arr_15m": json.dumps(iteration_data.get("arr_15m", [])),
            "arr_1h": json.dumps(iteration_data.get("arr_1h", [])),
            "arr_google_trend": json.dumps(iteration_data.get("arr_google_trend", [])),
            "arr_santiment": json.dumps(iteration_data.get("arr_santiment", [])),
            "arr_ta_63": json.dumps(iteration_data.get("arr_ta_63", [])),
            "arr_ctx_11": json.dumps(iteration_data.get("arr_ctx_11", [])),
            "price": iteration_data.get("current_price", 0.0),
        }
        self.feature_buffer[now_str] = rowdict

        # Check data from 3h ago => compute future price change => log
        dt_ago = now_ts - datetime.timedelta(hours=3)
        dt_ago_str = dt_ago.strftime("%Y-%m-%d %H:%M")

        if dt_ago_str in self.feature_buffer:
            oldrow = self.feature_buffer[dt_ago_str]
            old_price = oldrow["price"]
            new_price = iteration_data.get("current_price", 0.0)

            if old_price > 0.0:
                pct = ((new_price - old_price)/old_price) * 100.0
                # clamp to [-1,1] for y
                ratio = max(-1, min(1, pct))
            else:
                ratio = 0.0

            with file_write_lock:
                with open(TRAINING_DATA_FILE, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        dt_ago_str,
                        oldrow["arr_5m"],
                        oldrow["arr_15m"],
                        oldrow["arr_1h"],
                        oldrow["arr_google_trend"],
                        oldrow["arr_santiment"],
                        oldrow["arr_ta_63"],
                        oldrow["arr_ctx_11"],
                        f"{ratio:.4f}"
                    ])

            # Remove that entry from buffer so it doesn't get reused
            del self.feature_buffer[dt_ago_str]

    def handle_scenarios(self, iteration_data):
        """
        Each scenario invests based on a single numeric signal:
          if signal > 10 => LONG, if signal < -10 => SHORT, else HOLD.
        """
        ts_str = self.current_time.strftime("%Y-%m-%d %H:%M")
        price = iteration_data.get("current_price", 0.0)

        # For google trend, check if array is non-empty
        arr_gt = iteration_data.get("arr_google_trend", [])
        if arr_gt and len(arr_gt) > 0 and len(arr_gt[0]) > 0:
            google_trend = arr_gt[0][-1][0]
        else:
            google_trend = 0.0

        reddit_sent = int(iteration_data.get("reddit_sent", 0))
        santim_social_volume_total = iteration_data.get("santiment", {}).get('social_volume_total', 0)

        # The aggregator's local_gpt_signals => [gpt1, gpt2]
        local_signals = iteration_data.get("local_gpt_signals", [0, 0])
        gpt1, gpt2 = local_signals
        gpt1 = round(gpt1, 2)
        gpt2 = round(gpt2, 2)

        news_signal  = round(iteration_data.get("news_signal", 0.0), 2)
        final_signal = round(iteration_data.get("final_signal", 0.0), 2)

        def update_scenario(scenario_key, sig_val):
            sc = self.scenarios[scenario_key]
            eq = sc["equity"]
            pos = sc["position"]
            ent = sc["entry_price"]
            act = "HOLD"

            if sig_val > 10:
                if pos == "NONE":
                    eq *= 0.999
                    sc["position"] = "LONG"
                    sc["entry_price"] = price
                    act = "OPEN_LONG"
                elif pos == "SHORT":
                    # close short => open long
                    ratio = ent / price
                    eq *= ratio
                    eq *= 0.999
                    sc["position"] = "LONG"
                    sc["entry_price"] = price
                    act = "CLOSE_SHORT_OPEN_LONG"
                else:
                    act = "KEEP_LONG"
            elif sig_val < -10:
                if pos == "NONE":
                    eq *= 0.999
                    sc["position"] = "SHORT"
                    sc["entry_price"] = price
                    act = "OPEN_SHORT"
                elif pos == "LONG":
                    ratio = price / ent
                    eq *= ratio
                    eq *= 0.999
                    sc["position"] = "SHORT"
                    sc["entry_price"] = price
                    act = "CLOSE_LONG_OPEN_SHORT"
                else:
                    act = "KEEP_SHORT"
            else:
                # close if open
                if pos == "LONG":
                    ratio = price / ent
                    eq *= ratio
                    eq *= 0.999
                    sc["position"] = "NONE"
                    sc["entry_price"] = None
                    act = "CLOSE_LONG"
                elif pos == "SHORT":
                    ratio = ent / price
                    eq *= ratio
                    eq *= 0.999
                    sc["position"] = "NONE"
                    sc["entry_price"] = None
                    act = "CLOSE_SHORT"
                else:
                    act = "HOLD_NONE"

            sc["equity"] = eq
            return act, eq

        # local_gpt_1
        a1, eq1 = update_scenario("local_gpt_1", gpt1)
        with file_write_lock:
            with open(CSV_GPT1, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts_str, price, google_trend, reddit_sent,
                    santim_social_volume_total, news_signal,
                    gpt1, a1, eq1
                ])

        # local_gpt_2
        a2, eq2 = update_scenario("local_gpt_2", gpt2)
        with file_write_lock:
            with open(CSV_GPT2, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts_str, price, google_trend, reddit_sent,
                    santim_social_volume_total, news_signal,
                    gpt2, a2, eq2
                ])

        # final_signal
        aF, eqF = update_scenario("final_signal", final_signal)
        with file_write_lock:
            with open(CSV_FINAL, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts_str, price, google_trend, reddit_sent,
                    santim_social_volume_total, news_signal,
                    final_signal, aF, eqF
                ])

    # Optional: Called only in single-thread mode
    def handle_ctrl_c(self, sig, frame):
        self.logger.info("=== Exiting gracefully ===\n")
        sys.exit(0)


def run_thread_chunk(thread_id, start_dt, end_dt):
    """
    Each thread runs its own Backtester over a portion of the time range.
    NOTE: We do not call signal.signal(...) here because Python only
    allows that in the main thread.
    """
    backtester = Backtester(start_time=start_dt, end_time=end_dt)
    backtester.logger.info(f"Thread {thread_id} => chunk: {start_dt} to {end_dt}")
    backtester.run_backtest()


def main():
    # Handle Ctrl+C in main thread
    def handle_main_ctrl_c(sig, frame):
        print("=== Ctrl+C intercepted (main thread) => exiting ===")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_main_ctrl_c)

    os.system('cls' if os.name == 'nt' else 'clear')  # clear terminal on Windows/*nix

    # Calculate total hours in the range
    total_hours = int((END_TIME - START_TIME).total_seconds() // 3600)
    if total_hours <= 0:
        # Fallback if date range is invalid or zero
        backtester = Backtester()
        backtester.run_backtest()
        return

    # Divide time range among CONCURRENT_THREADS
    chunk_size = math.ceil(total_hours / CONCURRENT_THREADS)
    threads = []

    for i in range(CONCURRENT_THREADS):
        chunk_start = START_TIME + datetime.timedelta(hours=(i * chunk_size))
        chunk_end   = START_TIME + datetime.timedelta(hours=((i+1) * chunk_size))
        if chunk_start >= END_TIME:
            break
        if chunk_end > END_TIME:
            chunk_end = END_TIME

        t = threading.Thread(
            target=run_thread_chunk,
            args=(i, chunk_start, chunk_end),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
