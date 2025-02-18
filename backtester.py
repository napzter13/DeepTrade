#!/usr/bin/env python3
"""
backtester.py

This version:
- Iterates from START_TIME up to END_TIME, stepping every STEP_INTERVAL_MINUTES (default 20 min).
- For each step:
   1) We set the HistoricalTradingBot to that 'current_time'.
   2) We run the normal TradingBot logic => produce last_log_data (including signals).
   3) We store those signals in scenario-based CSV logs (if single-thread).
   4) We buffer feature arrays for building the training_data with multi-output y_1..y_10.
   5) We also buffer RL transitions, so that each transition is created only if we have
      +10-step future price to compute a reward.

At the end:
- We have 3 scenario-based CSVs (local_gpt_1, local_gpt_2, final_signal).
- We produce "training_data.csv" with columns:
   [timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta_63, arr_ctx_11,
    y_1, y_2, ..., y_10]
  each y_i is the future %change from t0 to t0 + i steps (clamped to [-1,1]).

- We produce "rl_transitions.csv" with columns:
   [old_state, action, reward, new_state, done]
  where 'reward' is computed from the price difference after 10 steps (simple approach).
"""

import os
import csv
import json
import datetime
import sys
import signal
import math
import threading
from binance.client import Client

from botlib.tradebot import TradingBot
from botlib.environment import (
    PAPER_TRADING,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    STEP_INTERVAL_MINUTES,
    NUM_FUTURE_STEPS,
    get_logger
)

###############################################################################
# CONFIG
###############################################################################
MAKE_NN_TRAINING_DATA = True
MAKE_RL_TRAINING_DATA = False       # Only True when NN model is trained already.
BLOCK_SANTIMENT_FETCHING = True
DO_USE_REAL_GPT = False
DO_USE_MODEL_PRED = False

# Adjust these for the time range:
START_TIME = datetime.datetime(2024, 1, 1)
END_TIME   = datetime.datetime(2025, 1, 1)

# If >1, we only reliably gather training_data. (Scenario logs may be duplicated.)
CONCURRENT_THREADS = 1  # or 3, etc.

# Directory structure
os.makedirs("input_cache", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("debug", exist_ok=True)
os.makedirs("training_data", exist_ok=True)

# CSV outputs
TRAINING_DATA_FILE = os.path.join(
    "training_data",
    f"training_data_{START_TIME:%Y_%m_%d}_to_{END_TIME:%Y_%m_%d}.csv"
)
RL_TRANSITIONS_FILE = os.path.join(
    "training_data",
    f"rl_transitions_{START_TIME:%Y_%m_%d}_to_{END_TIME:%Y_%m_%d}.csv"
)
CSV_GPT1  = os.path.join("output", "backtest_result_local_gpt_1.csv")
CSV_GPT2  = os.path.join("output", "backtest_result_local_gpt_2.csv")
CSV_FINAL = os.path.join("output", "backtest_result_final_signal.csv")

# Lock for file writing
file_write_lock = threading.Lock()
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


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
        self.binance_client = binance_client
        super().__init__()

        # Force PAPER_TRADING
        self.simulated_balances = {"BTC": 0.0, "EUR": 10000.0}
        self.current_position = None
        self.last_iteration_data = {}
        self.logger = get_logger("HistoricalTradingBot")

        # Track the previous iteration's RL state & action
        self.prev_rl_state_vec = None
        self.prev_action = None

    def set_current_datetime(self, dt: datetime.datetime):
        """
        The backtester calls this each iteration to set the time we simulate.
        """
        self.current_datetime = dt

    # Overridden fetch => pass self.current_datetime to super
    def fetch_klines(self, symbol="BTCEUR", interval=None, limit=241, end_dt:datetime.datetime=None):
        return super().fetch_klines(symbol, interval, limit, self.current_datetime)

    def fetch_price_at_hour(self, symbol="BTCEUR", dt: datetime.datetime = None):
        return super().fetch_price_at_hour(symbol, dt or self.current_datetime)

    def fetch_order_book(self, symbol="BTCEUR", limit=20, dt=None):
        return super().fetch_order_book(symbol, limit, dt or self.current_datetime)

    def fetch_news_data(self, days=1, topic="Bitcoin", end_dt=None):
        return super().fetch_news_data(days, topic, end_dt or self.current_datetime)

    def fetch_google_trends(self, end_dt=None, topic="Bitcoin"):
        return super().fetch_google_trends(end_dt or self.current_datetime, topic)

    def fetch_santiment_data(self, end_dt=None, topic="bitcoin", only_look_in_cache=False):
        return super().fetch_santiment_data(
            end_dt or self.current_datetime,
            topic,
            BLOCK_SANTIMENT_FETCHING
        )

    def fetch_reddit_sentiment(self, dt=None, topic="Bitcoin"):
        return super().fetch_reddit_sentiment(dt or self.current_datetime, topic)
    
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
            gpt_cache, tokenizer, model, model_type, type, prompt,
            temperature, timestamp,
            use_real_gpt = DO_USE_REAL_GPT
        )

    def run_cycle(self):
        """
        Normal TradingBot cycle. We'll store last_log_data => last_iteration_data
        and copy RL state/action so the backtester can store them.
        """
        super().run_cycle()

        if hasattr(self, "last_log_data"):
            self.last_iteration_data = self.last_log_data.copy()

            # RL: store old state & action in last_iteration_data
            self.last_iteration_data["old_rl_state"] = self.prev_rl_state_vec
            self.last_iteration_data["action"] = self.last_log_data.get("action", "HOLD")

            # Build new RL state
            final_signal = self.last_log_data.get("final_signal", 0.0)
            atr_percent  = self.last_log_data.get("atr_percent", 0.0)
            balances     = self.last_log_data.get("balances", {"BTC": 0, "EUR": 0})
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
    def __init__(self, start_time=None, end_time=None, skip_until=None):
        self.logger = get_logger("Backtester")
        self.bot = HistoricalTradingBot()
        self.bot.logger = self.logger

        if not PAPER_TRADING:
            self.logger.warning("Forcing PAPER_TRADING=True in backtester!")

        self.start_time = start_time or START_TIME
        self.end_time   = end_time   or END_TIME
        self.skip_until = skip_until
        
        # We'll floor to the nearest minute for start
        self.current_time = self.start_time.replace(second=0, microsecond=0)

        # CSV init
        self.init_csv_scenarios()
        if MAKE_NN_TRAINING_DATA:
            self.init_csv_training()

        # Buffers:
        self.feature_buffer = {}   # For multi-output LSTM data
        self.rl_buffer      = {}   # For RL transitions

        # Scenario equity tracking
        self.scenarios = {
            "local_gpt_1":  {"equity": 10000.0, "position": "NONE", "entry_price": None},
            "local_gpt_2":  {"equity": 10000.0, "position": "NONE", "entry_price": None},
            "final_signal": {"equity": 10000.0, "position": "NONE", "entry_price": None},
        }
        self.prev_iteration_data = None

    def init_csv_scenarios(self):
        headers = [
            "timestamp","price","google_trend","reddit_sent",
            "santim_social_volume","news_sent","pred_y","action","equity"
        ]
        with file_write_lock:
            for fn in [CSV_GPT1, CSV_GPT2, CSV_FINAL]:
                mode = "w"
                with open(fn, mode, newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(headers)

    def init_csv_training(self):
        """
        We store 10 outputs: y_1..y_10
        """
        with file_write_lock:
            os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
            with open(TRAINING_DATA_FILE, "w", newline="", encoding="utf-8") as f:
                # columns: [timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend,
                #           arr_santiment, arr_ta_63, arr_ctx_11,
                #           y_1, y_2, ..., y_10]
                columns = [
                    "timestamp",
                    "arr_5m", "arr_15m", "arr_1h",
                    "arr_google_trend", "arr_santiment",
                    "arr_ta_63", "arr_ctx_11",
                ]
                y_cols = [f"y_{i}" for i in range(1, NUM_FUTURE_STEPS+1)]
                columns.extend(y_cols)
                w = csv.writer(f)
                w.writerow(columns)

            # RL transitions:
            if MAKE_RL_TRAINING_DATA:
                with open(RL_TRANSITIONS_FILE, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["old_state","action","reward","new_state","done"])

    def run_backtest(self):
        self.logger.info(
            f"=== Starting Backtest from {self.start_time} to {self.end_time}, "
            f"step={STEP_INTERVAL_MINUTES} min, future_steps={NUM_FUTURE_STEPS} ==="
        )
        step_delta = datetime.timedelta(minutes=STEP_INTERVAL_MINUTES)

        while self.current_time < self.end_time:
            if shutdown_event.is_set():
                self.logger.info("=== Exiting gracefully ===")
                return
                
            self.logger.info(f"[Backtest] => {self.current_time}")
            self.bot.set_current_datetime(self.current_time)
            self.bot.run_cycle()  # normal TradingBot logic
            iteration_data = self.bot.last_iteration_data.copy()

            if iteration_data:
                if MAKE_NN_TRAINING_DATA:
                    self.handle_training_buffer(iteration_data)

                # Only write scenario logs if single-thread to avoid duplicates
                if CONCURRENT_THREADS == 1:
                    self.handle_scenarios(iteration_data)

                if MAKE_RL_TRAINING_DATA:
                    self.handle_rl_buffer(iteration_data)

            self.current_time += step_delta
            self.prev_iteration_data = iteration_data

        self.logger.info("=== Backtest complete ===")


    ###########################################################################
    # Multi-output training data
    ###########################################################################
    def handle_training_buffer(self, iteration_data):
        """
        We store the current iteration's features in a buffer keyed by now_ts.
        Then we check if there's an entry in the buffer from T0 = now_ts - 10 steps.
        If so, we see if we can find the prices for T0+1, T0+2, ..., T0+10 steps.
        If not all present, skip. If all present, write one row with y_1..y_10.
        Then remove T0 from the buffer.
        """
        now_ts = self.current_time.replace(second=0, microsecond=0)
        if self.skip_until and now_ts < self.skip_until:
            return
        
        # 1) Store data from this iteration
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
        self.feature_buffer[now_ts] = rowdict

        # 2) Check if we have an entry from T0 = now_ts - 10 steps
        step_delta = datetime.timedelta(minutes=STEP_INTERVAL_MINUTES)
        T0 = now_ts - (step_delta * NUM_FUTURE_STEPS)

        if T0 not in self.feature_buffer:
            return

        # We have T0's row, let's see if T0+ i*delta for i=1..10 is also in buffer
        future_prices = []
        for i in range(1, NUM_FUTURE_STEPS+1):
            t_i = T0 + i * step_delta
            if t_i not in self.feature_buffer:
                # not all future steps are present => skip
                return
            future_prices.append(self.feature_buffer[t_i]["price"])

        # All future steps present => build y_1..y_10
        old_price = self.feature_buffer[T0]["price"]
        if old_price <= 0:
            # trivial or invalid price => skip
            return

        y_vals = []
        for i in range(NUM_FUTURE_STEPS):
            pct = ((future_prices[i] - old_price) / old_price) * 100.0
            # clamp to [-1,1]
            clamp = max(-1.0, min(1.0, pct))
            y_vals.append(clamp)

        # 3) Write a row for T0
        T0_str = T0.strftime("%Y-%m-%d %H:%M")
        oldrow = self.feature_buffer[T0]
        with file_write_lock:
            with open(TRAINING_DATA_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    T0_str,
                    oldrow["arr_5m"],
                    oldrow["arr_15m"],
                    oldrow["arr_1h"],
                    oldrow["arr_google_trend"],
                    oldrow["arr_santiment"],
                    oldrow["arr_ta_63"],
                    oldrow["arr_ctx_11"],
                    *[f"{val:.4f}" for val in y_vals]
                ])

        # 4) Remove T0 from the buffer so it doesn't get reused
        del self.feature_buffer[T0]

    ###########################################################################
    # RL transitions with 10-step future
    ###########################################################################
    def handle_rl_buffer(self, iteration_data):
        """
        Single-step RL transitions:
        - For the current iteration, store (old_state, action, price, new_state) in self.rl_buffer.
        - Then look up T_prev = current_time - 1 step to finalize that transition.
        - Reward is price difference in % from T_prev to current_time.
        """
        now_ts = self.current_time
        old_state = iteration_data.get("old_rl_state", None)
        action    = iteration_data.get("action", "HOLD")
        new_state = iteration_data.get("new_rl_state", None)
        curr_price= iteration_data.get("current_price", 0.0)

        # 1) Store the current iteration’s data in rl_buffer
        self.rl_buffer[now_ts] = {
            "old_state": old_state,
            "action": action,
            "new_state": new_state,
            "price": curr_price,
        }

        # 2) Look 1 step behind
        step_delta = datetime.timedelta(minutes=STEP_INTERVAL_MINUTES)
        T_prev = now_ts - step_delta
        if T_prev not in self.rl_buffer:
            return  # no previous step => skip

        # 3) Finalize the previous transition
        prev_data = self.rl_buffer[T_prev]
        old_price = prev_data["price"]
        if old_price <= 0:
            # invalid or zero price => skip
            return

        # Reward is %change from T_prev price to current price
        reward_pct = ((curr_price - old_price) / old_price) * 100.0
        reward = float(reward_pct)

        # old_state, action come from T_prev
        old_state_arr = prev_data["old_state"]
        action_str    = prev_data["action"]
        # next_state can be the *current* iteration's old_state
        # (typical RL: next_state = the next observation)
        next_state_arr = old_state

        done = 0  # or define a condition for done

        # 4) Write out the transition if states are valid
        if old_state_arr is not None and next_state_arr is not None:
            with file_write_lock:
                with open(RL_TRANSITIONS_FILE, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        json.dumps(old_state_arr.tolist()),
                        action_str,
                        reward,
                        json.dumps(next_state_arr.tolist()),
                        done
                    ])

        # 5) Remove the previous entry from the buffer (we've finalized it).
        del self.rl_buffer[T_prev]


    ###########################################################################
    # Scenario logs
    ###########################################################################
    def handle_scenarios(self, iteration_data):
        ts_str = self.current_time.strftime("%Y-%m-%d %H:%M")
        price = iteration_data.get("current_price", 0.0)

        arr_gt = iteration_data.get("arr_google_trend", [])
        if arr_gt and len(arr_gt) > 0 and len(arr_gt[0]) > 0:
            google_trend = arr_gt[0][-1][0]
        else:
            google_trend = 0.0

        reddit_sent = int(iteration_data.get("reddit_sent", 0))
        santim_social_volume_total = iteration_data.get("santiment", {}).get('social_volume_total', 0)

        # aggregator local_gpt_signals => [gpt1, gpt2]
        local_signals = iteration_data.get("local_gpt_signals", [0, 0])
        gpt1, gpt2 = local_signals
        gpt1 = round(gpt1, 2)
        gpt2 = round(gpt2, 2)

        news_signal  = round(iteration_data.get("news_signal", 0.0), 2)
        
        final_signal_list = iteration_data.get("final_signal", [0.0]*NUM_FUTURE_STEPS)
        final_signal_val = final_signal_list[-1]
        final_signal = round(final_signal_val, 2)

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


###############################################################################
# Multi-thread chunk logic
###############################################################################
def run_thread_chunk(thread_id, start_dt, end_dt, skip_until):
    backtester = Backtester(
        start_time=start_dt,
        end_time=end_dt,
        skip_until=skip_until
    )
    backtester.logger.info(f"[Thread {thread_id}] range: {start_dt} to {end_dt}, skip_until={skip_until}")
    backtester.run_backtest()


def main():
    def handle_main_ctrl_c(sig, frame):
        print("=== Ctrl+C intercepted (main thread) => exiting ===")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_main_ctrl_c)

    os.system('cls' if os.name == 'nt' else 'clear')

    # total_steps in minutes
    total_minutes = int((END_TIME - START_TIME).total_seconds() // 60)
    step_len = STEP_INTERVAL_MINUTES
    if total_minutes <= 0:
        backtester = Backtester()
        backtester.run_backtest()
        return

    total_steps = total_minutes // step_len
    chunk_size = math.ceil(total_steps / CONCURRENT_THREADS)
    threads = []

    for i in range(CONCURRENT_THREADS):
        chunk_start_step = i * chunk_size
        chunk_end_step   = (i+1) * chunk_size

        # Start/end times
        chunk_start = START_TIME + datetime.timedelta(minutes=chunk_start_step*step_len)
        chunk_end   = START_TIME + datetime.timedelta(minutes=chunk_end_step*step_len)
        if chunk_start >= END_TIME:
            break
        if chunk_end > END_TIME:
            chunk_end = END_TIME

        # Overlap for multi-step
        if i < CONCURRENT_THREADS - 1:
            chunk_end += datetime.timedelta(minutes=step_len * NUM_FUTURE_STEPS)
            if chunk_end > END_TIME:
                chunk_end = END_TIME

        skip_until = None
        if i > 0:
            skip_until = chunk_start + datetime.timedelta(minutes=step_len*NUM_FUTURE_STEPS)
            if skip_until > END_TIME:
                skip_until = END_TIME

        t = threading.Thread(
            target=run_thread_chunk,
            args=(i, chunk_start, chunk_end, skip_until),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()



shutdown_event = threading.Event()
def signal_handler(sig, frame):
    print("=== Exiting gracefully ===")
    shutdown_event.set()  # Signal all threads to exit
    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    main()
