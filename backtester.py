#!/usr/bin/env python3
"""
backtester.py

- Iterates from the earliest date (or 1 month back if none specified)
  up to a user-specified end_date (default=now), stepping 1 hour at a time.
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

from botlib.tradebot import TradingBot
from botlib.environment import PAPER_TRADING
from botlib.environment import get_logger

# Create needed directories
os.makedirs("input_cache", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("debug", exist_ok=True)  # for sanity-check logs

# We'll have 4 scenario-based CSVs plus final:
CSV_GPT1  = os.path.join("output", "backtest_result_local_gpt_1.csv")
CSV_GPT2  = os.path.join("output", "backtest_result_local_gpt_2.csv")
CSV_FINAL = os.path.join("output", "backtest_result_final_signal.csv")

# training_data + a sanity-check file
TRAINING_DATA_FILE = os.path.join("output", "training_data.csv")
RL_TRANSITIONS_FILE = os.path.join("output", "rl_transitions.csv")


# Start/end
START_TIME = datetime.datetime(2024, 1, 1)
END_TIME = datetime.datetime.now()
        
DO_USE_REAL_GPT = False
DO_USE_MODEL_PRED = False
BLOCK_SANTIMENT_FETCHING = True


###############################################################################
# HistoricalTradingBot
###############################################################################
class HistoricalTradingBot(TradingBot):
    """
    A TradingBot subclass that doesn't store extra local historical_data.
    It overrides data fetch but simply calls `super()` to rely on
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
        return super().fetch_santiment_data(end_dt or self.current_datetime, BLOCK_SANTIMENT_FETCHING)

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

        We also copy out the RL state and action so the backtester can store them offline.
        """
        super().run_cycle()

        if hasattr(self, "last_log_data"):
            self.last_iteration_data = self.last_log_data.copy()

            # We'll keep track of RL states for offline usage:
            self.last_iteration_data["old_rl_state"] = self.prev_rl_state_vec
            self.last_iteration_data["action"] = self.last_log_data.get("action", "HOLD")

            # The TradingBot's run_cycle built a new state for the next iteration
            # but we didn't store it. We'll replicate building it here to capture
            # the new state, so the backtester can store it as next_state.
            final_signal = self.last_log_data.get("final_signal", 0.0)
            atr_percent  = self.last_log_data.get("atr_percent", 0.0)
            balances     = self.last_log_data.get("balances", {"BTC":0,"EUR":0})
            price        = self.last_log_data.get("current_price", 0.0)

            new_state_vec = None
            if price>0:
                new_state_vec = self.build_rl_state(final_signal, atr_percent, balances, price)

            self.last_iteration_data["new_rl_state"] = new_state_vec
            # Also store the "reward" from the last iteration:
            self.last_iteration_data["reward"] = self.last_log_data.get("reward", 0.0)

            # We'll update prev_rl_state_vec for next iteration:
            self.prev_rl_state_vec = new_state_vec
            self.prev_action = self.last_iteration_data["action"]
        else:
            self.last_iteration_data = {}


###############################################################################
# Backtester
###############################################################################
class Backtester:
    def __init__(self):
        self.logger = get_logger("Backtester")

        # RL Transitions
        if not os.path.exists(RL_TRANSITIONS_FILE):
            with open(RL_TRANSITIONS_FILE, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["old_state","action","reward","new_state","done"])


        # Our specialized historical bot
        self.bot = HistoricalTradingBot()
        self.bot.logger = self.logger

        if not PAPER_TRADING:
            self.logger.warning("Forcing PAPER_TRADING=True in backtester!")

        # We'll floor current_time to the hour
        self.current_time = START_TIME.replace(minute=0, second=0, microsecond=0)

        # CSV init
        self.init_csv_scenarios()
        self.init_csv_training()

        # Buffer for training
        self.feature_buffer = {}

        # 4 GPT scenarios + final
        self.scenarios = {
            "local_gpt_1":  {"equity": 10000.0, "position": "NONE", "entry_price": None},
            "local_gpt_2":  {"equity": 10000.0, "position": "NONE", "entry_price": None},
            "final_signal": {"equity": 10000.0, "position": "NONE", "entry_price": None},
        }

        # We'll keep track of the old iteration data so we can store transitions
        self.prev_iteration_data = None

    def init_csv_scenarios(self):
        headers = ["timestamp","price","google_trend","reddit_sent","santim_social_volume","news_sent","pred_y","action","equity"]
        for fn in [CSV_GPT1, CSV_GPT2, CSV_FINAL]:
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w=csv.writer(f)
                w.writerow(headers)

    def init_csv_training(self):
        """
        training_data.csv => columns: [
           timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend,
           arr_santiment, arr_ta_63, arr_ctx_11, y
        ]
        """
        if not os.path.exists(TRAINING_DATA_FILE):
            os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
            with open(TRAINING_DATA_FILE,"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f)
                w.writerow([
                    "timestamp", "arr_5m", "arr_15m", "arr_1h",
                    "arr_google_trend", "arr_santiment",
                    "arr_ta_63", "arr_ctx_11",
                    "y"
                ])

    def run_backtest(self):
        self.logger.info(f"=== Starting Backtest from {START_TIME} to {END_TIME} stepping 1h ===")
        while self.current_time <= END_TIME:
            self.logger.info(f"[Backtest] hour => {self.current_time}")
            self.bot.set_current_datetime(self.current_time)
            self.bot.run_cycle()  # normal TradingBot logic
            iteration_data = self.bot.last_iteration_data.copy()

            if iteration_data:
                # Build training data for LSTM
                self.handle_training_buffer(iteration_data)
                # Build scenario-based CSV logs
                self.handle_scenarios(iteration_data)

                # ------------------- RL    ---------------------------------
                old_state_vec = iteration_data.get("old_rl_state", None)
                next_state_vec= iteration_data.get("new_rl_state", None)
                reward = iteration_data.get("reward", 0.0)
                action = iteration_data.get("action", "HOLD")
                done = False

                if old_state_vec is not None and next_state_vec is not None:
                    # append row to rl_transitions.csv
                    with open(RL_TRANSITIONS_FILE, "a", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([
                            json.dumps(old_state_vec.tolist()), 
                            action,
                            reward,
                            json.dumps(next_state_vec.tolist()),
                            int(done)
                        ])
                # -----------------------------------------------------------

            self.current_time += datetime.timedelta(hours=1)
            self.prev_iteration_data = iteration_data

        self.logger.info("=== Backtest complete ===")

    def handle_training_buffer(self, iteration_data):
        """
        Store each iteration's data => after +3h we get the actual future price,
        then we log to training_data.csv for supervised LSTM fitting.
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

        # Check if we have data from 3h ago => we can compute the future price change
        dt_ago = now_ts - datetime.timedelta(hours=3)
        dt_ago_str = dt_ago.strftime("%Y-%m-%d %H:%M")

        if dt_ago_str in self.feature_buffer:
            oldrow = self.feature_buffer[dt_ago_str]
            old_price = oldrow["price"]
            new_price = iteration_data.get("current_price", 0.0)

            if old_price > 0.0:
                pct = ((new_price - old_price)/old_price) * 100.0
                ratio = pct
                if ratio > 1:
                    ratio = 1
                if ratio < -1:
                    ratio = -1
                y_val = ratio
            else:
                y_val = 0.0

            with open(TRAINING_DATA_FILE, "a", newline="", encoding="utf-8") as f:
                w=csv.writer(f)
                w.writerow([
                    dt_ago_str,
                    oldrow["arr_5m"],
                    oldrow["arr_15m"],
                    oldrow["arr_1h"],
                    oldrow["arr_google_trend"],
                    oldrow["arr_santiment"],
                    oldrow["arr_ta_63"],
                    oldrow["arr_ctx_11"],
                    f"{y_val:.4f}"
                ])

            # Remove that entry from buffer so it doesn't get re-used
            del self.feature_buffer[dt_ago_str]

    def handle_scenarios(self, iteration_data):
        """
        Each scenario invests based on a single numeric signal:
          if signal>10 => LONG, if<-10 => SHORT, else HOLD.
        We log each scenario's action & equity to CSV.
        """
        ts_str = self.current_time.strftime("%Y-%m-%d %H:%M")
        price = iteration_data.get("current_price", 0.0)
        
        # For google trend, check if array is non-empty
        arr_gt = iteration_data.get("arr_google_trend", [])
        if arr_gt and len(arr_gt) > 0 and len(arr_gt[0]) > 0:
            google_trend = arr_gt[0][-1][0]
        else:
            google_trend = 0.0

        reddit_sent  = int(iteration_data.get("reddit_sent", 0))
        santim_social_volume_total       = iteration_data.get("santiment", {})['social_volume_total']

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

            if sig_val>10:
                if pos=="NONE":
                    # open LONG
                    eq *= 0.999  # sim fee or slip
                    sc["position"]="LONG"
                    sc["entry_price"]=price
                    act="OPEN_LONG"
                elif pos=="SHORT":
                    # close short => open long
                    ratio= ent/price
                    eq *= ratio
                    eq *= 0.999
                    sc["position"]="LONG"
                    sc["entry_price"]=price
                    act="CLOSE_SHORT_OPEN_LONG"
                else:
                    act="KEEP_LONG"
            elif sig_val<-10:
                if pos=="NONE":
                    eq *= 0.999
                    sc["position"]="SHORT"
                    sc["entry_price"]=price
                    act="OPEN_SHORT"
                elif pos=="LONG":
                    ratio= price/ent
                    eq *= ratio
                    eq *= 0.999
                    sc["position"]="SHORT"
                    sc["entry_price"]=price
                    act="CLOSE_LONG_OPEN_SHORT"
                else:
                    act="KEEP_SHORT"
            else:
                # hold => maybe close existing
                if pos=="LONG":
                    ratio= price/ent
                    eq *= ratio
                    eq *= 0.999
                    sc["position"]="NONE"
                    sc["entry_price"]=None
                    act="CLOSE_LONG"
                elif pos=="SHORT":
                    ratio= ent/price
                    eq *= ratio
                    eq *= 0.999
                    sc["position"]="NONE"
                    sc["entry_price"]=None
                    act="CLOSE_SHORT"
                else:
                    act="HOLD_NONE"

            sc["equity"]= eq
            return act, eq

        # local_gpt_1
        a1, eq1 = update_scenario("local_gpt_1", gpt1)
        with open(CSV_GPT1, "a", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([
                ts_str, price, google_trend, reddit_sent,
                santim_social_volume_total, news_signal, gpt1, a1, eq1
            ])

        # local_gpt_2
        a2, eq2 = update_scenario("local_gpt_2", gpt2)
        with open(CSV_GPT2, "a", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([
                ts_str, price, google_trend, reddit_sent,
                santim_social_volume_total, news_signal, gpt2, a2, eq2
            ])

        # final_signal
        aF, eqF = update_scenario("final_signal", final_signal)
        with open(CSV_FINAL, "a", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([
                ts_str, price, google_trend, reddit_sent,
                santim_social_volume_total, news_signal, final_signal, aF, eqF
            ])

    # Handle exit
    def handle_ctrl_c(self, sig, frame):
        """
        Custom handler for CTRL+C (SIGINT).
        """
        self.logger.info("=== Exiting gracefully ===\n")
        
        sys.exit(0)



def main():
    os.system('cls')  # clear terminal on Windows (no-op on other OS)
    backtester = Backtester()
    signal.signal(signal.SIGINT, backtester.handle_ctrl_c)
    backtester.run_backtest()

if __name__ == "__main__":
    main()
