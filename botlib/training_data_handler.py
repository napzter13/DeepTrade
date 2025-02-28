import os
import csv
import json
import threading
import datetime
import schedule
import time
from pathlib import Path

from .environment import get_logger

class TrainingDataHandler:
    """
    Manages rolling training data for both the LSTM model and RL model:
      - Append a new sample every 20 minutes, storing them in CSV.
      - Keep a max of `max_days` in the CSV (prune older).
      - At `daily_training_time` (e.g. "16:00"), trigger a daily job
        that prunes the CSV and calls fitter for retraining.
    """

    def __init__(
        self,
        lstm_data_file="training_data/lstm_samples.csv",
        rl_data_file="training_data/rl_transitions.csv",
        max_days=35,
        daily_training_time="16:00"
    ):
        self.logger = get_logger("TrainingDataHandler")

        self.lstm_data_file = lstm_data_file
        self.rl_data_file = rl_data_file
        self.max_days = max_days

        # Lock for file I/O
        self._lock = threading.Lock()

        # Create directories if needed
        os.makedirs(os.path.dirname(self.lstm_data_file), exist_ok=True)
        self._init_lstm_csv()
        self._init_rl_csv()

        # Start a background thread => schedule daily training
        self._thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True
        )
        self._thread.start()

        # Schedule the daily job
        schedule.every().day.at(daily_training_time).do(self._daily_training_job)
        self.logger.info(f"TrainingDataHandler scheduled daily job at {daily_training_time} UTC.")

    def _init_lstm_csv(self):
        if not os.path.exists(self.lstm_data_file):
            self.logger.info(f"Creating new LSTM CSV => {self.lstm_data_file}")
            with open(self.lstm_data_file, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp",
                    "arr_5m",
                    "arr_15m",
                    "arr_1h",
                    "arr_google_trend",
                    "arr_santiment",
                    "arr_ta_63",
                    "arr_ctx_11",
                    "price"
                ])

    def _init_rl_csv(self):
        if not os.path.exists(self.rl_data_file):
            self.logger.info(f"Creating new RL CSV => {self.rl_data_file}")
            with open(self.rl_data_file, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp",
                    "old_state",
                    "action",
                    "reward",
                    "new_state",
                    "done"
                ])

    def add_lstm_sample(self, sample_dict):
        """
        Appends one new LSTM sample => CSV row
        sample_dict keys:
          "timestamp", "arr_5m", "arr_15m", "arr_1h", "arr_google_trend",
          "arr_santiment", "arr_ta_63", "arr_ctx_11", "price"
        """
        with self._lock:
            try:
                with open(self.lstm_data_file, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        sample_dict.get("timestamp", ""),
                        json.dumps(sample_dict.get("arr_5m", [])),
                        json.dumps(sample_dict.get("arr_15m", [])),
                        json.dumps(sample_dict.get("arr_1h", [])),
                        json.dumps(sample_dict.get("arr_google_trend", [])),
                        json.dumps(sample_dict.get("arr_santiment", [])),
                        json.dumps(sample_dict.get("arr_ta_63", [])),
                        json.dumps(sample_dict.get("arr_ctx_11", [])),
                        sample_dict.get("price", 0.0)
                    ])
            except Exception as e:
                self.logger.error(f"Error adding LSTM sample: {e}")

    def add_rl_transition(self, transition_dict):
        """
        Appends one RL transition => CSV row
        transition_dict keys:
          "timestamp", "old_state", "action", "reward", "new_state", "done"
        """
        with self._lock:
            try:
                with open(self.rl_data_file, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        transition_dict.get("timestamp", ""),
                        json.dumps(transition_dict.get("old_state", [])),
                        transition_dict.get("action", "HOLD"),
                        transition_dict.get("reward", 0.0),
                        json.dumps(transition_dict.get("new_state", [])),
                        int(transition_dict.get("done", False))
                    ])
            except Exception as e:
                self.logger.error(f"Error adding RL transition: {e}")

    def _daily_training_job(self):
        """
        1) Prune old rows in both CSVs (older than `self.max_days`).
        2) Call fitter to retrain model on the updated CSVs.
        """
        self.logger.info("[TrainingDataHandler] => daily training job started.")
        with self._lock:
            # prune old data
            self._prune_csv_older_than(self.lstm_data_file, self.max_days)
            self._prune_csv_older_than(self.rl_data_file, self.max_days, has_header=True)

        # Then call fitter script to retrain
        try:
            # If fitter.py is a module function:
            # fitter_main() 
            # or use:
            # os.system("python fitter.py --csv training_data/lstm_samples.csv --rl_csv training_data/rl_transitions.csv")
            # For example:
            os.system("python fitter.py --csv training_data/lstm_samples.csv --rl_csv training_data/rl_transitions.csv --no_scale")
            self.logger.info("[TrainingDataHandler] => daily training job DONE.")
        except Exception as e:
            self.logger.error(f"Error during daily training job: {e}")

    def _prune_csv_older_than(self, csv_file, days_to_keep=35, has_header=False):
        """
        Remove rows older than X days based on the first column (timestamp in ISO).
        If has_header=True, we skip the first row as a header.
        """
        if not os.path.exists(csv_file):
            return
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days_to_keep)
        keep_rows = []
        header = None

        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                if has_header:
                    header = next(reader, None)

                for row in reader:
                    if not row:
                        continue
                    ts_str = row[0].strip()
                    try:
                        dt = datetime.datetime.fromisoformat(ts_str)
                        if dt >= cutoff:
                            keep_rows.append(row)
                    except:
                        # If parse fails, keep row (maybe it's malformed => better not to lose data)
                        keep_rows.append(row)

            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if has_header and header:
                    w.writerow(header)
                w.writerows(keep_rows)

            self.logger.info(f"Pruned {csv_file}: kept {len(keep_rows)} rows out of total.")
        except Exception as e:
            self.logger.error(f"Error pruning {csv_file}: {e}")

    def _schedule_loop(self):
        """
        Background loop to handle schedule.run_pending() every 30s.
        """
        while True:
            schedule.run_pending()
            time.sleep(30)
