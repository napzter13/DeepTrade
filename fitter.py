#!/usr/bin/env python3
"""
fitter.py

- Multi-output LSTM with 10 outputs (y_1..y_10).
- Single-step RL that uses the 10 LSTM outputs as the state dimension.
- We assume the training CSV has columns:
     timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta_63, arr_ctx_11,
     y_1, y_2, ..., y_10
  each y_i in [-1,1].

Usage:

python fitter.py \
  --csv training_data/training_data.csv \
  --model_out models/advanced_lstm_model.keras \
  --rl_csv training_data/rl_transitions.csv \
  --rl_out models/rl_DQNAgent.weights.h5 \
  --rl_state_dim 10
"""

import os
import json
import csv
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from botlib.environment import (
    NUM_FUTURE_STEPS,
)
# Import advanced LSTM model builder (must produce Dense(10) final layer!)
from botlib.models import load_advanced_lstm_model

# For normalizing input data
from botlib.input_preprocessing import ModelScaler, prepare_for_model_inputs

# RL DQN
from botlib.rl import DQNAgent, ACTIONS

mixed_precision.set_global_policy("mixed_float16")

RL_TRANSITIONS_FILE = os.path.join("training_data", "rl_transitions.csv")

class Trainer:
    def __init__(
        self,
        training_csv="training_data/training_data.csv",
        model_out="models/advanced_lstm_model.keras",
        window_5m=241,
        feature_5m=9,
        window_15m=241,
        feature_15m=9,
        window_1h=241,
        feature_1h=9,
        window_google_trend=24,
        feature_google_trend=1,
        santiment_dim=12,
        ta_dim=63,
        signal_dim=11,
        epochs=1000,
        early_stop_patience=10,
        batch_size=64,
        apply_scaling=True,
        train_ratio=0.8,
        val_ratio=0.2,
        skip_lstm=False,
        max_rows=0
    ):
        """
        :param training_csv:  Path to training_data.csv from the backtester
                              which must have y_1..y_{NUM_FUTURE_STEPS} columns.
        :param model_out:     LSTM model output path
        :param skip_lstm:     If True, skip LSTM training entirely.
        ...
        """
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self.logger.addHandler(ch)

        self.training_csv = training_csv
        self.model_out = model_out
        self.window_5m = window_5m
        self.feature_5m= feature_5m
        self.window_15m= window_15m
        self.feature_15m= feature_15m
        self.window_1h = window_1h
        self.feature_1h= feature_1h
        self.window_google_trend = window_google_trend
        self.feature_google_trend= feature_google_trend
        self.santiment_dim = santiment_dim
        self.ta_dim = ta_dim
        self.signal_dim = signal_dim
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size
        self.apply_scaling= apply_scaling
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.skip_lstm   = skip_lstm
        self.max_rows   = max_rows

        if not skip_lstm:
            self.logger.info(f"Initializing multi-timeframe LSTM model with {NUM_FUTURE_STEPS} outputs.")
            self.model = load_advanced_lstm_model(
                model_5m_window=self.window_5m,
                model_15m_window=self.window_15m,
                model_1h_window=self.window_1h,
                feature_dim=self.feature_5m,
                santiment_dim=self.santiment_dim,
                ta_dim=self.ta_dim,
                signal_dim=self.signal_dim
            )
            # compile with MSE for multi-output regression
            # self.model.compile(optimizer='adam', loss='mse')
        else:
            self.logger.info("Skipping LSTM model initialization (skip_lstm=True).")
            self.model = None

    def load_training_data(self):
        """
        Expects CSV columns:
          timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend,
          arr_santiment, arr_ta_63, arr_ctx_11,
          y_1, y_2, ..., y_{NUM_FUTURE_STEPS}

        We'll parse them into Python lists => build big NumPy arrays.
        The final Y will be shape (N, NUM_FUTURE_STEPS).
        """
        if not os.path.exists(self.training_csv):
            self.logger.error(f"No training CSV at {self.training_csv}")
            return [None]*9  # consistent but empty

        all_5m   = []
        all_15m  = []
        all_1h   = []
        all_gt   = []
        all_sa   = []
        all_ta   = []
        all_ctx  = []
        all_Y    = []
        all_ts   = []
        
        with open(self.training_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing_cols = []
            # check columns exist
            for i in range(1, NUM_FUTURE_STEPS+1):
                col_name = f"y_{i}"
                if col_name not in reader.fieldnames:
                    missing_cols.append(col_name)
            if missing_cols:
                self.logger.error(f"CSV missing columns: {missing_cols}")
                return [None]*9
            
            a = 0
            for row in reader:
                if self.max_rows > 0 and a >= self.max_rows:
                    break

                try:
                    timestamp_str = row["timestamp"]
                    arr_5m_str  = row["arr_5m"]
                    arr_15m_str = row["arr_15m"]
                    arr_1h_str  = row["arr_1h"]
                    arr_gt_str  = row["arr_google_trend"]
                    arr_sa_str  = row["arr_santiment"]
                    arr_ta_str  = row["arr_ta_63"]
                    arr_ctx_str = row["arr_ctx_11"]

                    # parse JSON
                    arr_5m_list   = json.loads(arr_5m_str)[0]
                    arr_15m_list  = json.loads(arr_15m_str)[0]
                    arr_1h_list   = json.loads(arr_1h_str)[0]
                    arr_gt_list   = json.loads(arr_gt_str)[0]
                    arr_sa_list   = json.loads(arr_sa_str)[0]
                    arr_ta_list   = json.loads(arr_ta_str)[0]
                    arr_ctx_list  = json.loads(arr_ctx_str)[0]

                    if len(arr_5m_list)!=241:  continue
                    if len(arr_15m_list)!=241: continue
                    if len(arr_1h_list)!=241:  continue
                    if len(arr_gt_list)!=24:   continue
                    if len(arr_sa_list)!=12:   continue
                    if len(arr_ta_list)!=63:   continue
                    if len(arr_ctx_list)!=11:  continue
                    
                    if sum(arr_5m_list[0]) == 0:  continue

                    # parse all y_i
                    y_vec = []
                    for i in range(1, NUM_FUTURE_STEPS+1):
                        col_name = f"y_{i}"
                        val_str = row[col_name]
                        val_f   = float(val_str)  # each in [-1,1]
                        y_vec.append(val_f)

                    all_5m.append(arr_5m_list)
                    all_15m.append(arr_15m_list)
                    all_1h.append(arr_1h_list)
                    all_gt.append(arr_gt_list)
                    all_sa.append(arr_sa_list)
                    all_ta.append(arr_ta_list)
                    all_ctx.append(arr_ctx_list)
                    all_Y.append(y_vec)
                    all_ts.append(timestamp_str)
                    
                    a += 1

                except Exception as e:
                    self.logger.warning(f"Skipping row parse error: {e}")
                    continue

        X_5m  = np.array(all_5m, dtype=np.float32)          # (N,241,9)
        X_15m = np.array(all_15m, dtype=np.float32)         # (N,241,9)
        X_1h  = np.array(all_1h, dtype=np.float32)          # (N,241,9)
        X_gt  = np.array(all_gt, dtype=np.float32)          # (N,24,1)
        X_sa  = np.array(all_sa, dtype=np.float32)          # (N,12)
        X_ta  = np.array(all_ta, dtype=np.float32)          # (N,63)
        X_ctx = np.array(all_ctx, dtype=np.float32)         # (N,11)
        Y     = np.array(all_Y, dtype=np.float32)           # (N, NUM_FUTURE_STEPS)

        self.logger.info(
            f"Loaded {len(all_5m)} rows. X_5m={X_5m.shape}, Y={Y.shape}"
        )
        if len(all_5m) < 1:
            return [None]*9

        return X_5m, X_15m, X_1h, X_gt, X_sa, X_ta, X_ctx, Y, all_ts

    def split_data(self, N):
        """
        Based on train_ratio and val_ratio, compute indices
        """
        train_end = int(N * self.train_ratio)
        val_end   = int(N * (self.train_ratio + self.val_ratio))
        return train_end, val_end

    def train_lstm(self):
        """
        Train the multi-output LSTM (with 10 outputs) on the loaded data.
        """
        data = self.load_training_data()
        if not data or data[0] is None:
            self.logger.error("No valid training data found => abort LSTM training.")
            return

        (X_5m, X_15m, X_1h,
         X_gt, X_sa,
         X_ta, X_ctx,
         Y, all_ts) = data

        N = len(Y)
        if N < 10:
            self.logger.error("Too few samples => abort LSTM training.")
            return
        
        idx = np.arange(N)
        # np.random.shuffle(idx)    # intentionally NOT shuffling (time-based)

        X_5m  = X_5m[idx]
        X_15m = X_15m[idx]
        X_1h  = X_1h[idx]
        X_gt  = X_gt[idx]
        X_sa  = X_sa[idx]
        X_ta  = X_ta[idx]
        X_ctx = X_ctx[idx]
        Y     = Y[idx]
        all_ts = np.array(all_ts)[idx]

        # Split
        train_end, val_end = self.split_data(N)
        X_5m_train, X_5m_val   = X_5m[:train_end], X_5m[train_end:val_end]
        X_15m_train,X_15m_val  = X_15m[:train_end],X_15m[train_end:val_end]
        X_1h_train, X_1h_val   = X_1h[:train_end], X_1h[train_end:val_end]
        X_gt_train, X_gt_val   = X_gt[:train_end], X_gt[train_end:val_end]
        X_sa_train, X_sa_val   = X_sa[:train_end], X_sa[train_end:val_end]
        X_ta_train, X_ta_val   = X_ta[:train_end], X_ta[train_end:val_end]
        X_ctx_train,X_ctx_val  = X_ctx[:train_end],X_ctx[train_end:val_end]
        Y_train,     Y_val     = Y[:train_end],    Y[train_end:val_end]
        ts_train = all_ts[:train_end]

        self.logger.info(
            f"Train={len(Y_train)}, Val={len(Y_val)}"
        )

        # DEBUG: row_train_start, row_train_end. Remove all_ts after confirming order.
        if len(ts_train) > 0:
            row_train_start = ts_train[0]
            row_train_end   = ts_train[-1]
            self.logger.info(f"Train slice => start={row_train_start}, end={row_train_end}")

        # load/fallback scalers
        try:
            model_scaler = ModelScaler.load("models/scalers.pkl")
            self.logger.info("Loaded scalers from models/scalers.pkl.")
        except FileNotFoundError:
            self.logger.warning("No scalers found => creating new ModelScaler.")
            model_scaler = ModelScaler()

        if self.apply_scaling:
            model_scaler.fit_all(
                X_5m_train, X_15m_train, X_1h_train,
                X_gt_train, X_sa_train,
                X_ta_train, X_ctx_train
            )
        else:
            self.logger.info("Scaling disabled => pass-thru transforms.")

        # Transform
        (X_5m_train, X_15m_train, X_1h_train,
         X_gt_train, X_sa_train,
         X_ta_train, X_ctx_train) = prepare_for_model_inputs(
            X_5m_train, X_15m_train, X_1h_train,
            X_gt_train, X_sa_train, X_ta_train, X_ctx_train,
            model_scaler
        )
        (X_5m_val, X_15m_val, X_1h_val,
         X_gt_val, X_sa_val,
         X_ta_val, X_ctx_val) = prepare_for_model_inputs(
            X_5m_val, X_15m_val, X_1h_val,
            X_gt_val, X_sa_val, X_ta_val, X_ctx_val,
            model_scaler
        )

        # Train
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.early_stop_patience, 
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=2/3,         # Reduce LR by 33%, new LR is 2/3 of the previous LR
            patience=10,
            min_lr=1e-7,
            verbose=1
        )

        self.logger.info(f"Fitting LSTM => output dim={NUM_FUTURE_STEPS}, epochs={self.epochs}, batch={self.batch_size}")
        self.logger.info(f"Chronological split => train={len(Y_train)}, val={len(Y_val)}")

        self.model.fit(
            x=[
                X_5m_train, X_15m_train, X_1h_train,
                X_gt_train, X_sa_train, X_ta_train, X_ctx_train
            ],
            y=Y_train,  # shape(N,10)
            validation_data=(
                [
                    X_5m_val, X_15m_val, X_1h_val,
                    X_gt_val, X_sa_val, X_ta_val, X_ctx_val
                ],
                Y_val
            ) if len(Y_val) > 0 else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
            shuffle=False
        )

        # Save
        try:
            tf.keras.models.save_model(self.model, self.model_out)
            self.logger.info(f"LSTM model saved => {self.model_out}")
        except Exception as e:
            self.logger.error(f"Error saving model => {e}")

    def train_rl_offline(
        self,
        rl_csv: str,
        rl_out: str = "models/rl_DQNAgent.weights.h5",
        rl_epochs: int = 5,
        rl_batches: int = 500,
        state_dim: int = 10   # <--- changed default to 10 because we feed the 10 LSTM outputs
    ):
        """
        Offline training for a DQN from a CSV of transitions.

        Columns: old_state,action,reward,new_state,done
        - old_state,new_state: JSON array of length=10 (the 10 LSTM outputs).
        - action in {LONG, SHORT, HOLD}
        - reward is float (the single-step %change).
        - done is 0 or 1
        """
        if not os.path.exists(rl_csv):
            self.logger.error(f"No RL CSV found => {rl_csv}")
            return

        transitions = []
        failed_rows = 0
        with open(rl_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    old_state_str = row["old_state"]
                    action_str    = row["action"]
                    reward_val    = float(row["reward"])
                    new_state_str = row["new_state"]
                    done_val      = row["done"].strip()

                    old_st = np.array(json.loads(old_state_str), dtype=np.float32)
                    new_st = np.array(json.loads(new_state_str), dtype=np.float32)
                    done_flag = (done_val in ["1","true","True"])

                    if old_st.shape[0] != state_dim: 
                        continue
                    if new_st.shape[0] != state_dim:
                        continue
                    if action_str not in ACTIONS:
                        continue

                    transitions.append((old_st, action_str, reward_val, new_st, done_flag))
                except Exception:
                    failed_rows += 1
                    continue

        self.logger.warning(f"RL parse => skipped {failed_rows} invalid rows.")
        if len(transitions) < 10:
            self.logger.error("Too few RL transitions => abort RL training.")
            return

        self.logger.info(f"Loaded {len(transitions)} RL transitions from {rl_csv}.")

        # Build a DQNAgent
        dqn = DQNAgent(
            state_dim=state_dim,
            gamma=0.99,
            lr=0.001,
            batch_size=32,
            max_memory=len(transitions)+1,
            epsilon_start=0.0,  # offline => no exploration
            epsilon_min=0.0,
            epsilon_decay=1.0,
            update_target_steps=50
        )

        # Store transitions
        for (s, a, r, s2, d) in transitions:
            dqn.store_transition(s, a, r, s2, d)

        # Offline training
        for ep in range(rl_epochs):
            self.logger.info(f"[RL] Epoch {ep+1}/{rl_epochs}")
            for b in range(rl_batches):
                dqn.train_step()

        # Save DQN
        try:
            dqn.save()
            self.logger.info(f"RL weights saved => {rl_out}")
        except Exception as e:
            self.logger.error(f"Error saving RL => {e}")

    def run(self, rl_csv=None, rl_out=None, rl_epochs=5, rl_batches=500):
        """
        Main entry point:
         1) LSTM multi-output training if skip_lstm=False
         2) RL offline training if rl_csv is provided
        """
        if not self.skip_lstm:
            self.train_lstm()
        else:
            self.logger.info("Skipping LSTM training.")

        if rl_csv:
            self.train_rl_offline(
                rl_csv=rl_csv,
                rl_out=rl_out,
                rl_epochs=rl_epochs,
                rl_batches=rl_batches
            )
        else:
            self.logger.info("No RL CSV => skipping offline RL training.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-output LSTM + offline DQN from CSV."
    )
    parser.add_argument("--csv", type=str, default="training_data/training_data.csv",
                        help="Path to multi-output training_data.csv.")
    parser.add_argument("--model_out", type=str, default="models/advanced_lstm_model.keras",
                        help="File path for the LSTM model.")
    parser.add_argument("--epochs", type=int, default=1000, help="LSTM epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="early_stop_patience")
    parser.add_argument("--batch_size", type=int, default=16, help="LSTM batch size.")
    parser.add_argument("--no_scale", action="store_true",
                        help="Disable feature scaling.")
    parser.add_argument("--skip_lstm", action="store_true",
                        help="Skip LSTM training entirely.")
    parser.add_argument("--max_rows", type=int, default=0, 
                        help="Load x rows from csv file. 0 is all.")

    # RL
    parser.add_argument("--rl_csv", type=str, default=RL_TRANSITIONS_FILE,
                        help="Path to RL transitions CSV.")
    parser.add_argument("--rl_out", type=str, default="models/rl_DQNAgent.weights.h5",
                        help="Output file for RL weights.")
    parser.add_argument("--rl_epochs", type=int, default=5,
                        help="Offline RL training epochs.")
    parser.add_argument("--rl_batches", type=int, default=500,
                        help="Offline RL mini-batch updates per epoch.")
    parser.add_argument("--rl_state_dim", type=int, default=10,
                        help="Dimension of RL state (should match LSTM output=10).")

    return parser.parse_args()


def main():
    os.system('cls' if os.name=='nt' else 'clear')
    args = parse_args()

    trainer = Trainer(
        training_csv=args.csv,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        apply_scaling=not args.no_scale,
        skip_lstm=args.skip_lstm,
        max_rows=args.max_rows
    )
    trainer.run(
        rl_csv=args.rl_csv,
        rl_out=args.rl_out,
        rl_epochs=args.rl_epochs,
        rl_batches=args.rl_batches
    )


if __name__ == "__main__":
    main()




# Before   learning_rate=0.0005:     loss: 0.2082 - val_loss: 0.2415
# Before   learning_rate=0.001:      loss: 0.2082 - val_loss: 0.2415


# python fitter.py --early_stop_patience 50 --batch_size 32
