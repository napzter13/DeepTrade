#!/usr/bin/env python3
"""
fitter.py

Extended version:
- Trains a multi-input LSTM model from training_data.csv (as originally).
- Optionally trains a DQN (RL) model offline from an rl_transitions CSV.

Usage examples:

1) **Train only the LSTM**:
   python fitter.py --csv training_data/training_data.csv --model_out models/advanced_lstm_model.keras --epochs 600

2) **Train LSTM + RL**:
   python fitter.py --csv training_data/training_data.csv --model_out models/advanced_lstm_model.keras \
                    --rl_csv training_data/rl_transitions.csv --rl_out models/rl_DQNAgent.weights.h5 --rl_epochs 5

3) **Train only RL** (skip LSTM by passing a non-existent CSV or removing it):
   python fitter.py --rl_csv training_data/rl_transitions.csv --rl_out models/rl_DQNAgent.weights.h5 --rl_epochs 5 --skip_lstm

The RL transitions CSV must have columns:
   old_state, action, reward, new_state, done
Where:
   old_state, new_state = JSON string of shape (state_dim,)
   action in {LONG, SHORT, HOLD}
   reward is float
   done is 0 or 1

"""

import os
import json
import csv
import argparse
import logging
import numpy as np
import tensorflow as tf
import pickle

# Import advanced LSTM model build
from botlib.models import load_advanced_lstm_model

# For normalizing input data
from botlib.input_preprocessing import ModelScaler, prepare_for_model_inputs

# Import DQNAgent for RL training (make sure we have it in botlib/rl.py)
from botlib.rl import DQNAgent, ACTIONS

RL_TRANSITIONS_FILE = os.path.join("training_data", "rl_transitions.csv")

class Trainer:
    def __init__(
        self,
        training_csv="training_data/training_data.csv",
        model_out="models/advanced_lstm_model.keras",
        window_5m=60,
        feature_5m=9,
        window_15m=60,
        feature_15m=9,
        window_1h=60,
        feature_1h=9,
        window_google_trend=8,
        feature_google_trend=1,
        santiment_dim=12,
        ta_dim=63,
        signal_dim=11,
        epochs=600,
        batch_size=4096,
        apply_scaling=True,
        train_ratio=0.7,
        val_ratio=0.2,
        skip_lstm=False
    ):
        """
        :param training_csv:     Path to training_data.csv from the backtester.
        :param model_out:        File path to save the LSTM model (H5 or SavedModel).
        :param window_5m:        LSTM window for 5m branch
        :param feature_5m:       Feature dimension for 5m
        :param window_15m:       LSTM window for 15m branch
        :param feature_15m:      Feature dimension for 15m
        :param window_1h:        LSTM window for 1h branch
        :param feature_1h:       Feature dimension for 1h
        :param window_google_trend:    LSTM window for google_trend branch
        :param feature_google_trend:   Feature dimension for google_trend
        :param santiment_dim:    Size of the santiment vector (12)
        :param ta_dim:           Size of the TA vector (63)
        :param signal_dim:       Size of the "signal" context vector (11)
        :param epochs:           Number of training epochs for the LSTM
        :param batch_size:       Training batch size for the LSTM
        :param apply_scaling:    Whether to apply scaling with ModelScaler
        :param train_ratio:      Fraction of data for training
        :param val_ratio:        Fraction of data for validation
        :param skip_lstm:        If True, skip LSTM training entirely.
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
        self.feature_15m=feature_15m
        self.window_1h = window_1h
        self.feature_1h= feature_1h
        self.window_google_trend = window_google_trend
        self.feature_google_trend= feature_google_trend
        self.santiment_dim = santiment_dim
        self.ta_dim = ta_dim
        self.signal_dim = signal_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.apply_scaling= apply_scaling
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.skip_lstm   = skip_lstm

        if not skip_lstm:
            self.logger.info("Initializing multi-timeframe LSTM model structure.")
            self.model = load_advanced_lstm_model(
                model_5m_window=self.window_5m,
                model_15m_window=self.window_15m,
                model_1h_window=self.window_1h,
                feature_dim=self.feature_5m,
                santiment_dim=self.santiment_dim,
                ta_dim=self.ta_dim,
                signal_dim=self.signal_dim
            )
        else:
            self.logger.info("Skipping LSTM model initialization (skip_lstm=True).")
            self.model = None

    def load_training_data(self):
        """
        Read training_data.csv with columns:
          timestamp, arr_5m, arr_15m, arr_1h, arr_google_trend,
          arr_santiment, arr_ta_63, arr_ctx_11, y
        Each arr_* is a JSON string representing the array. y in [-1,1].
        We'll parse them into Python lists, then build big NumPy arrays.
        """
        if not os.path.exists(self.training_csv):
            self.logger.error(f"No training CSV at {self.training_csv}")
            return None, None, None, None, None, None, None, None

        all_5m   = []
        all_15m  = []
        all_1h   = []
        all_google_trend   = []
        all_santiment   = []
        all_ta   = []
        all_ctx  = []
        all_y    = []

        with open(self.training_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    arr_5m_str  = row["arr_5m"]
                    arr_15m_str = row["arr_15m"]
                    arr_1h_str  = row["arr_1h"]
                    arr_google_trend_str  = row["arr_google_trend"]
                    arr_santiment_str  = row["arr_santiment"]
                    arr_ta_str  = row["arr_ta_63"]
                    arr_ctx_str = row["arr_ctx_11"]
                    y_str       = row["y"]

                    arr_5m_list   = json.loads(arr_5m_str)[0]
                    arr_15m_list  = json.loads(arr_15m_str)[0]
                    arr_1h_list   = json.loads(arr_1h_str)[0]
                    arr_google_trend_list   = json.loads(arr_google_trend_str)[0]
                    arr_santiment_list   = json.loads(arr_santiment_str)[0]
                    arr_ta_63list = json.loads(arr_ta_str)[0]
                    arr_ctx_11list= json.loads(arr_ctx_str)[0]
                    y_val         = float(y_str)

                    # Check shapes
                    if len(arr_5m_list)!=60:      continue
                    if len(arr_15m_list)!=60:     continue
                    if len(arr_1h_list)!=60:      continue
                    if len(arr_google_trend_list)!=8:      continue
                    if len(arr_santiment_list)!=12:    continue
                    if len(arr_ta_63list)!=63:    continue
                    if len(arr_ctx_11list)!=11:   continue

                    all_5m.append(arr_5m_list)
                    all_15m.append(arr_15m_list)
                    all_1h.append(arr_1h_list)
                    all_google_trend.append(arr_google_trend_list)
                    all_santiment.append(arr_santiment_list)
                    all_ta.append(arr_ta_63list)
                    all_ctx.append(arr_ctx_11list)
                    all_y.append(y_val)

                except Exception as e:
                    self.logger.warning(f"Skipping row parse error: {e}")
                    continue

        X_5m   = np.array(all_5m,   dtype=np.float32)   # shape=(N,60,9)
        X_15m  = np.array(all_15m,  dtype=np.float32)   # shape=(N,60,9)
        X_1h   = np.array(all_1h,   dtype=np.float32)   # shape=(N,60,9)
        X_google_trend = np.array(all_google_trend, dtype=np.float32)  # shape=(N,8,1)
        X_santiment    = np.array(all_santiment,    dtype=np.float32)  # shape=(N,12)
        X_ta           = np.array(all_ta,           dtype=np.float32)  # shape=(N,63)
        X_ctx          = np.array(all_ctx,          dtype=np.float32)  # shape=(N,11)
        Y              = np.array(all_y,            dtype=np.float32)  # shape=(N,)

        self.logger.info(
            f"Loaded data => {len(all_5m)} samples. "
            f"X_5m={X_5m.shape}, X_ta={X_ta.shape}, Y={Y.shape}"
        )
        if len(all_5m) < 1:
            return None, None, None, None, None, None, None, None

        return X_5m, X_15m, X_1h, X_google_trend, X_santiment, X_ta, X_ctx, Y

    def split_data(self, N):
        """
        Based on train_ratio and val_ratio, compute indices
        """
        train_end = int(N * self.train_ratio)
        val_end   = int(N * (self.train_ratio + self.val_ratio))
        return train_end, val_end

    def train_lstm(self):
        """
        Train the multi-input LSTM model on training_data.csv
        """
        # 1) load data
        data = self.load_training_data()
        if data[0] is None:
            self.logger.error("No training data found => abort LSTM training.")
            return

        (X_5m, X_15m, X_1h, X_google_trend,
         X_santiment, X_ta, X_ctx, Y) = data

        N = len(Y)
        if N < 10:
            self.logger.error("Too few samples => abort LSTM training.")
            return

        # 2) shuffle data
        idx = np.arange(N)
        np.random.shuffle(idx)
        X_5m  = X_5m[idx]
        X_15m = X_15m[idx]
        X_1h  = X_1h[idx]
        X_google_trend = X_google_trend[idx]
        X_santiment    = X_santiment[idx]
        X_ta  = X_ta[idx]
        X_ctx = X_ctx[idx]
        Y     = Y[idx]

        # 3) train/val/test split
        train_end, val_end = self.split_data(N)
        X_5m_train  = X_5m[:train_end]
        X_5m_val    = X_5m[train_end:val_end]
        X_5m_test   = X_5m[val_end:]

        X_15m_train = X_15m[:train_end]
        X_15m_val   = X_15m[train_end:val_end]
        X_15m_test  = X_15m[val_end:]

        X_1h_train  = X_1h[:train_end]
        X_1h_val    = X_1h[train_end:val_end]
        X_1h_test   = X_1h[val_end:]

        X_google_trend_train = X_google_trend[:train_end]
        X_google_trend_val   = X_google_trend[train_end:val_end]
        X_google_trend_test  = X_google_trend[val_end:]

        X_santiment_train = X_santiment[:train_end]
        X_santiment_val   = X_santiment[train_end:val_end]
        X_santiment_test  = X_santiment[val_end:]

        X_ta_train  = X_ta[:train_end]
        X_ta_val    = X_ta[train_end:val_end]
        X_ta_test   = X_ta[val_end:]

        X_ctx_train = X_ctx[:train_end]
        X_ctx_val   = X_ctx[train_end:val_end]
        X_ctx_test  = X_ctx[val_end:]

        Y_train = Y[:train_end]
        Y_val   = Y[train_end:val_end]
        Y_test  = Y[val_end:]
        
        self.logger.info(
            f"Data splits => train={len(Y_train)}, val={len(Y_val)}, test={len(Y_test)}"
        )

        # 4) load/fallback scalers
        try:
            model_scaler = ModelScaler.load("models/scalers.pkl")
            self.logger.info("Loaded model scalers from models/scalers.pkl.")
        except FileNotFoundError:
            self.logger.warning("No scalers found at models/scalers.pkl. Creating new...")
            model_scaler = ModelScaler()  # pass-thru

        if self.apply_scaling:
            model_scaler.fit_all(
                X_5m_train, X_15m_train, X_1h_train,
                X_google_trend_train, X_santiment_train,
                X_ta_train, X_ctx_train
            )
        else:
            self.logger.info("No scaling => pass-thru transforms.")
            
        # 5) transform & optionally fit scaling
        (X_5m_train, X_15m_train, X_1h_train,
         X_google_trend_train, X_santiment_train,
         X_ta_train, X_ctx_train) = prepare_for_model_inputs(
            X_5m_train, X_15m_train, X_1h_train,
            X_google_trend_train, X_santiment_train,
            X_ta_train, X_ctx_train,
            model_scaler
        )
        (X_5m_val, X_15m_val, X_1h_val,
         X_google_trend_val, X_santiment_val,
         X_ta_val, X_ctx_val) = prepare_for_model_inputs(
            X_5m_val, X_15m_val, X_1h_val,
            X_google_trend_val, X_santiment_val,
            X_ta_val, X_ctx_val,
            model_scaler
        )
        (X_5m_test, X_15m_test, X_1h_test,
         X_google_trend_test, X_santiment_test,
         X_ta_test, X_ctx_test) = prepare_for_model_inputs(
            X_5m_test, X_15m_test, X_1h_test,
            X_google_trend_test, X_santiment_test,
            X_ta_test, X_ctx_test,
            model_scaler
        )

        # 6) Train LSTM
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        self.logger.info(f"Start LSTM training epochs={self.epochs}, batch_size={self.batch_size}")

        # print("==== Debug Train Data ====")
        # print("X_5m_train[0] =\n", X_5m_train[0])  # shape (60,9)
        # print("X_15m_train[0] =\n", X_15m_train[0])  # etc.

        # print("Some label samples (Y_train[:10]) =", Y_train[:10])
        # print("Y stats => min:", np.min(Y_train), "max:", np.max(Y_train), 
        #     "mean:", np.mean(Y_train), "std:", np.std(Y_train))
        
        # print("==== Debug Val Data ====")
        # print("X_5m_val[0] =", X_5m_val[0])
        # print("Y_val[:10]  =", Y_val[:10])
        # print("Y_val stats => min:", np.min(Y_val), ...)

        self.model.fit(
            x=[
                X_5m_train, X_15m_train, X_1h_train,
                X_google_trend_train, X_santiment_train,
                X_ta_train, X_ctx_train
            ],
            y=Y_train,
            validation_data=(
                [
                    X_5m_val, X_15m_val, X_1h_val,
                    X_google_trend_val, X_santiment_val,
                    X_ta_val, X_ctx_val
                ],
                Y_val
            ) if len(Y_val)>0 else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # 7) Evaluate on test set
        if len(Y_test)>0:
            loss = self.model.evaluate(
                [
                    X_5m_test, X_15m_test, X_1h_test,
                    X_google_trend_test, X_santiment_test,
                    X_ta_test, X_ctx_test
                ],
                Y_test,
                verbose=0
            )
            self.logger.info(f"Test Loss => {loss:.6f}")
        else:
            self.logger.info("No test set => skipping final evaluation.")

        # 8) Save LSTM model
        try:
            tf.keras.models.save_model(self.model, self.model_out)
            self.logger.info(f"LSTM model saved to => {self.model_out}")
        except Exception as e:
            self.logger.error(f"Error saving LSTM model => {e}")

    def train_rl_offline(
        self,
        rl_csv: str,
        rl_out: str = "models/rl_DQNAgent.weights.h5",
        rl_epochs: int = 5,
        rl_batches: int = 500,
        state_dim: int = 4
    ):
        """
        Offline training for a DQN from a CSV of transitions.

        :param rl_csv:   Path to a CSV with columns: old_state,action,reward,new_state,done
        :param rl_out:   Output path for the DQN weights
        :param rl_epochs: Number of epochs (full passes) over the transitions
        :param rl_batches: # of mini-batch updates per epoch
        :param state_dim: dimension of RL state (must match environment)
        """
        if not os.path.exists(rl_csv):
            self.logger.error(f"No RL CSV found at {rl_csv}")
            return

        # Load transitions
        transitions = []
        failed_rows = 0
        with open(rl_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    old_state_str = row["old_state"]
                    action = row["action"]        # LONG, SHORT, HOLD
                    reward = float(row["reward"])
                    new_state_str = row["new_state"]
                    done_val = row["done"]

                    old_state_arr = np.array(json.loads(old_state_str), dtype=np.float32)
                    new_state_arr = np.array(json.loads(new_state_str), dtype=np.float32)
                    done_flag = (done_val.strip() in ["1", "True", "true"])

                    # validate shapes
                    if old_state_arr.shape[0] != state_dim:
                        continue
                    if new_state_arr.shape[0] != state_dim:
                        continue
                    if action not in ACTIONS:
                        continue

                    transitions.append((old_state_arr, action, reward, new_state_arr, done_flag))
                except Exception as e:
                    failed_rows += 1
                    # self.logger.warning(f"Skipping invalid RL row => {e}")
                    continue
                
        self.logger.warning(f"Skipping invalid RL rows => " + str(failed_rows))

        if len(transitions) < 10:
            self.logger.error("Too few RL transitions => abort RL training.")
            return

        self.logger.info(f"Loaded {len(transitions)} RL transitions from {rl_csv}.")

        # Build a DQNAgent
        # Adjust hyperparams to liking
        dqn = DQNAgent(
            state_dim=state_dim,
            gamma=0.99,
            lr=0.001,
            batch_size=4096,
            max_memory=len(transitions)+1,  # to hold all transitions
            epsilon_start=0.0,   # no exploration needed for offline
            epsilon_min=0.0,
            epsilon_decay=1.0,   # no decay if offline
            update_target_steps=50
        )

        # Put transitions into memory
        for (s, a, r, s2, d) in transitions:
            dqn.store_transition(s, a, r, s2, d)

        # We do 'rl_epochs' full passes. Each epoch => 'rl_batches' random mini-batch updates
        for ep in range(rl_epochs):
            self.logger.info(f"RL epoch {ep+1}/{rl_epochs}")
            for b in range(rl_batches):
                dqn.train_step()
            # optionally shuffle transitions or do other logic

        # Save the DQN
        try:
            dqn.save()
            self.logger.info(f"DQN weights saved to => {rl_out}")
        except Exception as e:
            self.logger.error(f"Error saving RL weights => {e}")

    def run(self, rl_csv=None, rl_out=None, rl_epochs=5, rl_batches=500, skip_lstm=False):
        """
        Main entry point:
          1) Train LSTM if not skip_lstm.
          2) Train RL offline if rl_csv is provided.
        """
        if not skip_lstm:
            # Train the LSTM aggregator
            self.train_lstm()
        else:
            self.logger.info("Skipping LSTM training (skip_lstm=True).")

        if rl_csv:
            # Offline DQN training
            self.train_rl_offline(
                rl_csv=rl_csv,
                rl_out=rl_out if rl_out else "models/rl_DQNAgent.weights.h5",
                rl_epochs=rl_epochs,
                rl_batches=rl_batches
            )
        else:
            self.logger.info("No RL CSV provided => skipping offline RL training.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-input timeframe LSTM from training_data.csv + optional offline RL."
    )
    parser.add_argument("--csv", type=str, default="training_data/training_data.csv",
                        help="Path to training_data.csv for LSTM.")
    parser.add_argument("--model_out", type=str, default="models/advanced_lstm_model.keras",
                        help="File to save the LSTM model.")
    parser.add_argument("--epochs", type=int, default=600,
                        help="Epochs for LSTM.")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for LSTM.")
    parser.add_argument("--no_scale", action="store_true",
                        help="Disable scaling for time-series branches.")
    parser.add_argument("--skip_lstm", action="store_true",
                        help="Skip LSTM training entirely.")

    # RL training arguments
    parser.add_argument("--rl_csv", type=str, default=RL_TRANSITIONS_FILE,
                        help="Path to RL transitions CSV. If provided, offline RL training is run.")
    parser.add_argument("--rl_out", type=str, default="models/rl_DQNAgent.weights.h5",
                        help="File to save the DQN weights.")
    parser.add_argument("--rl_epochs", type=int, default=5,
                        help="Number of RL training epochs offline.")
    parser.add_argument("--rl_batches", type=int, default=500,
                        help="Number of mini-batch updates per RL epoch.")
    parser.add_argument("--rl_state_dim", type=int, default=4,
                        help="Dimension of the RL state vectors in transitions CSV.")

    return parser.parse_args()


def main():
    os.system('cls')            # clear terminal on Windows (no-op on other OS)
    args = parse_args()

    trainer = Trainer(
        training_csv=args.csv,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        apply_scaling=not args.no_scale,
        skip_lstm=args.skip_lstm
    )
    trainer.run(
        rl_csv=args.rl_csv,
        rl_out=args.rl_out,
        rl_epochs=args.rl_epochs,
        rl_batches=args.rl_batches,
        skip_lstm=args.skip_lstm
    )


if __name__=="__main__":
    main()


