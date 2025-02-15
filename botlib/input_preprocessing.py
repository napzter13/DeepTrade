#!/usr/bin/env python3
"""
input_preprocessing.py
A single place for normalizing/scaling data for a 5-input multi-timeframe model.

We define a ModelScaler class that handles:
  1) StandardScaler for each of the 3 time-series branches: 5m, 15m, 1h
  2) 63-dim TA features => each column has its own MinMaxScaler(feature_range=(-1,1))
  3) 11-dim context => 
     - columns [0,1,2,3,8] => each has its own StandardScaler
     - columns [4,5,6,7] => clamp [-100,100] => scale => [-1,1]
     - columns [9,10] => clamp [0,1] => scale => [-1,1]

Usage (in trainer):
-------------------
1) from input_preprocessing import ModelScaler, prepare_for_model_inputs
2) model_scaler = ModelScaler()
3) model_scaler.fit_5m(X_5m_train)
   model_scaler.fit_15m(X_15m_train)
   model_scaler.fit_1h(X_1h_train)
   model_scaler.fit_ta(X_ta_train)    # => builds 63 MinMaxScalers
   model_scaler.fit_ctx(X_ctx_train)  # => standard scalers for ctx cols 0,1,2,3,8
4) Then transform:
   X_5m_train, X_15m_train, X_1h_train, X_ta_train, X_ctx_train = prepare_for_model_inputs(
       X_5m_train, X_15m_train, X_1h_train, X_ta_train, X_ctx_train, model_scaler
   )
5) Save:
   model_scaler.save("models/scalers.pkl")

Usage (in tradebot/backtester):
------------------------------
1) model_scaler = ModelScaler.load("models/scalers.pkl")
2) s_5m, s_15m, s_1h, s_ta, s_ctx = prepare_for_model_inputs(arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx, model_scaler)
3) model.predict([...])

This ensures consistent scaling of all inputs in both training and inference.
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .environment import (
    get_logger
)

def clamp_and_scale(val, minv, maxv):
    """
    Simple helper to clamp 'val' to [minv, maxv], then scale to [-1,1].
    If (maxv - minv) < 1e-9 => return 0.0 to avoid numerical issues.
    """
    if val < minv:
        val = minv
    if val > maxv:
        val = maxv
    rng = maxv - minv
    if rng < 1e-9:
        return 0.0
    scaled01 = (val - minv) / rng
    return scaled01 * 2.0 - 1.0


class ModelScaler:
    """
    Scalers for:
    1) Time-series:
       - self.scaler_5m
       - self.scaler_15m
       - self.scaler_1h
       - self.scaler_google_trend
       Each is a StandardScaler fit on shape (N*60,9).
    2) TA features (63-dim):
       - self.ta_scalers => list of 63 MinMaxScalers(feature_range=(-1,1)),
         one per column. So each TA can have a separate min & max.
    3) Santiment features (12-dim):
       - self.sa_scalers => list of 12 StandardScalers,
    4) Context (11-dim):
       - columns 0,1,2,3,8 => each its own StandardScaler
       - columns 4..7 => clamp [-100,100], scale => [-1,1]
       - columns 9..10 => clamp [0,1],   scale => [-1,1]
    """

    def __init__(self):
        # Time-series scalers
        self.scaler_5m  = None
        self.scaler_15m = None
        self.scaler_1h  = None
        self.scaler_google_trend  = None
        
        # Santiment => 12 separate StandardScalers
        self.scalers_santiment  = None

        # TA => 63 separate MinMaxScalers
        self.ta_scalers = None  # list of length=63, each is MinMaxScaler

        # Context => standard scalers for columns [0,1,2,3,8]
        self.ctx_scaler_0 = None
        self.ctx_scaler_1 = None
        self.ctx_scaler_2 = None
        self.ctx_scaler_3 = None
        self.ctx_scaler_8 = None

        self.logger = get_logger("ModelScaler")


    def fit_all(self, X_5m, X_15m, X_1h, X_google_trend, X_sa, X_ta, X_ctx):
        self.fit_5m(X_5m)
        self.fit_15m(X_15m)
        self.fit_1h(X_1h)
        self.fit_google_trend(X_google_trend)
        self.fit_santiment(X_sa)
        self.fit_ta(X_ta)
        self.fit_ctx(X_ctx)
        
        try:
            self.save("models/scalers.pkl")
            self.logger.info(f"models/scalers.pkl saved")
        except Exception as e:
            self.logger.error(f"Error saving models/scalers.pkl => {e}")
            
        self.logger.info("Fitted StandardScalers.")
        
    # =========================
    # Time-series
    # =========================
    def fit_5m(self, X_5m):
        # Lazily create the scaler if none
        if self.scaler_5m is None:
            self.scaler_5m = StandardScaler()
        # Ensure it's an array
        if not hasattr(X_5m, "shape"):
            X_5m = np.array(X_5m, dtype=float)
        if len(X_5m.shape) == 1:
            # reshape (N,) => (N,60,1) if needed
            X_5m = X_5m.reshape(-1, 60, 1)
            
        N, T, D = X_5m.shape
        flatten = X_5m.reshape((N * T, D))
        
        # If this scaler is already fit, we can use its .mean_, else compute from data.
        if hasattr(self.scaler_5m, 'mean_'):
            mean_val = self.scaler_5m.mean_
        else:
            mean_val = np.nanmean(flatten, axis=0)  # shape (D,)

        # Replace any NaNs with the mean
        flatten = np.where(np.isnan(flatten), mean_val, flatten)

        # Finally fit
        self.scaler_5m.fit(flatten)

    def transform_5m(self, X_5m):
        if not hasattr(X_5m, "shape"):
            X_5m = np.array(X_5m, dtype=float)
        if len(X_5m.shape) == 1:
            X_5m = X_5m.reshape(-1, 60, 1)

        N, T, D = X_5m.shape
        flatten = X_5m.reshape((N * T, D))

        # Impute NaNs with the scaler's existing mean_
        flatten = np.where(np.isnan(flatten), self.scaler_5m.mean_, flatten)

        out = self.scaler_5m.transform(flatten)
        return out.reshape((N, T, D))


    def fit_15m(self, X_15m):
        if self.scaler_15m is None:
            self.scaler_15m = StandardScaler()
        if not hasattr(X_15m, "shape"):
            X_15m = np.array(X_15m, dtype=float)
        if len(X_15m.shape) == 1:
            X_15m = X_15m.reshape(-1, 60, 1)
            
        N, T, D = X_15m.shape
        flatten = X_15m.reshape((N * T, D))

        if hasattr(self.scaler_15m, 'mean_'):
            mean_val = self.scaler_15m.mean_
        else:
            mean_val = np.nanmean(flatten, axis=0)
        flatten = np.where(np.isnan(flatten), mean_val, flatten)

        self.scaler_15m.fit(flatten)

    def transform_15m(self, X_15m):
        if not hasattr(X_15m, "shape"):
            X_15m = np.array(X_15m, dtype=float)
        if len(X_15m.shape) == 1:
            X_15m = X_15m.reshape(-1, 60, 1)

        N, T, D = X_15m.shape
        flatten = X_15m.reshape((N * T, D))

        flatten = np.where(np.isnan(flatten), self.scaler_15m.mean_, flatten)
        out = self.scaler_15m.transform(flatten)
        return out.reshape((N, T, D))


    def fit_1h(self, X_1h):
        if self.scaler_1h is None:
            self.scaler_1h = StandardScaler()
        if not hasattr(X_1h, "shape"):
            X_1h = np.array(X_1h, dtype=float)
        if len(X_1h.shape) == 1:
            X_1h = X_1h.reshape(-1, 60, 1)
            
        N, T, D = X_1h.shape
        flatten = X_1h.reshape((N * T, D))

        if hasattr(self.scaler_1h, 'mean_'):
            mean_val = self.scaler_1h.mean_
        else:
            mean_val = np.nanmean(flatten, axis=0)
        flatten = np.where(np.isnan(flatten), mean_val, flatten)

        self.scaler_1h.fit(flatten)

    def transform_1h(self, X_1h):
        if not hasattr(X_1h, "shape"):
            X_1h = np.array(X_1h, dtype=float)
        if len(X_1h.shape) == 1:
            X_1h = X_1h.reshape(-1, 60, 1)

        N, T, D = X_1h.shape
        flatten = X_1h.reshape((N * T, D))

        flatten = np.where(np.isnan(flatten), self.scaler_1h.mean_, flatten)
        out = self.scaler_1h.transform(flatten)
        return out.reshape((N, T, D))


    def fit_google_trend(self, X_google_trend):
        if self.scaler_google_trend is None:
            self.scaler_google_trend = StandardScaler()
        if not hasattr(X_google_trend, "shape"):
            X_google_trend = np.array(X_google_trend, dtype=float)
        if len(X_google_trend.shape) == 1:
            X_google_trend = X_google_trend.reshape(-1, 8, 1)
                
        N, T, D = X_google_trend.shape
        flatten = X_google_trend.reshape((N*T, D))
        # Use the scaler's mean if it exists; otherwise, compute it manually ignoring NaNs.
        if hasattr(self.scaler_google_trend, 'mean_'):
            mean_value = self.scaler_google_trend.mean_
        else:
            mean_value = np.nanmean(flatten, axis=0)  # shape: (D,)
        flatten = np.where(np.isnan(flatten), mean_value, flatten)
        self.scaler_google_trend.fit(flatten)
        
    def transform_google_trend(self, X_google_trend):
        N,T,D = X_google_trend.shape
        flatten = X_google_trend.reshape((N*T, D))
        flatten = np.where(np.isnan(flatten), self.scaler_google_trend.mean_, flatten)
        out = self.scaler_google_trend.transform(flatten)
        return out.reshape((N,T,D))
    
    # =========================
    # Santiment => 12 separate StandardScalers
    # =========================
    def fit_santiment(self, X_santiment):
        # Convert to array if needed
        if not hasattr(X_santiment, "shape"):
            X_santiment = np.array(X_santiment, dtype=float)
        if len(X_santiment.shape) == 1:
            X_santiment = X_santiment.reshape(-1, 1)
        
        N, D = X_santiment.shape
        
        # Lazily initialize if needed
        if self.scalers_santiment is None:
            self.scalers_santiment = [StandardScaler() for _ in range(D)]
            
        # Fit each scaler to the corresponding column
        for i in range(D):
            col_i = X_santiment[:, i:i+1]  # shape (N,1)
            # If the scaler has already been fit, use its mean;
            # otherwise compute the mean manually ignoring NaNs.
            if hasattr(self.scalers_santiment[i], 'mean_'):
                mean_val = self.scalers_santiment[i].mean_
            else:
                mean_val = np.nanmean(col_i, axis=0)  # shape (1,)
            col_imputed = np.where(np.isnan(col_i), mean_val, col_i)
            self.scalers_santiment[i].fit(col_imputed)


    def transform_santiment(self, X_santiment):       
        # Ensure 2D
        if not hasattr(X_santiment, "ndim"):
            X_santiment = np.array(X_santiment, dtype=float)
        if X_santiment.ndim < 2:
            X_santiment = X_santiment.reshape((1, -1))

        N, D = X_santiment.shape
        X_transformed = np.empty_like(X_santiment, dtype=float)

        # Transform each column individually
        for i in range(D):
            # Extract the column (shape: [N,])
            col = X_santiment[:, i]
            
            # Replace any np.nan values with the mean computed by the scaler.
            # Each StandardScaler instance has a `mean_` attribute (a scalar for 1D data).
            col_imputed = np.where(np.isnan(col), self.scalers_santiment[i].mean_, col)
            
            # StandardScaler expects a 2D array, so reshape the column to (N, 1)
            col_imputed_reshaped = col_imputed.reshape(-1, 1)
            
            # Transform the imputed column and flatten back to 1D.
            X_transformed[:, i] = self.scalers_santiment[i].transform(col_imputed_reshaped).flatten()
        
        return X_transformed

    # =========================
    # TA => 63 separate MinMaxScalers
    # =========================
    def fit_ta(self, X_ta):
        N, D = X_ta.shape
        if self.ta_scalers is None:
            self.ta_scalers = [MinMaxScaler(feature_range=(-1,1)) for _ in range(D)]
        if not hasattr(X_ta, "shape"):
            X_ta = np.array(X_ta, dtype=float)
        if len(X_ta.shape) == 1:
            X_ta = X_ta.reshape(-1, 1)
            
        for i in range(D):
            col_i = X_ta[:, i:i+1]  # shape (N,1)
            self.ta_scalers[i].fit(col_i)

    def transform_ta(self, arr_ta):
        N, D = arr_ta.shape
        if arr_ta.ndim < 2:
            arr_ta = arr_ta.reshape((1, -1))

        out = arr_ta.copy()
        for i in range(D):
            col_i = out[:, i:i+1]  # shape (N,1)
            out[:, i:i+1] = self.ta_scalers[i].transform(col_i)
        return out

    # =========================
    # Context => 11 dims
    #   - columns [0,1,2,3] => standard scaler
    #   - columns [4,5,6,7,8] => clamp [-100,100] => scale => [-1,1]
    #   - columns [9,10] => clamp [0,1] => scale => [-1,1]
    # =========================
    def fit_ctx(self, X_ctx):
        if not hasattr(X_ctx, "shape"):
            X_ctx = np.array(X_ctx, dtype=float)
            
        # Lazily create the context scalers if they are not already created
        if self.ctx_scaler_0 is None:
            self.ctx_scaler_0 = StandardScaler()
        if self.ctx_scaler_1 is None:
            self.ctx_scaler_1 = StandardScaler()
        if self.ctx_scaler_2 is None:
            self.ctx_scaler_2 = StandardScaler()
        if self.ctx_scaler_3 is None:
            self.ctx_scaler_3 = StandardScaler()
        if self.ctx_scaler_8 is None:
            self.ctx_scaler_8 = StandardScaler()
            
        # Helper function to compute a safe mean from a column:
        def safe_mean(col):
            m = np.nanmean(col, axis=0)
            # If m is NaN (e.g. if col was all NaN), return 0.0
            if np.any(np.isnan(m)):
                m = np.array([0.0])
            return m

        # For column 0:
        col0 = X_ctx[:, 0:1]
        mean_val0 = safe_mean(col0)
        col0_imputed = np.where(np.isnan(col0), mean_val0, col0)
        self.ctx_scaler_0.fit(col0_imputed)

        # For column 1:
        col1 = X_ctx[:, 1:2]
        mean_val1 = safe_mean(col1)
        col1_imputed = np.where(np.isnan(col1), mean_val1, col1)
        self.ctx_scaler_1.fit(col1_imputed)

        # For column 2:
        col2 = X_ctx[:, 2:3]
        mean_val2 = safe_mean(col2)
        col2_imputed = np.where(np.isnan(col2), mean_val2, col2)
        self.ctx_scaler_2.fit(col2_imputed)

        # For column 3:
        col3 = X_ctx[:, 3:4]
        mean_val3 = safe_mean(col3)
        col3_imputed = np.where(np.isnan(col3), mean_val3, col3)
        self.ctx_scaler_3.fit(col3_imputed)

        # For column 8:
        col8 = X_ctx[:, 8:9]
        mean_val8 = safe_mean(col8)
        col8_imputed = np.where(np.isnan(col8), mean_val8, col8)
        self.ctx_scaler_8.fit(col8_imputed)

    def transform_ctx(self, arr_ctx):
        """
        Expects arr_ctx of shape (N, 11). 
        For columns 0-3 and 8, use the corresponding StandardScaler (after replacing any NaNs with the scaler’s mean).
        For columns 4-7, clamp each value to [-100,100] and scale to [-1,1] (if NaN, output 0.0).
        For columns 9-10, clamp each value to [0,1] and scale to [-1,1] (if NaN, output 0.0).
        """
        if arr_ctx.ndim < 2:
            arr_ctx = arr_ctx.reshape((1, -1))
        out = arr_ctx.copy()
        N = out.shape[0]

        # For columns 0-3, use the corresponding StandardScaler.
        col_indices_std = [0, 1, 2, 3, 8]
        scaler_dict = {
            0: self.ctx_scaler_0,
            1: self.ctx_scaler_1,
            2: self.ctx_scaler_2,
            3: self.ctx_scaler_3,
            8: self.ctx_scaler_8,
        }
        for col in col_indices_std:
            col_data = out[:, col]
            # Replace any NaNs with the scaler’s mean (or 0 if mean is NaN)
            mean_val = scaler_dict[col].mean_
            if np.any(np.isnan(mean_val)):
                mean_val = np.array([0.0])
            col_data = np.where(np.isnan(col_data), mean_val, col_data)
            # Transform the column (reshape as 2D for StandardScaler)
            out[:, col] = scaler_dict[col].transform(col_data.reshape(-1, 1)).flatten()

        # For columns 4-7: clamp to [-100,100] then scale to [-1,1].
        for col in [4, 5, 6, 7]:
            for n in range(N):
                val = out[n, col]
                if np.isnan(val):
                    out[n, col] = 0.0
                else:
                    out[n, col] = clamp_and_scale(val, -100, 100)

        # For columns 9 and 10: clamp to [0,1] then scale to [-1,1].
        for col in [9, 10]:
            for n in range(N):
                val = out[n, col]
                if np.isnan(val):
                    out[n, col] = 0.0
                else:
                    out[n, col] = clamp_and_scale(val, 0, 1)

        return out



    # =========================
    # Save / Load
    # =========================
    def save(self, filename):
        data = {
            "scaler_5m":   self.scaler_5m,
            "scaler_15m":  self.scaler_15m,
            "scaler_1h":   self.scaler_1h,
            "scaler_google_trend":   self.scaler_google_trend,
            "scalers_santiment":   self.scalers_santiment,
            "ta_scalers":  self.ta_scalers,  # list of 63 MinMaxScalers
            "ctx_scaler_0": self.ctx_scaler_0,
            "ctx_scaler_1": self.ctx_scaler_1,
            "ctx_scaler_2": self.ctx_scaler_2,
            "ctx_scaler_3": self.ctx_scaler_3,
            "ctx_scaler_8": self.ctx_scaler_8
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        ms = cls()
        ms.scaler_5m   = data.get("scaler_5m")
        ms.scaler_15m  = data.get("scaler_15m")
        ms.scaler_1h   = data.get("scaler_1h")
        ms.scaler_google_trend   = data.get("scaler_google_trend")
        ms.scalers_santiment   = data.get("scalers_santiment")
        ms.ta_scalers  = data.get("ta_scalers")
        ms.ctx_scaler_0= data.get("ctx_scaler_0")
        ms.ctx_scaler_1= data.get("ctx_scaler_1")
        ms.ctx_scaler_2= data.get("ctx_scaler_2")
        ms.ctx_scaler_3= data.get("ctx_scaler_3")
        ms.ctx_scaler_8= data.get("ctx_scaler_8")
        return ms


def prepare_for_model_inputs(
    arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx,
    model_scaler: ModelScaler
):
    """
    Convenience function to transform all five inputs with the given model_scaler.
    """
    s_5m  = model_scaler.transform_5m(arr_5m)
    s_15m = model_scaler.transform_15m(arr_15m)
    s_1h  = model_scaler.transform_1h(arr_1h)
    s_gt  = model_scaler.transform_google_trend(arr_google_trend)
    s_sa  = model_scaler.transform_santiment(arr_santiment)
    s_ta  = model_scaler.transform_ta(arr_ta)
    s_ctx = model_scaler.transform_ctx(arr_ctx)
    return s_5m, s_15m, s_1h, s_gt, s_sa, s_ta, s_ctx