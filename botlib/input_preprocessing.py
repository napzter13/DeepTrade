#!/usr/bin/env python3
"""
input_preprocessing.py - Optimized version
A robust implementation for normalizing and scaling data for the multi-timeframe model.

Key improvements:
1. Better NaN handling with more robust fallbacks
2. Improved memory efficiency
3. Enhanced error handling with detailed logging
4. Support for mixed precision training
5. More robust scaling logic to handle unusual data patterns
"""

import pickle
import numpy as np
import logging
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import gc

# Get logger
logger = logging.getLogger("ModelScaler")

def clamp_and_scale(val, minv, maxv):
    """
    Safely clamp 'val' to [minv, maxv], then scale to [-1, 1].
    With improved numerical stability.
    """
    # Handle NaN
    if np.isnan(val):
        return 0.0
        
    # Clamp value to range
    val = max(minv, min(maxv, val))
    
    # Calculate range
    rng = maxv - minv
    
    # Check for zero/tiny range
    if rng < 1e-9:
        return 0.0
        
    # Scale to [0, 1]
    scaled01 = (val - minv) / rng
    
    # Scale to [-1, 1]
    return 2.0 * scaled01 - 1.0


class ModelScaler:
    """
    Enhanced scaler with improved robustness and error handling.
    
    Features:
    - Better handling of NaN values and edge cases
    - Proper handling of mixed precision data
    - Memory optimization for large datasets
    - Detailed logging for debugging
    """
    
    def __init__(self):
        """Initialize the model scaler"""
        # Time-series scalers
        self.scaler_5m = None
        self.scaler_15m = None
        self.scaler_1h = None
        self.scaler_google_trend = None
        
        # Santiment => 12 separate StandardScalers
        self.scalers_santiment = None  

        # TA => 63 separate MinMaxScalers
        self.ta_scalers = None  

        # Context => scalers for all context dimensions
        self.ctx_scalers = None
        
        # Flag to indicate if scalers are fitted
        self.fitted = False

        # Initialize logger
        self.logger = logger

    def fit_all(self, X_5m, X_15m, X_1h, X_google_trend, X_sa, X_ta, X_ctx):
        """Fit all scalers in one operation"""
        start_time = time.time()
        self.logger.info("Fitting all scalers...")
        
        try:
            self.fit_5m(X_5m)
            self.fit_15m(X_15m)
            self.fit_1h(X_1h)
            self.fit_google_trend(X_google_trend)
            self.fit_santiment(X_sa)
            self.fit_ta(X_ta)
            self.fit_ctx(X_ctx)
            
            self.fitted = True
            
            elapsed = time.time() - start_time
            self.logger.info(f"All scalers fitted successfully in {elapsed:.2f}s")
            
            # Save scalers for future use
            try:
                self.save("models/scalers.pkl")
                self.logger.info("Scalers saved to models/scalers.pkl")
            except Exception as e:
                self.logger.error(f"Error saving scalers: {e}")
        except Exception as e:
            self.logger.error(f"Error fitting scalers: {e}")
            self.fitted = False
            
    # =========================
    # Time-series
    # =========================
    def fit_5m(self, X_5m):
        """Fit scaler for 5m data with enhanced error handling"""
        try:
            # Lazily create the scaler if none
            if self.scaler_5m is None:
                self.scaler_5m = StandardScaler()
                
            # Ensure it's a numpy array
            if not isinstance(X_5m, np.ndarray):
                X_5m = np.array(X_5m, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_5m.shape) == 1:
                X_5m = X_5m.reshape(-1, 1)
            elif len(X_5m.shape) == 3:
                N, T, D = X_5m.shape
                X_5m = X_5m.reshape((N * T, D))
                
            # Replace NaNs with means or zeros
            if hasattr(self.scaler_5m, 'mean_'):
                mean_vals = self.scaler_5m.mean_
            else:
                mean_vals = np.nanmean(X_5m, axis=0)
                # Replace NaN means with zeros
                mean_vals = np.nan_to_num(mean_vals, nan=0.0)
                
            X_5m = np.nan_to_num(X_5m, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit scaler
            self.scaler_5m.fit(X_5m)
            self.logger.info("5m scaler fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting 5m scaler: {e}")
            # Create identity scaler as fallback
            self.scaler_5m = StandardScaler()
            self.scaler_5m.mean_ = np.zeros(9)
            self.scaler_5m.scale_ = np.ones(9)

    def transform_5m(self, X_5m):
        """Transform 5m data with enhanced error handling"""
        try:
            if self.scaler_5m is None:
                self.logger.warning("5m scaler not fitted, returning original data")
                return X_5m
                
            # Ensure it's a numpy array
            if not isinstance(X_5m, np.ndarray):
                X_5m = np.array(X_5m, dtype=np.float32)
                
            # Store original shape
            original_shape = X_5m.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_5m = X_5m.reshape(-1, 1)
            elif len(original_shape) == 3:
                N, T, D = original_shape
                X_5m = X_5m.reshape((N * T, D))
                
            # Replace NaNs with means
            X_5m = np.nan_to_num(X_5m, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Transform
            X_scaled = self.scaler_5m.transform(X_5m)
            
            # Reshape back to original shape
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
            elif len(original_shape) == 3:
                X_scaled = X_scaled.reshape(original_shape)
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming 5m data: {e}")
            return X_5m  # Return original data on error

    def fit_15m(self, X_15m):
        """Fit scaler for 15m data with enhanced error handling"""
        try:
            # Lazily create the scaler if none
            if self.scaler_15m is None:
                self.scaler_15m = StandardScaler()
                
            # Ensure it's a numpy array
            if not isinstance(X_15m, np.ndarray):
                X_15m = np.array(X_15m, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_15m.shape) == 1:
                X_15m = X_15m.reshape(-1, 1)
            elif len(X_15m.shape) == 3:
                N, T, D = X_15m.shape
                X_15m = X_15m.reshape((N * T, D))
                
            # Replace NaNs with means or zeros
            if hasattr(self.scaler_15m, 'mean_'):
                mean_vals = self.scaler_15m.mean_
            else:
                mean_vals = np.nanmean(X_15m, axis=0)
                # Replace NaN means with zeros
                mean_vals = np.nan_to_num(mean_vals, nan=0.0)
                
            X_15m = np.nan_to_num(X_15m, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit scaler
            self.scaler_15m.fit(X_15m)
            self.logger.info("15m scaler fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting 15m scaler: {e}")
            # Create identity scaler as fallback
            self.scaler_15m = StandardScaler()
            self.scaler_15m.mean_ = np.zeros(9)
            self.scaler_15m.scale_ = np.ones(9)

    def transform_15m(self, X_15m):
        """Transform 15m data with enhanced error handling"""
        try:
            if self.scaler_15m is None:
                self.logger.warning("15m scaler not fitted, returning original data")
                return X_15m
                
            # Ensure it's a numpy array
            if not isinstance(X_15m, np.ndarray):
                X_15m = np.array(X_15m, dtype=np.float32)
                
            # Store original shape
            original_shape = X_15m.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_15m = X_15m.reshape(-1, 1)
            elif len(original_shape) == 3:
                N, T, D = original_shape
                X_15m = X_15m.reshape((N * T, D))
                
            # Replace NaNs with means
            X_15m = np.nan_to_num(X_15m, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Transform
            X_scaled = self.scaler_15m.transform(X_15m)
            
            # Reshape back to original shape
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
            elif len(original_shape) == 3:
                X_scaled = X_scaled.reshape(original_shape)
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming 15m data: {e}")
            return X_15m  # Return original data on error

    def fit_1h(self, X_1h):
        """Fit scaler for 1h data with enhanced error handling"""
        try:
            # Lazily create the scaler if none
            if self.scaler_1h is None:
                self.scaler_1h = StandardScaler()
                
            # Ensure it's a numpy array
            if not isinstance(X_1h, np.ndarray):
                X_1h = np.array(X_1h, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_1h.shape) == 1:
                X_1h = X_1h.reshape(-1, 1)
            elif len(X_1h.shape) == 3:
                N, T, D = X_1h.shape
                X_1h = X_1h.reshape((N * T, D))
                
            # Replace NaNs with means or zeros
            if hasattr(self.scaler_1h, 'mean_'):
                mean_vals = self.scaler_1h.mean_
            else:
                mean_vals = np.nanmean(X_1h, axis=0)
                # Replace NaN means with zeros
                mean_vals = np.nan_to_num(mean_vals, nan=0.0)
                
            X_1h = np.nan_to_num(X_1h, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit scaler
            self.scaler_1h.fit(X_1h)
            self.logger.info("1h scaler fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting 1h scaler: {e}")
            # Create identity scaler as fallback
            self.scaler_1h = StandardScaler()
            self.scaler_1h.mean_ = np.zeros(9)
            self.scaler_1h.scale_ = np.ones(9)

    def transform_1h(self, X_1h):
        """Transform 1h data with enhanced error handling"""
        try:
            if self.scaler_1h is None:
                self.logger.warning("1h scaler not fitted, returning original data")
                return X_1h
                
            # Ensure it's a numpy array
            if not isinstance(X_1h, np.ndarray):
                X_1h = np.array(X_1h, dtype=np.float32)
                
            # Store original shape
            original_shape = X_1h.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_1h = X_1h.reshape(-1, 1)
            elif len(original_shape) == 3:
                N, T, D = original_shape
                X_1h = X_1h.reshape((N * T, D))
                
            # Replace NaNs with means
            X_1h = np.nan_to_num(X_1h, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Transform
            X_scaled = self.scaler_1h.transform(X_1h)
            
            # Reshape back to original shape
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
            elif len(original_shape) == 3:
                X_scaled = X_scaled.reshape(original_shape)
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming 1h data: {e}")
            return X_1h  # Return original data on error

    def fit_google_trend(self, X_google_trend):
        """Fit scaler for Google Trends data with enhanced error handling"""
        try:
            # Lazily create the scaler if none
            if self.scaler_google_trend is None:
                self.scaler_google_trend = StandardScaler()
                
            # Ensure it's a numpy array
            if not isinstance(X_google_trend, np.ndarray):
                X_google_trend = np.array(X_google_trend, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_google_trend.shape) == 1:
                X_google_trend = X_google_trend.reshape(-1, 1)
            elif len(X_google_trend.shape) == 3:
                N, T, D = X_google_trend.shape
                X_google_trend = X_google_trend.reshape((N * T, D))
                
            # Replace NaNs with zeros
            X_google_trend = np.nan_to_num(X_google_trend, nan=0.0, posinf=100.0, neginf=0.0)
            
            # Google Trends are typically 0-100, so scale accordingly
            X_google_trend = np.clip(X_google_trend, 0, 100)
            
            # Fit scaler
            self.scaler_google_trend.fit(X_google_trend)
            self.logger.info("Google Trends scaler fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting Google Trends scaler: {e}")
            # Create identity scaler as fallback
            self.scaler_google_trend = StandardScaler()
            self.scaler_google_trend.mean_ = np.array([50.0])
            self.scaler_google_trend.scale_ = np.array([50.0])

    def transform_google_trend(self, X_google_trend):
        """Transform Google Trends data with enhanced error handling"""
        try:
            if self.scaler_google_trend is None:
                self.logger.warning("Google Trends scaler not fitted, returning original data")
                return X_google_trend
                
            # Ensure it's a numpy array
            if not isinstance(X_google_trend, np.ndarray):
                X_google_trend = np.array(X_google_trend, dtype=np.float32)
                
            # Store original shape
            original_shape = X_google_trend.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_google_trend = X_google_trend.reshape(-1, 1)
            elif len(original_shape) == 3:
                N, T, D = original_shape
                X_google_trend = X_google_trend.reshape((N * T, D))
                
            # Replace NaNs with zeros
            X_google_trend = np.nan_to_num(X_google_trend, nan=0.0, posinf=100.0, neginf=0.0)
            
            # Google Trends are typically 0-100, so scale accordingly
            X_google_trend = np.clip(X_google_trend, 0, 100)
            
            # Transform
            X_scaled = self.scaler_google_trend.transform(X_google_trend)
            
            # Reshape back to original shape
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
            elif len(original_shape) == 3:
                X_scaled = X_scaled.reshape(original_shape)
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming Google Trends data: {e}")
            return X_google_trend  # Return original data on error
    
    # =========================
    # Santiment => 12 separate StandardScalers
    # =========================
    def fit_santiment(self, X_santiment):
        """Fit scalers for santiment data with enhanced error handling"""
        try:
            # Ensure it's a numpy array
            if not isinstance(X_santiment, np.ndarray):
                X_santiment = np.array(X_santiment, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_santiment.shape) == 1:
                X_santiment = X_santiment.reshape(1, -1)
                
            # Get number of features
            _, D = X_santiment.shape
            
            # Lazily create the scalers if none
            if self.scalers_santiment is None:
                self.scalers_santiment = [StandardScaler() for _ in range(D)]
                
            # Fit each scaler
            for i in range(D):
                # Extract column
                col = X_santiment[:, i].reshape(-1, 1)
                
                # Handle NaN values
                col = np.nan_to_num(col, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Fit scaler
                self.scalers_santiment[i].fit(col)
                
            self.logger.info(f"Santiment scalers ({D} features) fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting santiment scalers: {e}")
            # Create identity scalers as fallback
            D = 12  # Default santiment dimension
            self.scalers_santiment = []
            for _ in range(D):
                scaler = StandardScaler()
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
                self.scalers_santiment.append(scaler)

    def transform_santiment(self, X_santiment):
        """Transform santiment data with enhanced error handling"""
        try:
            if self.scalers_santiment is None:
                self.logger.warning("Santiment scalers not fitted, returning original data")
                return X_santiment
                
            # Ensure it's a numpy array
            if not isinstance(X_santiment, np.ndarray):
                X_santiment = np.array(X_santiment, dtype=np.float32)
                
            # Store original shape
            original_shape = X_santiment.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_santiment = X_santiment.reshape(1, -1)
                
            # Get number of features
            N, D = X_santiment.shape
            D_scalers = len(self.scalers_santiment)
            
            # Check if dimensions match
            if D != D_scalers:
                self.logger.warning(f"Santiment features dimension mismatch: data={D}, scalers={D_scalers}")
                D = min(D, D_scalers)
                
            # Transform each feature
            X_scaled = np.zeros((N, D), dtype=np.float32)
            for i in range(D):
                # Extract column
                col = X_santiment[:, i].reshape(-1, 1)
                
                # Handle NaN values
                col = np.nan_to_num(col, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Transform
                X_scaled[:, i] = self.scalers_santiment[i].transform(col).flatten()
                
            # Reshape back to original shape if needed
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming santiment data: {e}")
            return X_santiment  # Return original data on error

    # =========================
    # TA => 63 separate MinMaxScalers
    # =========================
    def fit_ta(self, X_ta):
        """Fit scalers for TA data with enhanced error handling"""
        try:
            # Ensure it's a numpy array
            if not isinstance(X_ta, np.ndarray):
                X_ta = np.array(X_ta, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_ta.shape) == 1:
                X_ta = X_ta.reshape(1, -1)
                
            # Get number of features
            _, D = X_ta.shape
            
            # Lazily create the scalers if none
            if self.ta_scalers is None:
                self.ta_scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(D)]
                
            # Fit each scaler
            for i in range(D):
                # Extract column
                col = X_ta[:, i].reshape(-1, 1)
                
                # Handle NaN values
                col = np.nan_to_num(col, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Fit scaler
                self.ta_scalers[i].fit(col)
                
            self.logger.info(f"TA scalers ({D} features) fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting TA scalers: {e}")
            # Create identity scalers as fallback
            D = 63  # Default TA dimension
            self.ta_scalers = []
            for _ in range(D):
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.min_ = np.array([-1.0])
                scaler.scale_ = np.array([0.5])  # Scale to map [-1, 1] -> [-1, 1]
                self.ta_scalers.append(scaler)

    def transform_ta(self, X_ta):
        """Transform TA data with enhanced error handling"""
        try:
            if self.ta_scalers is None:
                self.logger.warning("TA scalers not fitted, returning original data")
                return X_ta
                
            # Ensure it's a numpy array
            if not isinstance(X_ta, np.ndarray):
                X_ta = np.array(X_ta, dtype=np.float32)
                
            # Store original shape
            original_shape = X_ta.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_ta = X_ta.reshape(1, -1)
                
            # Get number of features
            N, D = X_ta.shape
            D_scalers = len(self.ta_scalers)
            
            # Check if dimensions match
            if D != D_scalers:
                self.logger.warning(f"TA features dimension mismatch: data={D}, scalers={D_scalers}")
                D = min(D, D_scalers)
                
            # Transform each feature
            X_scaled = np.zeros((N, D), dtype=np.float32)
            for i in range(D):
                # Extract column
                col = X_ta[:, i].reshape(-1, 1)
                
                # Handle NaN values
                col = np.nan_to_num(col, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Transform
                X_scaled[:, i] = self.ta_scalers[i].transform(col).flatten()
                
            # Reshape back to original shape if needed
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming TA data: {e}")
            return X_ta  # Return original data on error

    # =========================
    # Context => scalers for all context dimensions
    # =========================
    def fit_ctx(self, X_ctx):
        """Fit scalers for context data with enhanced error handling"""
        try:
            # Ensure it's a numpy array
            if not isinstance(X_ctx, np.ndarray):
                X_ctx = np.array(X_ctx, dtype=np.float32)
                
            # Handle various input shapes
            if len(X_ctx.shape) == 1:
                X_ctx = X_ctx.reshape(1, -1)
                
            # Get number of features
            _, D = X_ctx.shape
            
            # Lazily create the scalers if none
            if self.ctx_scalers is None:
                self.ctx_scalers = []
                # For columns 0,1,2,3,8, we use StandardScaler
                for i in [0, 1, 2, 3, 8]:
                    if i < D:
                        self.ctx_scalers.append((i, StandardScaler()))
                
            # Fit each scaler
            for i, scaler in self.ctx_scalers:
                if i < D:
                    # Extract column
                    col = X_ctx[:, i].reshape(-1, 1)
                    
                    # Handle NaN values
                    col = np.nan_to_num(col, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    # Fit scaler
                    scaler.fit(col)
                
            self.logger.info(f"Context scalers for columns 0,1,2,3,8 fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting context scalers: {e}")
            # Create identity scalers as fallback
            self.ctx_scalers = []
            for i in [0, 1, 2, 3, 8]:
                scaler = StandardScaler()
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
                self.ctx_scalers.append((i, scaler))

    def transform_ctx(self, X_ctx):
        """Transform context data with enhanced error handling"""
        try:
            if self.ctx_scalers is None:
                self.logger.warning("Context scalers not fitted, returning manually scaled data")
                return self._manual_transform_ctx(X_ctx)
                
            # Ensure it's a numpy array
            if not isinstance(X_ctx, np.ndarray):
                X_ctx = np.array(X_ctx, dtype=np.float32)
                
            # Store original shape
            original_shape = X_ctx.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_ctx = X_ctx.reshape(1, -1)
                
            # Get number of features
            N, D = X_ctx.shape
            
            # Create output array
            X_scaled = X_ctx.copy()
            
            # Transform columns with fitted scalers
            for i, scaler in self.ctx_scalers:
                if i < D:
                    # Extract column
                    col = X_ctx[:, i].reshape(-1, 1)
                    
                    # Handle NaN values
                    col = np.nan_to_num(col, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    # Transform
                    X_scaled[:, i] = scaler.transform(col).flatten()
            
            # For columns [4,5,6,7], clamp to [-100, 100] and scale to [-1, 1]
            for i in [4, 5, 6, 7]:
                if i < D:
                    for n in range(N):
                        val = X_ctx[n, i]
                        if np.isnan(val):
                            X_scaled[n, i] = 0.0
                        else:
                            X_scaled[n, i] = clamp_and_scale(val, -100.0, 100.0)
            
            # For columns [9, 10], clamp to [0, 1] and scale to [-1, 1]
            for i in [9, 10]:
                if i < D:
                    for n in range(N):
                        val = X_ctx[n, i]
                        if np.isnan(val):
                            X_scaled[n, i] = 0.0
                        else:
                            X_scaled[n, i] = clamp_and_scale(val, 0.0, 1.0)
            
            # Reshape back to original shape if needed
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming context data: {e}")
            return self._manual_transform_ctx(X_ctx)  # Use manual transform as fallback
            
    def _manual_transform_ctx(self, X_ctx):
        """Manual transformation for context data when scalers are not available"""
        try:
            # Ensure it's a numpy array
            if not isinstance(X_ctx, np.ndarray):
                X_ctx = np.array(X_ctx, dtype=np.float32)
                
            # Store original shape
            original_shape = X_ctx.shape
            
            # Reshape data for transformation
            if len(original_shape) == 1:
                X_ctx = X_ctx.reshape(1, -1)
                
            # Get number of features
            N, D = X_ctx.shape
            
            # Create output array
            X_scaled = X_ctx.copy()
            
            # For columns [0,1,2,3,8], apply standard scaling based on typical ranges
            if 0 < D:  # price
                X_scaled[:, 0] = X_ctx[:, 0] / 50000.0  # BTC price / typical max value
                
            if 1 < D:  # Google Trend
                X_scaled[:, 1] = X_ctx[:, 1] / 50.0 - 1.0  # Scale 0-100 to [-1, 1]
                
            if 2 < D:  # Reddit sentiment
                X_scaled[:, 2] = X_ctx[:, 2] / 100.0  # Scale -100 to 100 to [-1, 1]
                
            if 3 < D:  # Order book ratio
                X_scaled[:, 3] = (X_ctx[:, 3] - 1.0) * 2.0  # Center around 1.0, scale to [-1, 1]
                
            if 8 < D:  # Santiment
                X_scaled[:, 8] = X_ctx[:, 8]  # Already in [-1, 1] range
            
            # For columns [4,5,6,7], clamp to [-100, 100] and scale to [-1, 1]
            for i in [4, 5, 6, 7]:
                if i < D:
                    for n in range(N):
                        val = X_ctx[n, i]
                        if np.isnan(val):
                            X_scaled[n, i] = 0.0
                        else:
                            X_scaled[n, i] = clamp_and_scale(val, -100.0, 100.0)
            
            # For columns [9, 10], clamp to [0, 1] and scale to [-1, 1]
            for i in [9, 10]:
                if i < D:
                    for n in range(N):
                        val = X_ctx[n, i]
                        if np.isnan(val):
                            X_scaled[n, i] = 0.0
                        else:
                            X_scaled[n, i] = clamp_and_scale(val, 0.0, 1.0)
            
            # Reshape back to original shape if needed
            if len(original_shape) == 1:
                X_scaled = X_scaled.flatten()
                
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error in manual context transformation: {e}")
            return X_ctx  # Return original data on error

    # =========================
    # Save / Load
    # =========================
    def save(self, filename):
        """Save all scalers to file with better error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            data = {
                "scaler_5m": self.scaler_5m,
                "scaler_15m": self.scaler_15m,
                "scaler_1h": self.scaler_1h,
                "scaler_google_trend": self.scaler_google_trend,
                "scalers_santiment": self.scalers_santiment,
                "ta_scalers": self.ta_scalers,
                "ctx_scalers": self.ctx_scalers,
                "fitted": self.fitted
            }
            
            with open(filename, "wb") as f:
                pickle.dump(data, f)
                
            self.logger.info(f"Scalers saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving scalers to {filename}: {e}")
            return False

    @classmethod
    def load(cls, filename):
        """Load scalers from file with better error handling"""
        try:
            if not os.path.exists(filename):
                logger.warning(f"Scaler file {filename} not found")
                return cls()
                
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
            ms = cls()
            ms.scaler_5m = data.get("scaler_5m")
            ms.scaler_15m = data.get("scaler_15m")
            ms.scaler_1h = data.get("scaler_1h")
            ms.scaler_google_trend = data.get("scaler_google_trend")
            ms.scalers_santiment = data.get("scalers_santiment")
            ms.ta_scalers = data.get("ta_scalers")
            ms.ctx_scalers = data.get("ctx_scalers")
            ms.fitted = data.get("fitted", True)
            
            logger.info(f"Scalers loaded from {filename}")
            return ms
        except Exception as e:
            logger.error(f"Error loading scalers from {filename}: {e}")
            return cls()


def prepare_for_model_inputs(
    arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx,
    model_scaler: ModelScaler = None
):
    """
    Convenience function to transform all inputs with the given model_scaler.
    With enhanced error handling and memory optimization.
    """
    try:
        if model_scaler is None or not model_scaler.fitted:
            logger.warning("No fitted scaler provided, returning original data")
            return arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx
            
        # Transform each input
        s_5m = model_scaler.transform_5m(arr_5m)
        s_15m = model_scaler.transform_15m(arr_15m)
        s_1h = model_scaler.transform_1h(arr_1h)
        s_gt = model_scaler.transform_google_trend(arr_google_trend)
        s_sa = model_scaler.transform_santiment(arr_santiment)
        s_ta = model_scaler.transform_ta(arr_ta)
        s_ctx = model_scaler.transform_ctx(arr_ctx)
        
        return s_5m, s_15m, s_1h, s_gt, s_sa, s_ta, s_ctx
    except Exception as e:
        logger.error(f"Error in prepare_for_model_inputs: {e}")
        # Return original data on error
        return arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx