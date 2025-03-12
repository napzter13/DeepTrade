#!/usr/bin/env python3
"""
fitter.py - Ultra-optimized version for memory-constrained environments

Usage:
python fitter.py --model_size small --batch_size 2 --grad_accum --accum_steps 16
"""

import os
import json
import csv
import argparse
import logging
import numpy as np
import tensorflow as tf
import math
import time
import sys
import gc
from pathlib import Path
from datetime import datetime



from botlib.environment import NUM_FUTURE_STEPS
from botlib.models import load_advanced_lstm_model
from botlib.input_preprocessing import ModelScaler, prepare_for_model_inputs
from botlib.rl import DQNAgent, ACTIONS
from botlib.models import safe_mse_loss, TimeSeriesEncoder, TabularEncoder, LightEnsembleModel

# Constants
RL_TRANSITIONS_FILE = os.path.join("training_data", "rl_transitions.csv")



# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Force TensorFlow to be memory efficient
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Fix 1: Disable XLA JIT compilation which may cause the libdevice error
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# Define NUM_FUTURE_STEPS here to avoid circular dependencies
NUM_FUTURE_STEPS = 5

# Configure GPU setup first before any other imports
def setup_gpu():
    """Configure GPUs for optimal performance"""
    # Check if there are GPUs available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found. Running on CPU.")
        return False
    
    try:
        # Don't attempt to configure memory growth or limits,
        # just use whatever configuration is already in place
        
        # Enable tensor cores if available
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("TensorFloat-32 execution enabled if supported by hardware")
        except:
            pass
        
        print(f"GPU setup successful. Found {len(gpus)} GPU(s)")
        return True
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
        return False

# Configure GPUs at module import time
setup_gpu()

# Force garbage collection
gc.enable()




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
        epochs=100,
        early_stop_patience=20,
        batch_size=2,  # Reduced default batch size
        apply_scaling=True,
        train_ratio=0.8,
        val_ratio=0.2,
        skip_lstm=False,
        max_rows=0,
        grad_accum=True,
        accum_steps=16,  # Increased accumulation steps
        model_size="small",  # Default to small for memory efficiency
        reduce_precision=True  # Use FP16 to save memory
    ):
        """
        Ultra-optimized Trainer for memory-constrained environments
        """
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
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
        self.grad_accum   = grad_accum
        self.accum_steps   = accum_steps
        self.model_size = model_size
        self.reduce_precision = reduce_precision
        
        # Set base units based on model size (much smaller values)
        base_units_map = {
            "tiny": 24,
            "small": 32,
            "medium": 48,
            "large": 64,
            "xlarge": 96
        }
        self.base_units = base_units_map.get(model_size, 32)

        os.makedirs("models", exist_ok=True)
        os.makedirs("logs_training", exist_ok=True)
        
        if not skip_lstm:
            self.logger.info(f"Initializing memory-efficient model with {NUM_FUTURE_STEPS} outputs")
            self.logger.info(f"Model size: {model_size.upper()} ({self.base_units} units)")
            
            try:
                # Try to load an existing model
                if os.path.exists(self.model_out):
                    self.logger.info(f"Loading existing model from {self.model_out}")
                    
                    # Try to load with a simpler approach that handles errors better
                    try:
                        self.model = tf.keras.models.load_model(self.model_out)
                        self.logger.info(f"Successfully loaded model from {self.model_out}")
                    except Exception as e:
                        self.logger.warning(f"Error loading model: {e}")
                        self.logger.warning("Creating new model instead.")
                        self.model = None
                else:
                    self.logger.info("No existing model found, creating new one.")
                    self.model = None
                    
                # Create a new model if loading failed
                if self.model is None:
                    self.logger.info("Creating new model...")
                    self.model = load_advanced_lstm_model(
                        model_5m_window=self.window_5m,
                        model_15m_window=self.window_15m,
                        model_1h_window=self.window_1h,
                        feature_dim=self.feature_5m,
                        santiment_dim=self.santiment_dim,
                        ta_dim=self.ta_dim,
                        signal_dim=self.signal_dim,
                        base_units=self.base_units,
                        memory_efficient=True,
                        gradient_accumulation=self.grad_accum,
                        gradient_accumulation_steps=self.accum_steps,
                        mixed_precision=self.reduce_precision
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize model: {e}")
                raise
        else:
            self.logger.info("Skipping model initialization (skip_lstm=True).")
            self.model = None

    def load_training_data(self):
        """
        Loads training data from CSV with memory optimizations
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
        
        start_time = time.time()
        
        # Count rows first to preallocate memory
        total_rows = 0
        with open(self.training_csv, "r", encoding="utf-8") as f:
            for _ in csv.reader(f):
                total_rows += 1
        total_rows -= 1  # Account for header
        
        # Limit to max_rows if specified
        if self.max_rows > 0:
            total_rows = min(total_rows, self.max_rows)
            
        # Now read the actual data
        row_count = 0
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
            
            for row in reader:
                if self.max_rows > 0 and row_count >= self.max_rows:
                    break

                try:
                    # Progress reporting
                    if row_count % 1000 == 0:
                        self.logger.info(f"Loading data: {row_count}/{total_rows} rows processed")
                        # Force garbage collection
                        gc.collect()
                        
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
                    
                    row_count += 1

                except Exception as e:
                    self.logger.warning(f"Skipping row parse error: {e}")
                    continue

        # Convert to numpy arrays using float16 for half precision if enabled
        dtype = np.float16 if self.reduce_precision else np.float32
        
        X_5m  = np.array(all_5m, dtype=dtype)          # (N,241,9)
        X_15m = np.array(all_15m, dtype=dtype)         # (N,241,9)
        X_1h  = np.array(all_1h, dtype=dtype)          # (N,241,9)
        X_gt  = np.array(all_gt, dtype=dtype)          # (N,24,1)
        X_sa  = np.array(all_sa, dtype=dtype)          # (N,12)
        X_ta  = np.array(all_ta, dtype=dtype)          # (N,63)
        X_ctx = np.array(all_ctx, dtype=dtype)         # (N,11)
        Y     = np.array(all_Y, dtype=dtype)           # (N, NUM_FUTURE_STEPS)

        # Clean up to free memory
        del all_5m, all_15m, all_1h, all_gt, all_sa, all_ta, all_ctx, all_Y
        gc.collect()

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Loaded {len(X_5m)} rows in {elapsed_time:.2f}s. X_5m={X_5m.shape}, Y={Y.shape}"
        )
        if len(X_5m) < 1:
            return [None]*9

        return X_5m, X_15m, X_1h, X_gt, X_sa, X_ta, X_ctx, Y, all_ts

    def split_data(self, N):
        """
        Split data into train/validation sets
        """
        train_end = int(N * self.train_ratio)
        val_end   = int(N * (self.train_ratio + self.val_ratio))
        return train_end, val_end

    def custom_gradient_accumulation_training(self, x_train, y_train, x_val=None, y_val=None):
        """
        Memory-efficient custom training function with gradient accumulation
        """
        best_val_loss = float('inf')
        patience_counter = 0
        optimizer = self.model.optimizer
        
        # Get loss function
        if isinstance(self.model.loss, str):
            loss_fn = tf.keras.losses.MeanSquaredError()
        else:
            loss_fn = self.model.loss
        
        # Training metrics
        train_loss = tf.keras.metrics.Mean()
        train_mae = tf.keras.metrics.MeanAbsoluteError()
        val_loss = tf.keras.metrics.Mean()
        val_mae = tf.keras.metrics.MeanAbsoluteError()
        
        # TensorBoard logging
        current_time = int(time.time())
        log_dir = f'logs_training/gradient_accum_{current_time}'
        summary_writer = tf.summary.create_file_writer(log_dir)
        
        # Extract input components
        X_5m_train, X_15m_train, X_1h_train, X_gt_train, X_sa_train, X_ta_train, X_ctx_train = x_train
        
        # Number of batches per epoch
        train_samples = len(y_train)
        steps_per_epoch = max(1, train_samples // self.batch_size)
        
        # Input validation
        if len(y_train) != len(X_5m_train):
            self.logger.error(f"Inconsistent data sizes: y_train={len(y_train)}, X_5m_train={len(X_5m_train)}")
            return
            
        # Prepare validation data if present
        has_validation = (x_val is not None and y_val is not None and len(y_val) > 0)
        if has_validation:
            X_5m_val, X_15m_val, X_1h_val, X_gt_val, X_sa_val, X_ta_val, X_ctx_val = x_val
            val_samples = len(y_val)
            val_steps = max(1, val_samples // self.batch_size)
        
        self.logger.info(f"Starting training with gradient accumulation")
        self.logger.info(f"Samples: {train_samples}, Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"Batch size: {self.batch_size}, Accumulation steps: {self.accum_steps}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            
            # Reset metrics - FIXED: use reset() instead of reset_states()
            train_loss.reset()
            train_mae.reset()
            
            # Manual batching and gradient accumulation
            accumulated_gradients = None
            accumulation_count = 0
            
            # Process in batches
            for step in range(steps_per_epoch):
                # Calculate batch indices
                start_idx = step * self.batch_size
                end_idx = min(start_idx + self.batch_size, train_samples)
                
                # Skip empty batches
                if start_idx >= end_idx:
                    continue
                    
                # Prepare batch data - always use float32 for stability
                X_5m_batch = tf.convert_to_tensor(X_5m_train[start_idx:end_idx], dtype=tf.float32)
                X_15m_batch = tf.convert_to_tensor(X_15m_train[start_idx:end_idx], dtype=tf.float32)
                X_1h_batch = tf.convert_to_tensor(X_1h_train[start_idx:end_idx], dtype=tf.float32)
                X_gt_batch = tf.convert_to_tensor(X_gt_train[start_idx:end_idx], dtype=tf.float32)
                X_sa_batch = tf.convert_to_tensor(X_sa_train[start_idx:end_idx], dtype=tf.float32)
                X_ta_batch = tf.convert_to_tensor(X_ta_train[start_idx:end_idx], dtype=tf.float32)
                X_ctx_batch = tf.convert_to_tensor(X_ctx_train[start_idx:end_idx], dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_train[start_idx:end_idx], dtype=tf.float32)
                
                # Initialize accumulated gradients if needed
                if accumulated_gradients is None:
                    accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
                
                # Forward pass and calculate gradients
                with tf.GradientTape() as tape:
                    y_pred = self.model(
                        [X_5m_batch, X_15m_batch, X_1h_batch, X_gt_batch, X_sa_batch, X_ta_batch, X_ctx_batch], 
                        training=True
                    )
                    batch_loss = loss_fn(y_batch, y_pred)
                
                # Calculate and accumulate gradients
                gradients = tape.gradient(batch_loss, self.model.trainable_variables)
                for i, grad in enumerate(gradients):
                    if grad is not None:
                        accumulated_gradients[i] += grad
                
                # Update metrics
                train_loss.update_state(batch_loss)
                train_mae.update_state(y_batch, y_pred)
                
                # Free memory
                del X_5m_batch, X_15m_batch, X_1h_batch, X_gt_batch, X_sa_batch, X_ta_batch, X_ctx_batch
                del y_batch, y_pred, gradients
                
                # Increment accumulation counter
                accumulation_count += 1
                
                # Apply accumulated gradients when needed
                is_last_batch = (step == steps_per_epoch - 1)
                if accumulation_count >= self.accum_steps or is_last_batch:
                    # Normalize gradients by the number of accumulation steps
                    for i in range(len(accumulated_gradients)):
                        if accumulated_gradients[i] is not None and accumulation_count > 0:
                            accumulated_gradients[i] = accumulated_gradients[i] / accumulation_count
                    
                    # Apply gradients
                    optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
                    
                    # Reset accumulation
                    accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
                    accumulation_count = 0
                    
                    # Force garbage collection
                    gc.collect()
                
                # Log progress
                if (step + 1) % 10 == 0:
                    self.logger.info(f"  Step {step+1}/{steps_per_epoch} - Loss: {train_loss.result():.4f}")
            
            # Validation if available
            if has_validation:
                # Reset validation metrics - FIXED: use reset() instead of reset_states()
                val_loss.reset()
                val_mae.reset()
                
                # Process validation in batches
                for val_step in range(val_steps):
                    start_idx = val_step * self.batch_size
                    end_idx = min(start_idx + self.batch_size, val_samples)
                    
                    # Skip empty batches
                    if start_idx >= end_idx:
                        continue
                        
                    # Prepare validation batch
                    X_5m_val_batch = tf.convert_to_tensor(X_5m_val[start_idx:end_idx], dtype=tf.float32)
                    X_15m_val_batch = tf.convert_to_tensor(X_15m_val[start_idx:end_idx], dtype=tf.float32)
                    X_1h_val_batch = tf.convert_to_tensor(X_1h_val[start_idx:end_idx], dtype=tf.float32)
                    X_gt_val_batch = tf.convert_to_tensor(X_gt_val[start_idx:end_idx], dtype=tf.float32)
                    X_sa_val_batch = tf.convert_to_tensor(X_sa_val[start_idx:end_idx], dtype=tf.float32)
                    X_ta_val_batch = tf.convert_to_tensor(X_ta_val[start_idx:end_idx], dtype=tf.float32)
                    X_ctx_val_batch = tf.convert_to_tensor(X_ctx_val[start_idx:end_idx], dtype=tf.float32)
                    y_val_batch = tf.convert_to_tensor(y_val[start_idx:end_idx], dtype=tf.float32)
                    
                    # Forward pass
                    y_pred = self.model(
                        [X_5m_val_batch, X_15m_val_batch, X_1h_val_batch, X_gt_val_batch, 
                        X_sa_val_batch, X_ta_val_batch, X_ctx_val_batch], 
                        training=False
                    )
                    
                    # Calculate loss and update metrics
                    val_batch_loss = loss_fn(y_val_batch, y_pred)
                    val_loss.update_state(val_batch_loss)
                    val_mae.update_state(y_val_batch, y_pred)
                    
                    # Free memory
                    del X_5m_val_batch, X_15m_val_batch, X_1h_val_batch, X_gt_val_batch
                    del X_sa_val_batch, X_ta_val_batch, X_ctx_val_batch, y_val_batch, y_pred
                
                # Log metrics
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('train_mae', train_mae.result(), step=epoch)
                    tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
                    tf.summary.scalar('val_mae', val_mae.result(), step=epoch)
                
                # Check for improvement
                current_val_loss = val_loss.result()
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    # Save best model
                    self.model.save(self.model_out)
                    self.logger.info(f"  Saved best model with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                # Log epoch stats
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"  Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                    f"Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}, "
                    f"Val Loss: {val_loss.result():.4f}, Val MAE: {val_mae.result():.4f}"
                )
            else:
                # Log training metrics only
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('train_mae', train_mae.result(), step=epoch)
                
                # Save periodically
                if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                    self.model.save(f"{self.model_out.replace('.keras', '')}_epoch{epoch+1}.keras")
                    self.logger.info(f"  Saved checkpoint at epoch {epoch+1}")
                    
                # Log epoch stats
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"  Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                    f"Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}"
                )
                
            # Force garbage collection
            gc.collect()
        
        # Save final model if not already saved
        if not has_validation:
            self.model.save(self.model_out)
            self.logger.info(f"Saved final model to {self.model_out}")

    def train_lstm(self):
        """
        Train the multi-output LSTM with memory optimizations
        """
        data = self.load_training_data()
        if not data or data[0] is None:
            self.logger.error("No valid training data found => abort training.")
            return

        (X_5m, X_15m, X_1h,
         X_gt, X_sa,
         X_ta, X_ctx,
         Y, all_ts) = data

        N = len(Y)
        if N < 10:
            self.logger.error("Too few samples => abort training.")
            return
        
        # No need to shuffle - chronological order is important
        idx = np.arange(N)

        X_5m  = X_5m[idx]
        X_15m = X_15m[idx]
        X_1h  = X_1h[idx]
        X_gt  = X_gt[idx]
        X_sa  = X_sa[idx]
        X_ta  = X_ta[idx]
        X_ctx = X_ctx[idx]
        Y     = Y[idx]
        all_ts = np.array(all_ts)[idx]

        # Split into train/validation sets
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
        
        # Free memory
        del X_5m, X_15m, X_1h, X_gt, X_sa, X_ta, X_ctx, Y
        gc.collect()

        self.logger.info(
            f"Train={len(Y_train)}, Val={len(Y_val)}"
        )

        # DEBUG: row_train_start, row_train_end
        if len(ts_train) > 0:
            row_train_start = ts_train[0]
            row_train_end   = ts_train[-1]
            self.logger.info(f"Train slice => start={row_train_start}, end={row_train_end}")

        # Load/fallback scalers
        try:
            model_scaler = ModelScaler.load("models/scalers.pkl")
            self.logger.info("Loaded scalers from models/scalers.pkl.")
        except (FileNotFoundError, AttributeError, ImportError) as e:
            self.logger.warning(f"No scalers found: {e} => creating new ModelScaler.")
            model_scaler = ModelScaler()

        if self.apply_scaling:
            self.logger.info("Fitting and applying scalers...")
            try:
                model_scaler.fit_all(
                    X_5m_train, X_15m_train, X_1h_train,
                    X_gt_train, X_sa_train,
                    X_ta_train, X_ctx_train
                )
            except Exception as e:
                self.logger.warning(f"Error during scaler fitting: {e}. Proceeding without scaling.")
                self.apply_scaling = False
        else:
            self.logger.info("Scaling disabled => pass-thru transforms.")

        # Transform data if scaling is enabled
        if self.apply_scaling:
            try:
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
            except Exception as e:
                self.logger.warning(f"Error during data transformation: {e}. Using original data.")

        self.logger.info(f"Fitting model => output dim={NUM_FUTURE_STEPS}, epochs={self.epochs}, batch={self.batch_size}")
        self.logger.info(f"Chronological split => train={len(Y_train)}, val={len(Y_val)}")

        # Use our memory-efficient custom training function
        if self.grad_accum:
            self.custom_gradient_accumulation_training(
                x_train=[
                    X_5m_train, X_15m_train, X_1h_train,
                    X_gt_train, X_sa_train, X_ta_train, X_ctx_train
                ],
                y_train=Y_train,
                x_val=[
                    X_5m_val, X_15m_val, X_1h_val,
                    X_gt_val, X_sa_val, X_ta_val, X_ctx_val
                ] if len(Y_val) > 0 else None,
                y_val=Y_val if len(Y_val) > 0 else None
            )
        else:
            # Standard Keras training with callbacks
            def create_callbacks():
                # Early stopping
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=self.early_stop_patience, 
                    restore_best_weights=True,
                    verbose=1
                )
                
                # Learning rate reduction on plateau
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,         # Reduce by half
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                )
                
                # Checkpoint best models
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    f"models/checkpoint-{int(time.time())}-ep{{epoch:03d}}_val{{val_loss:.4f}}.keras",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
                
                # Memory-efficient TensorBoard logging
                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=f"logs_training/fit_{int(time.time())}",
                    histogram_freq=0,  # Disable histogram to save memory
                    update_freq='epoch'
                )
                
                # Free memory at the end of each epoch
                class MemoryCleanupCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        gc.collect()
                
                return [early_stop, reduce_lr, checkpoint, tensorboard, MemoryCleanupCallback()]

            self.model.fit(
                x=[
                    X_5m_train, X_15m_train, X_1h_train,
                    X_gt_train, X_sa_train, X_ta_train, X_ctx_train
                ],
                y=Y_train,
                validation_data=(
                    [
                        X_5m_val, X_15m_val, X_1h_val,
                        X_gt_val, X_sa_val, X_ta_val, X_ctx_val
                    ],
                    Y_val
                ) if len(Y_val) > 0 else None,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=create_callbacks(),
                verbose=1,
                shuffle=False
            )

    def train_rl_offline(
        self,
        rl_csv: str,
        rl_out: str = "models/rl_DQNAgent.weights.h5",
        rl_epochs: int = 5,
        rl_batches: int = 500,
        state_dim: int = NUM_FUTURE_STEPS + 3
    ):
        """
        Offline RL training with memory optimizations
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
                except Exception as e:
                    failed_rows += 1
                    self.logger.debug(f"Error parsing row: {e}")
                    continue

        self.logger.warning(f"RL parse => skipped {failed_rows} invalid rows.")
        if len(transitions) < 10:
            self.logger.error("Too few RL transitions => abort RL training.")
            return

        self.logger.info(f"Loaded {len(transitions)} RL transitions from {rl_csv}.")

        # Initialize DQN with compatible parameters
        dqn = DQNAgent(
            state_dim=state_dim,
            gamma=0.99,
            lr=0.001,
            batch_size=64,
            max_memory=len(transitions)+1,
            epsilon_start=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            update_target_steps=100
        )

        # Store transitions
        for (s, a, r, s2, d) in transitions:
            dqn.store_transition(s, a, r, s2, d)

        # Offline training with progress tracking
        for ep in range(rl_epochs):
            self.logger.info(f"[RL] Epoch {ep+1}/{rl_epochs}")
            epoch_losses = []
            
            for b in range(rl_batches):
                loss = dqn.train_step()
                if loss is not None:
                    epoch_losses.append(loss)
                
                if (b+1) % 100 == 0:
                    avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                    self.logger.info(f"  Batch {b+1}/{rl_batches} - Average loss: {avg_loss:.6f}")
            
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
            self.logger.info(f"  Epoch {ep+1} complete - Average loss: {avg_epoch_loss:.6f}")
            
            # Save model after each epoch
            dqn.save()
            self.logger.info(f"  Saved DQN weights at epoch {ep+1}")

        # Final save
        try:
            dqn.save()
            self.logger.info(f"RL weights saved => {rl_out}")
        except Exception as e:
            self.logger.error(f"Error saving RL => {e}")

    def run(self, rl_csv=None, rl_out=None, rl_epochs=5, rl_batches=500):
        """
        Main entry point
        """
        if not self.skip_lstm:
            try:
                self.logger.info("=== Starting LSTM Training ===")
                self.train_lstm()
                self.logger.info("=== LSTM Training Complete ===")
            except Exception as e:
                self.logger.error(f"Error during LSTM training: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.info("Skipping LSTM training.")

        if rl_csv:
            try:
                self.logger.info("=== Starting RL Training ===")
                self.train_rl_offline(
                    rl_csv=rl_csv,
                    rl_out=rl_out,
                    rl_epochs=rl_epochs,
                    rl_batches=rl_batches
                )
                self.logger.info("=== RL Training Complete ===")
            except Exception as e:
                self.logger.error(f"Error during RL training: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.info("No RL CSV => skipping offline RL training.")
            
        self.logger.info("=== Training Pipeline Complete ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ultra-optimized training script for memory-constrained environments"
    )
    parser.add_argument("--csv", type=str, default="training_data/training_data.csv",
                        help="Path to multi-output training_data.csv.")
    parser.add_argument("--model_out", type=str, default="models/advanced_lstm_model.keras",
                        help="File path for the LSTM model.")
    parser.add_argument("--epochs", type=int, default=100, help="LSTM epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="early_stop_patience")
    parser.add_argument("--batch_size", type=int, default=2, help="LSTM batch size (use small values like 2).")
    parser.add_argument("--no_scale", action="store_true",
                        help="Disable feature scaling.")
    parser.add_argument("--skip_lstm", action="store_true",
                        help="Skip LSTM training entirely.")
    parser.add_argument("--max_rows", type=int, default=0, 
                        help="Load x rows from csv file. 0 is all.")
    parser.add_argument("--grad_accum", action="store_true", 
                        help="Use gradient accumulation for memory efficiency")
    parser.add_argument("--accum_steps", type=int, default=16,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["tiny", "small", "medium", "large", "xlarge"],
                        help="Model size (tiny=24, small=32, medium=48, large=64, xlarge=96 units)")
    parser.add_argument("--no_reduce_precision", action="store_true",
                        help="Don't use reduced precision (use full float32)")

    # RL
    parser.add_argument("--rl_csv", type=str, default=RL_TRANSITIONS_FILE,
                        help="Path to RL transitions CSV.")
    parser.add_argument("--rl_out", type=str, default="models/rl_DQNAgent.weights.h5",
                        help="Output file for RL weights.")
    parser.add_argument("--rl_epochs", type=int, default=5,
                        help="Offline RL training epochs.")
    parser.add_argument("--rl_batches", type=int, default=500,
                        help="Offline RL mini-batch updates per epoch.")

    return parser.parse_args()


def main():
    os.system('cls' if os.name=='nt' else 'clear')
    print("\n" + "="*80)
    print("ULTRA-OPTIMIZED TRADING BOT TRAINING SCRIPT")
    print("="*80 + "\n")
    
    args = parse_args()
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Number of GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    print(f"Model size: {args.model_size.upper()}")
    print("="*80 + "\n")

    trainer = Trainer(
        training_csv=args.csv,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        apply_scaling=not args.no_scale,
        skip_lstm=args.skip_lstm,
        max_rows=args.max_rows,
        grad_accum=args.grad_accum,
        accum_steps=args.accum_steps,
        model_size=args.model_size,
        reduce_precision=not args.no_reduce_precision
    )
    
    try:
        trainer.run(
            rl_csv=args.rl_csv,
            rl_out=args.rl_out,
            rl_epochs=args.rl_epochs,
            rl_batches=args.rl_batches
        )
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
    except Exception as e:
        print("\n" + "="*80)
        print(f"ERROR DURING TRAINING: {e}")
        print("="*80 + "\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
# python fitter.py --no_reduce_precision --batch_size 8 --model_size small --grad_accum --accum_steps 4
# tensorboard --logdir=logs_training
# nvidia-smi -l 1