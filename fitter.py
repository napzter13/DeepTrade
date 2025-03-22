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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Fitter")

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages
tf.get_logger().setLevel('ERROR')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TF from grabbing all GPU memory

# Import our modules - with proper error handling
try:
    from botlib.environment import NUM_FUTURE_STEPS
    from botlib.models import load_advanced_lstm_model
    from botlib.input_preprocessing import ModelScaler, prepare_for_model_inputs
    from botlib.rl import DQNAgent, ACTIONS
    from botlib.models import safe_mse_loss, TimeSeriesEncoder, TabularEncoder, LightEnsembleModel
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please make sure the botlib module is in your Python path.")
    sys.exit(1)

# Constants
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
        epochs=100,
        early_stop_patience=20,
        batch_size=2,
        apply_scaling=True,
        train_ratio=0.8,
        val_ratio=0.2,
        skip_lstm=False,
        max_rows=0,
        grad_accum=True,
        accum_steps=16,
        model_size="small",
        reduce_precision=True
    ):
        """
        Ultra-optimized Trainer for memory-constrained environments
        """
        self.logger = logger
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        os.makedirs("logs_training", exist_ok=True)
        
        self.training_csv = training_csv
        self.model_out = model_out
        self.window_5m = window_5m
        self.feature_5m = feature_5m
        self.window_15m = window_15m
        self.feature_15m = feature_15m
        self.window_1h = window_1h
        self.feature_1h = feature_1h
        self.window_google_trend = window_google_trend
        self.feature_google_trend = feature_google_trend
        self.santiment_dim = santiment_dim
        self.ta_dim = ta_dim
        self.signal_dim = signal_dim
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size
        self.apply_scaling = apply_scaling
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.skip_lstm = skip_lstm
        self.max_rows = max_rows
        self.grad_accum = grad_accum
        self.accum_steps = accum_steps
        self.model_size = model_size
        self.reduce_precision = reduce_precision
        
        # Set base units based on model size
        base_units_map = {
            "tiny": 24,
            "small": 32,
            "medium": 48,
            "large": 64,
            "xlarge": 96
        }
        self.base_units = base_units_map.get(model_size, 32)

        # Verify training data file exists
        if not os.path.exists(self.training_csv):
            self.logger.error(f"Training data file not found: {self.training_csv}")
            print(f"ERROR: Training data file not found: {self.training_csv}")
            return
            
        # Initialize model if not skipping LSTM training
        if not skip_lstm:
            self.logger.info(f"Initializing model with {NUM_FUTURE_STEPS} outputs and {self.base_units} units")
            try:
                # Try to load existing model
                if os.path.exists(self.model_out):
                    self.logger.info(f"Loading existing model from {self.model_out}")
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
                    
                    # Enable mixed precision if requested
                    if reduce_precision:
                        try:
                            policy = tf.keras.mixed_precision.Policy('mixed_float16')
                            tf.keras.mixed_precision.set_global_policy(policy)
                            self.logger.info(f"Mixed precision enabled with policy: {policy.name}")
                        except:
                            self.logger.warning("Mixed precision not available, using default precision")

                    # Check if GPU is available
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        try:
                            # Configure memory growth to avoid OOM errors
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            self.logger.info(f"Configured GPU memory growth: {len(gpus)} GPUs available")
                        except RuntimeError as e:
                            self.logger.error(f"GPU configuration error: {e}")
                            
                    # Create model
                    try:
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
                        
                        if self.model is None:
                            self.logger.error("Model creation returned None")
                            print("ERROR: Model creation returned None")
                        else:
                            self.logger.info(f"Model created successfully with {self.model.count_params()} parameters")
                    except Exception as e:
                        self.logger.error(f"Error creating model: {e}")
                        print(f"ERROR creating model: {e}")
                        raise
            except Exception as e:
                self.logger.error(f"Failed to initialize model: {e}")
                print(f"ERROR: Failed to initialize model: {e}")
                import traceback
                print(traceback.format_exc())
                raise
        else:
            self.logger.info("Skipping model initialization (skip_lstm=True).")
            self.model = None

    def load_training_data(self):
        """
        Loads training data from CSV with memory optimizations
        """
        if not os.path.exists(self.training_csv):
            self.logger.error(f"Training CSV not found: {self.training_csv}")
            print(f"ERROR: Training CSV not found: {self.training_csv}")
            return [None]*9
            
        # Check file size
        file_size = os.path.getsize(self.training_csv)
        if file_size == 0:
            self.logger.error(f"Training CSV is empty: {self.training_csv}")
            print(f"ERROR: Training CSV is empty: {self.training_csv}")
            return [None]*9
        else:
            self.logger.info(f"Training CSV size: {file_size} bytes")

        all_5m = []
        all_15m = []
        all_1h = []
        all_gt = []
        all_sa = []
        all_ta = []
        all_ctx = []
        all_Y = []
        all_ts = []
        
        start_time = time.time()
        
        # Count rows first to preallocate memory
        total_rows = 0
        try:
            with open(self.training_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                for _ in reader:
                    total_rows += 1
            
            if total_rows == 0:
                self.logger.error(f"Training CSV has no data rows: {self.training_csv}")
                print(f"ERROR: Training CSV has no data rows: {self.training_csv}")
                return [None]*9
                
            self.logger.info(f"Found {total_rows} data rows in CSV file.")
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            print(f"ERROR reading CSV file: {e}")
            return [None]*9
        
        # Limit to max_rows if specified
        if self.max_rows > 0:
            total_rows = min(total_rows, self.max_rows)
            self.logger.info(f"Limiting to {total_rows} rows as specified.")
            
        # Now read the actual data
        row_count = 0
        error_rows = 0
        
        try:
            with open(self.training_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                # Check required columns
                required_cols = ["timestamp", "arr_5m", "arr_15m", "arr_1h", "arr_google_trend", 
                              "arr_santiment", "arr_ta_63", "arr_ctx_11"]
                missing_cols = [col for col in required_cols if col not in reader.fieldnames]
                
                # Check output columns
                for i in range(1, NUM_FUTURE_STEPS+1):
                    col_name = f"y_{i}"
                    if col_name not in reader.fieldnames:
                        missing_cols.append(col_name)
                        
                if missing_cols:
                    self.logger.error(f"CSV missing columns: {missing_cols}")
                    print(f"ERROR: CSV missing columns: {missing_cols}")
                    return [None]*9
                
                # Process rows
                for row_idx, row in enumerate(reader):
                    if self.max_rows > 0 and row_count >= self.max_rows:
                        break

                    try:
                        # Report progress periodically
                        if row_count % 500 == 0:
                            self.logger.info(f"Loading data: {row_count}/{total_rows} rows processed")
                            # Force garbage collection
                            gc.collect()
                            
                        timestamp_str = row["timestamp"]
                        arr_5m_str = row["arr_5m"]
                        arr_15m_str = row["arr_15m"]
                        arr_1h_str = row["arr_1h"]
                        arr_gt_str = row["arr_google_trend"]
                        arr_sa_str = row["arr_santiment"]
                        arr_ta_str = row["arr_ta_63"]
                        arr_ctx_str = row["arr_ctx_11"]

                        # Parse JSON
                        try:
                            arr_5m_list = json.loads(arr_5m_str)[0]
                            arr_15m_list = json.loads(arr_15m_str)[0]
                            arr_1h_list = json.loads(arr_1h_str)[0]
                            arr_gt_list = json.loads(arr_gt_str)[0]
                            arr_sa_list = json.loads(arr_sa_str)[0]
                            arr_ta_list = json.loads(arr_ta_str)[0]
                            arr_ctx_list = json.loads(arr_ctx_str)[0]
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"JSON parsing error in row {row_idx+1}: {e}")
                            error_rows += 1
                            continue

                        # Verify dimensions
                        if len(arr_5m_list) != 241:
                            self.logger.warning(f"Row {row_idx+1}: arr_5m has wrong dimension: {len(arr_5m_list)}")
                            error_rows += 1
                            continue
                        if len(arr_15m_list) != 241:
                            self.logger.warning(f"Row {row_idx+1}: arr_15m has wrong dimension: {len(arr_15m_list)}")
                            error_rows += 1
                            continue
                        if len(arr_1h_list) != 241:
                            self.logger.warning(f"Row {row_idx+1}: arr_1h has wrong dimension: {len(arr_1h_list)}")
                            error_rows += 1
                            continue
                        if len(arr_gt_list) != 24:
                            self.logger.warning(f"Row {row_idx+1}: arr_google_trend has wrong dimension: {len(arr_gt_list)}")
                            error_rows += 1
                            continue
                        if len(arr_sa_list) != 12:
                            self.logger.warning(f"Row {row_idx+1}: arr_santiment has wrong dimension: {len(arr_sa_list)}")
                            error_rows += 1
                            continue
                        if len(arr_ta_list) != 63:
                            self.logger.warning(f"Row {row_idx+1}: arr_ta_63 has wrong dimension: {len(arr_ta_list)}")
                            error_rows += 1
                            continue
                        if len(arr_ctx_list) != 11:
                            self.logger.warning(f"Row {row_idx+1}: arr_ctx_11 has wrong dimension: {len(arr_ctx_list)}")
                            error_rows += 1
                            continue
                        
                        # Skip rows with all zeros
                        if sum(arr_5m_list[0]) == 0:
                            self.logger.warning(f"Row {row_idx+1}: Skipping zero data")
                            error_rows += 1
                            continue

                        # Parse target values
                        y_vec = []
                        for i in range(1, NUM_FUTURE_STEPS+1):
                            col_name = f"y_{i}"
                            try:
                                val_str = row[col_name]
                                val_f = float(val_str)  # each in [-1,1]
                                y_vec.append(val_f)
                            except (KeyError, ValueError) as e:
                                self.logger.warning(f"Row {row_idx+1}: Error parsing {col_name}: {e}")
                                error_rows += 1
                                continue

                        # Add to arrays
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
                        self.logger.warning(f"Error in row {row_idx+1}: {e}")
                        error_rows += 1
                        continue
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            print(f"ERROR processing CSV: {e}")
            import traceback
            print(traceback.format_exc())
            return [None]*9

        if error_rows > 0:
            self.logger.warning(f"Skipped {error_rows} rows with errors")
            
        # Check if we loaded any valid data
        if row_count == 0:
            self.logger.error("No valid data rows found in CSV")
            print("ERROR: No valid data rows found in CSV")
            return [None]*9
            
        # Convert to numpy arrays using appropriate precision
        dtype = np.float32 if not self.reduce_precision else np.float16
        
        try:
            X_5m = np.array(all_5m, dtype=dtype)
            X_15m = np.array(all_15m, dtype=dtype)
            X_1h = np.array(all_1h, dtype=dtype)
            X_gt = np.array(all_gt, dtype=dtype)
            X_sa = np.array(all_sa, dtype=dtype)
            X_ta = np.array(all_ta, dtype=dtype)
            X_ctx = np.array(all_ctx, dtype=dtype)
            Y = np.array(all_Y, dtype=dtype)
        except Exception as e:
            self.logger.error(f"Error converting to numpy arrays: {e}")
            print(f"ERROR converting to numpy arrays: {e}")
            return [None]*9

        # Clean up to free memory
        del all_5m, all_15m, all_1h, all_gt, all_sa, all_ta, all_ctx, all_Y
        gc.collect()

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Loaded {len(X_5m)} rows in {elapsed_time:.2f}s. X_5m={X_5m.shape}, Y={Y.shape}"
        )
        print(f"Successfully loaded {len(X_5m)} rows in {elapsed_time:.2f}s")
        print(f"Data shapes: X_5m={X_5m.shape}, Y={Y.shape}")
        
        return X_5m, X_15m, X_1h, X_gt, X_sa, X_ta, X_ctx, Y, all_ts

    def split_data(self, N):
        """
        Split data into train/validation sets
        """
        train_end = int(N * self.train_ratio)
        val_end = int(N * (self.train_ratio + self.val_ratio))
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
            print(f"ERROR: Inconsistent data sizes: y_train={len(y_train)}, X_5m_train={len(X_5m_train)}")
            return
            
        # Prepare validation data if present
        has_validation = (x_val is not None and y_val is not None and len(y_val) > 0)
        if has_validation:
            X_5m_val, X_15m_val, X_1h_val, X_gt_val, X_sa_val, X_ta_val, X_ctx_val = x_val
            val_samples = len(y_val)
            val_steps = max(1, val_samples // self.batch_size)
        
        self.logger.info(f"Starting training with gradient accumulation")
        print(f"Starting training with gradient accumulation")
        self.logger.info(f"Samples: {train_samples}, Steps per epoch: {steps_per_epoch}")
        print(f"Samples: {train_samples}, Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"Batch size: {self.batch_size}, Accumulation steps: {self.accum_steps}")
        print(f"Batch size: {self.batch_size}, Accumulation steps: {self.accum_steps}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Reset metrics
            try:
                train_loss.reset_states()
                train_mae.reset_states()
            except:
                # Fallback for older TF versions
                train_loss.reset()
                train_mae.reset()
            
            # Manual batching and gradient accumulation
            accumulated_gradients = None
            accumulation_count = 0
            
            # Process in batches
            for step in range(steps_per_epoch):
                try:
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
                        print(f"  Step {step+1}/{steps_per_epoch} - Loss: {train_loss.result():.4f}")
                except Exception as e:
                    self.logger.error(f"Error in training step {step+1}: {e}")
                    print(f"Error in training step {step+1}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            # Validation if available
            if has_validation:
                try:
                    # Reset validation metrics
                    try:
                        val_loss.reset_states()
                        val_mae.reset_states()
                    except:
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
                        print(f"  Saved best model with val_loss: {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stop_patience:
                            self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    
                    # Log epoch stats
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"  Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                        f"Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}, "
                        f"Val Loss: {val_loss.result():.4f}, Val MAE: {val_mae.result():.4f}"
                    )
                    print(
                        f"  Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                        f"Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}, "
                        f"Val Loss: {val_loss.result():.4f}, Val MAE: {val_mae.result():.4f}"
                    )
                except Exception as e:
                    self.logger.error(f"Error in validation: {e}")
                    print(f"Error in validation: {e}")
                    import traceback
                    print(traceback.format_exc())
            else:
                # Log training metrics only
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('train_mae', train_mae.result(), step=epoch)
                
                # Save periodically
                if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                    self.model.save(f"{self.model_out.replace('.keras', '')}_epoch{epoch+1}.keras")
                    self.logger.info(f"  Saved checkpoint at epoch {epoch+1}")
                    print(f"  Saved checkpoint at epoch {epoch+1}")
                    
                # Log epoch stats
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"  Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                    f"Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}"
                )
                print(
                    f"  Epoch {epoch+1}/{self.epochs} - {epoch_time:.2f}s - "
                    f"Loss: {train_loss.result():.4f}, MAE: {train_mae.result():.4f}"
                )
                
            # Force garbage collection
            gc.collect()
        
        # Save final model if not already saved
        if not has_validation:
            self.model.save(self.model_out)
            self.logger.info(f"Saved final model to {self.model_out}")
            print(f"Saved final model to {self.model_out}")

    def train_lstm(self):
        """
        Train the multi-output LSTM with memory optimizations
        """
        self.logger.info("Starting LSTM training...")
        print("Starting LSTM training...")
        
        # Validate training data first
        if not os.path.exists(self.training_csv):
            self.logger.error(f"Training data file not found: {self.training_csv}")
            print(f"ERROR: Training data file not found: {self.training_csv}")
            return
            
        if os.path.getsize(self.training_csv) == 0:
            self.logger.error(f"Training data file is empty: {self.training_csv}")
            print(f"ERROR: Training data file is empty: {self.training_csv}")
            return
        
        # Load the data
        data = self.load_training_data()
        if not data or data[0] is None:
            self.logger.error("No valid training data found => abort training.")
            print("ERROR: No valid training data found => abort training.")
            return

        (X_5m, X_15m, X_1h,
         X_gt, X_sa,
         X_ta, X_ctx,
         Y, all_ts) = data

        N = len(Y)
        if N < 10:
            self.logger.error("Too few samples => abort training.")
            print("ERROR: Too few samples => abort training.")
            return
        
        # No need to shuffle - chronological order is important
        self.logger.info(f"Total samples loaded: {N}")
        print(f"Total samples loaded: {N}")

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

        self.logger.info(f"Train={len(Y_train)}, Val={len(Y_val)}")
        print(f"Train={len(Y_train)}, Val={len(Y_val)}")

        # DEBUG: row_train_start, row_train_end
        if len(ts_train) > 0:
            row_train_start = ts_train[0]
            row_train_end   = ts_train[-1]
            self.logger.info(f"Train slice => start={row_train_start}, end={row_train_end}")
            print(f"Train slice => start={row_train_start}, end={row_train_end}")

        # Load/fallback scalers
        try:
            model_scaler = ModelScaler.load("models/scalers.pkl")
            self.logger.info("Loaded scalers from models/scalers.pkl.")
            print("Loaded scalers from models/scalers.pkl.")
        except Exception as e:
            self.logger.warning(f"No scalers found: {e} => creating new ModelScaler.")
            print(f"No scalers found: {e} => creating new ModelScaler.")
            model_scaler = ModelScaler()

        if self.apply_scaling:
            self.logger.info("Fitting and applying scalers...")
            print("Fitting and applying scalers...")
            try:
                model_scaler.fit_all(
                    X_5m_train, X_15m_train, X_1h_train,
                    X_gt_train, X_sa_train,
                    X_ta_train, X_ctx_train
                )
                print("Successfully fitted scalers.")
            except Exception as e:
                self.logger.warning(f"Error during scaler fitting: {e}. Proceeding without scaling.")
                print(f"Error during scaler fitting: {e}. Proceeding without scaling.")
                self.apply_scaling = False
        else:
            self.logger.info("Scaling disabled => pass-thru transforms.")
            print("Scaling disabled => pass-thru transforms.")

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
                print("Successfully transformed training data.")
                
                if len(Y_val) > 0:
                    (X_5m_val, X_15m_val, X_1h_val,
                     X_gt_val, X_sa_val,
                     X_ta_val, X_ctx_val) = prepare_for_model_inputs(
                        X_5m_val, X_15m_val, X_1h_val,
                        X_gt_val, X_sa_val, X_ta_val, X_ctx_val,
                        model_scaler
                    )
                    print("Successfully transformed validation data.")
            except Exception as e:
                self.logger.warning(f"Error during data transformation: {e}. Using original data.")
                print(f"Error during data transformation: {e}. Using original data.")

        self.logger.info(f"Fitting model => output dim={NUM_FUTURE_STEPS}, epochs={self.epochs}, batch={self.batch_size}")
        print(f"Fitting model => output dim={NUM_FUTURE_STEPS}, epochs={self.epochs}, batch={self.batch_size}")
        self.logger.info(f"Chronological split => train={len(Y_train)}, val={len(Y_val)}")
        print(f"Chronological split => train={len(Y_train)}, val={len(Y_val)}")

        # Use our memory-efficient custom training function
        if self.grad_accum:
            self.logger.info("Using gradient accumulation for training")
            print("Using gradient accumulation for training")
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
            self.logger.info("Using standard TensorFlow training")
            print("Using standard TensorFlow training")
            
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

            try:
                self.logger.info("Starting model.fit...")
                print("Starting model.fit...")
                
                validation_data = (
                    [
                        X_5m_val, X_15m_val, X_1h_val,
                        X_gt_val, X_sa_val, X_ta_val, X_ctx_val
                    ],
                    Y_val
                ) if len(Y_val) > 0 else None
                
                history = self.model.fit(
                    x=[
                        X_5m_train, X_15m_train, X_1h_train,
                        X_gt_train, X_sa_train, X_ta_train, X_ctx_train
                    ],
                    y=Y_train,
                    validation_data=validation_data,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=create_callbacks(),
                    verbose=1,
                    shuffle=False
                )
                
                self.logger.info("Model training completed successfully")
                print("Model training completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error in model.fit: {e}")
                print(f"ERROR in model.fit: {e}")
                import traceback
                print(traceback.format_exc())

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
            print(f"ERROR: No RL CSV found => {rl_csv}")
            return
            
        self.logger.info(f"Starting offline RL training with {rl_csv}")
        print(f"Starting offline RL training with {rl_csv}")

        transitions = []
        failed_rows = 0
        try:
            with open(rl_csv, "r", encoding="utf-8") as f:
                # Check file size
                if os.path.getsize(rl_csv) == 0:
                    self.logger.error(f"RL CSV is empty: {rl_csv}")
                    print(f"ERROR: RL CSV is empty: {rl_csv}")
                    return
                    
                reader = csv.DictReader(f)
                
                # Check required columns
                required_cols = ["old_state", "action", "reward", "new_state", "done"]
                missing_cols = [col for col in required_cols if col not in reader.fieldnames]
                if missing_cols:
                    self.logger.error(f"RL CSV missing columns: {missing_cols}")
                    print(f"ERROR: RL CSV missing columns: {missing_cols}")
                    return
                    
                total_rows = sum(1 for _ in reader)
                if total_rows == 0:
                    self.logger.error(f"RL CSV has no data rows: {rl_csv}")
                    print(f"ERROR: RL CSV has no data rows: {rl_csv}")
                    return
                
                # Return to start of file and skip header
                f.seek(0)
                next(csv.reader(f))
                
                row_count = 0
                for row in csv.DictReader(f):
                    try:
                        old_state_str = row["old_state"]
                        action_str    = row["action"]
                        reward_val    = float(row["reward"])
                        new_state_str = row["new_state"]
                        done_val      = row["done"].strip()

                        try:
                            old_st = np.array(json.loads(old_state_str), dtype=np.float32)
                            new_st = np.array(json.loads(new_state_str), dtype=np.float32)
                        except json.JSONDecodeError as e:
                            failed_rows += 1
                            self.logger.warning(f"JSON error in RL CSV row {row_count+1}: {e}")
                            continue

                        done_flag = (done_val.lower() in ["1", "true"])

                        # Verify dimensions
                        if old_st.shape[0] != state_dim: 
                            failed_rows += 1
                            self.logger.warning(f"RL CSV row {row_count+1}: Invalid old_state dimension {old_st.shape}")
                            continue
                        if new_st.shape[0] != state_dim:
                            failed_rows += 1
                            self.logger.warning(f"RL CSV row {row_count+1}: Invalid new_state dimension {new_st.shape}")
                            continue
                        if action_str not in ACTIONS:
                            failed_rows += 1
                            self.logger.warning(f"RL CSV row {row_count+1}: Invalid action {action_str}")
                            continue

                        transitions.append((old_st, action_str, reward_val, new_st, done_flag))
                        row_count += 1
                        
                        # Progress reporting
                        if row_count % 1000 == 0:
                            self.logger.info(f"Loading RL data: {row_count}/{total_rows} rows processed")
                            
                    except Exception as e:
                        failed_rows += 1
                        self.logger.debug(f"Error parsing RL row: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading RL CSV: {e}")
            print(f"ERROR reading RL CSV: {e}")
            import traceback
            print(traceback.format_exc())
            return

        self.logger.warning(f"RL parse => skipped {failed_rows} invalid rows.")
        print(f"RL parse => skipped {failed_rows} invalid rows.")
        
        if len(transitions) < 10:
            self.logger.error("Too few RL transitions => abort RL training.")
            print("ERROR: Too few RL transitions => abort RL training.")
            return

        self.logger.info(f"Loaded {len(transitions)} RL transitions from {rl_csv}.")
        print(f"Loaded {len(transitions)} RL transitions from {rl_csv}.")

        # Initialize DQN with compatible parameters
        try:
            self.logger.info("Initializing DQN agent...")
            print("Initializing DQN agent...")
            
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
            self.logger.info("Storing transitions in DQN memory...")
            print("Storing transitions in DQN memory...")
            
            for i, (s, a, r, s2, d) in enumerate(transitions):
                dqn.store_transition(s, a, r, s2, d)
                if i % 1000 == 0:
                    self.logger.info(f"Stored {i}/{len(transitions)} transitions")

            # Offline training with progress tracking
            self.logger.info(f"Starting DQN training for {rl_epochs} epochs...")
            print(f"Starting DQN training for {rl_epochs} epochs...")
            
            for ep in range(rl_epochs):
                self.logger.info(f"[RL] Epoch {ep+1}/{rl_epochs}")
                print(f"[RL] Epoch {ep+1}/{rl_epochs}")
                epoch_losses = []
                
                for b in range(rl_batches):
                    loss = dqn.train_step()
                    if loss is not None:
                        epoch_losses.append(loss)
                    
                    if (b+1) % 100 == 0:
                        avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                        self.logger.info(f"  Batch {b+1}/{rl_batches} - Average loss: {avg_loss:.6f}")
                        print(f"  Batch {b+1}/{rl_batches} - Average loss: {avg_loss:.6f}")
                
                avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
                self.logger.info(f"  Epoch {ep+1} complete - Average loss: {avg_epoch_loss:.6f}")
                print(f"  Epoch {ep+1} complete - Average loss: {avg_epoch_loss:.6f}")
                
                # Save model after each epoch
                dqn.save()
                self.logger.info(f"  Saved DQN weights at epoch {ep+1}")
                print(f"  Saved DQN weights at epoch {ep+1}")

            # Final save
            try:
                dqn.save()
                self.logger.info(f"RL weights saved => {rl_out}")
                print(f"RL weights saved => {rl_out}")
            except Exception as e:
                self.logger.error(f"Error saving RL => {e}")
                print(f"Error saving RL => {e}")
                
        except Exception as e:
            self.logger.error(f"Error in RL training: {e}")
            print(f"ERROR in RL training: {e}")
            import traceback
            print(traceback.format_exc())

    def run(self, rl_csv=None, rl_out=None, rl_epochs=5, rl_batches=500):
        """
        Main entry point
        """
        print("\nStarting the training run...")
        
        # Final GPU setup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"Memory growth enabled for GPU {gpu}")
                except RuntimeError as e:
                    self.logger.warning(f"Error setting GPU memory growth: {e}")
        
        if not self.skip_lstm:
            try:
                self.logger.info("=== Starting LSTM Training ===")
                print("=== Starting LSTM Training ===")
                self.train_lstm()
                self.logger.info("=== LSTM Training Complete ===")
                print("=== LSTM Training Complete ===")
            except Exception as e:
                self.logger.error(f"Error during LSTM training: {e}")
                print(f"ERROR during LSTM training: {e}")
                import traceback
                traceback_str = traceback.format_exc()
                self.logger.error(traceback_str)
                print(traceback_str)
        else:
            self.logger.info("Skipping LSTM training.")
            print("Skipping LSTM training.")

        if rl_csv and os.path.exists(rl_csv):
            try:
                self.logger.info("=== Starting RL Training ===")
                print("=== Starting RL Training ===")
                self.train_rl_offline(
                    rl_csv=rl_csv,
                    rl_out=rl_out,
                    rl_epochs=rl_epochs,
                    rl_batches=rl_batches
                )
                self.logger.info("=== RL Training Complete ===")
                print("=== RL Training Complete ===")
            except Exception as e:
                self.logger.error(f"Error during RL training: {e}")
                print(f"ERROR during RL training: {e}")
                import traceback
                traceback_str = traceback.format_exc()
                self.logger.error(traceback_str)
                print(traceback_str)
        else:
            if rl_csv:
                self.logger.info(f"RL CSV not found => {rl_csv}")
                print(f"RL CSV not found => {rl_csv}")
            self.logger.info("Skipping offline RL training.")
            print("Skipping offline RL training.")
            
        self.logger.info("=== Training Pipeline Complete ===")
        print("=== Training Pipeline Complete ===")


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
    print(f"Training file: {args.csv}")
    print(f"Script arguments: {args}")
    print("="*80 + "\n")

    # Configure GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found GPU: {gpu}")
            try:
                # Configure memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu}")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
                
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"WARNING: Training CSV file not found at {args.csv}")
    
    # Initialize trainer
    try:
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
        
        # Run training
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



# python fitter.py --no_reduce_precision --batch_size 8 --model_size xlarge --grad_accum --accum_steps 4
# tensorboard --logdir=logs_training
# nvidia-smi -l 1
