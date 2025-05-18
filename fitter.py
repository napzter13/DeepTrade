#!/usr/bin/env python3
"""
fitter.py - Ultra-optimized version for massive model training

Usage:
python fitter.py --model_size massive --batch_size 4 --grad_accum --accum_steps 8 --rl_epochs 10

This version is optimized for training 500MB+ models with sophisticated memory management
and gradient accumulation techniques.
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
import shutil  # Add shutil for directory removal
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
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
    from botlib.nn_model import safe_mse_loss
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please make sure the botlib module is in your Python path.")
    sys.exit(1)

# Constants
RL_TRANSITIONS_FILE = os.path.join("training_data", "rl_transitions.csv")
LSTM_DATA_FILE = os.path.join("training_data", "lstm_samples.csv")

# Optimize GPU setup
def setup_gpu():
    """Configure GPU for optimal training performance with massive models"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Try to enable tensor cores for better performance
            try:
                tf.config.experimental.enable_tensor_float_32_execution(True)
            except:
                pass
                
            # Enable XLA compilation
            try:
                tf.config.optimizer.set_jit(True)
            except:
                pass
            
            # Set memory limit per GPU (to avoid OOM with massive models)
            try:
                for i, gpu in enumerate(gpus):
                    # Leave 10% memory free for system
                    memory_limit = int(tf.config.experimental.get_memory_info(gpu)['total'] * 0.9)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    logger.info(f"Set memory limit for GPU {i} to {memory_limit/1024/1024:.2f} MB")
            except Exception as e:
                logger.warning(f"Could not set GPU memory limits: {e}")
                
            logger.info(f"GPU setup complete. Found {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    
    logger.warning("No GPUs found. Running on CPU only.")
    return False

# Call GPU setup
setup_gpu()

# Enhanced data batch generator with smart caching for memory efficiency
class MemoryEfficientGenerator(tf.keras.utils.Sequence):
    """Data generator with caching and memory management optimized for massive models"""
    
    def __init__(self, 
                 csv_file, 
                 batch_size=4,  # Smaller batch size for massive models
                 shuffle=True, 
                 validation=False, 
                 val_split=0.2,
                 max_rows=None,
                 scaler=None,
                 cache_size=32,  # Cache a few batches to speed up training
                 chunksize=10000):  # Read CSV in chunks to reduce memory usage
        """Initialize the generator"""
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation = validation
        self.val_split = val_split
        self.max_rows = max_rows
        self.scaler = scaler
        self.cache_size = cache_size
        self.cache = {}  # Batch cache to speed up training
        self.chunksize = chunksize
        self.chunks_loaded = False
        self.chunk_data = None
        self.prefetch_thread = None
        self.prefetch_lock = None
        self.next_chunk_idx = 0
        
        # Count rows in CSV file and preload column indices
        self._initialize_csv_metadata()
        
        # Limit rows if specified
        if self.max_rows and self.max_rows < self.total_rows:
            self.total_rows = self.max_rows
            
        # Split data indices
        if validation:
            # Validation set (last val_split portion)
            self.start_idx = int(self.total_rows * (1 - val_split))
            self.end_idx = self.total_rows
        else:
            # Training set (first 1-val_split portion)
            self.start_idx = 0
            self.end_idx = int(self.total_rows * (1 - val_split))
            
        # Row indices for batch selection
        self.indices = list(range(self.start_idx + 1, self.end_idx + 1))
        
        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        logger.info(f"Generator created with {len(self.indices)} {'validation' if validation else 'training'} samples")
        
        # Initialize thread pool for prefetching (optional)
        try:
            import threading
            self.prefetch_lock = threading.Lock()
            # Start prefetch thread
            self._start_prefetch_thread()
        except ImportError:
            logger.warning("Threading module not available, prefetching disabled")
            self.prefetch_thread = None
            self.prefetch_lock = None
        
    def _initialize_csv_metadata(self):
        """Count rows and preload column indices from CSV file efficiently"""
        try:
            import pandas as pd
            
            # Read just the header to get column names
            df_header = pd.read_csv(self.csv_file, nrows=0)
            self.column_indices = {name: i for i, name in enumerate(df_header.columns)}
            
            # Count rows efficiently without loading the entire file
            with open(self.csv_file, 'r') as f:
                # Skip header
                next(f)
                self.total_rows = sum(1 for _ in f)
                
            # Check required columns
            required_columns = ["arr_5m", "arr_15m", "arr_1h", "arr_google_trend", 
                               "arr_santiment", "arr_ta_63", "arr_ctx_11"]
            
            missing = [col for col in required_columns if col not in self.column_indices]
            if missing:
                logger.error(f"Missing required columns in CSV: {missing}")
                
            # Check y columns
            self.y_columns = []
            for i in range(1, NUM_FUTURE_STEPS+1):
                col = f"y_{i}"
                if col in self.column_indices:
                    self.y_columns.append(col)
                else:
                    logger.warning(f"Missing target column {col}")
            
            # If no target columns found, create default ones with zeros
            if not self.y_columns:
                logger.warning(f"No target columns found. Creating default targets with zeros.")
                self.default_target_count = NUM_FUTURE_STEPS
                self.use_default_targets = True
            else:
                self.use_default_targets = False
                
            logger.info(f"Found {len(self.y_columns)} target columns, using default targets: {self.use_default_targets}")
        except Exception as e:
            logger.error(f"Error initializing CSV metadata: {e}")
            self.column_indices = {}
            self.y_columns = [f"y_{i}" for i in range(1, NUM_FUTURE_STEPS+1)]
            self.use_default_targets = True
            self.default_target_count = NUM_FUTURE_STEPS
            self.total_rows = 0  # Will be corrected when reading chunks
    
    def _start_prefetch_thread(self):
        """Start a background thread to prefetch the next chunk of data"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return  # Thread already running
            
        import threading
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_chunk,
            daemon=True
        )
        self.prefetch_thread.start()
        
    def _prefetch_chunk(self):
        """Background thread function to load the next chunk of data"""
        try:
            with self.prefetch_lock:
                # If we already have chunk data loaded, don't reload
                if self.chunk_data is not None:
                    return
                    
                import pandas as pd
                
                # Calculate chunk bounds
                chunk_start = self.next_chunk_idx
                chunk_end = min(chunk_start + self.chunksize, self.total_rows)
                
                # Skip header and read only the rows we need
                skiprows = [i for i in range(1, chunk_start + 1)]  # +1 to skip header
                nrows = chunk_end - chunk_start
                
                # Read chunk
                df_chunk = pd.read_csv(
                    self.csv_file,
                    skiprows=skiprows,
                    nrows=nrows,
                    engine='c',  # Use faster C engine
                    low_memory=False,  # Avoid dtype warnings
                    usecols=lambda x: x in self.column_indices.keys()  # Only load needed columns
                )
                
                # Update next chunk index
                self.next_chunk_idx = chunk_end
                if self.next_chunk_idx >= self.total_rows:
                    self.next_chunk_idx = 0  # Loop back to start
                    
                # Store chunk data
                self.chunk_data = df_chunk
                self.chunks_loaded = True
                
                logger.info(f"Prefetched chunk {chunk_start}-{chunk_end} ({len(df_chunk)} rows)")
        except Exception as e:
            logger.error(f"Error prefetching chunk: {e}")
            self.chunk_data = None
            
    def _get_row_by_index(self, idx):
        """Get a single row from the CSV file by its index (1-based)"""
        try:
            import pandas as pd
            
            # If idx is outside our current chunk, load the appropriate chunk
            if not self.chunks_loaded or self.chunk_data is None or not self._is_index_in_current_chunk(idx):
                # Calculate which chunk contains this index
                chunk_idx = ((idx - 1) // self.chunksize) * self.chunksize
                
                # Skip header and read only the rows we need
                skiprows = [i for i in range(1, chunk_idx + 1)]  # +1 to skip header
                nrows = self.chunksize
                
                # Read chunk
                df_chunk = pd.read_csv(
                    self.csv_file,
                    skiprows=skiprows,
                    nrows=nrows,
                    engine='c',  # Use faster C engine
                    low_memory=False,  # Avoid dtype warnings
                    usecols=lambda x: x in self.column_indices.keys()  # Only load needed columns
                )
                
                self.chunk_data = df_chunk
                self.chunks_loaded = True
                self.next_chunk_idx = chunk_idx + self.chunksize
                
                # Start prefetching next chunk
                if self.prefetch_thread is not None:
                    self._start_prefetch_thread()
            
            # Get the row from the chunk
            rel_idx = (idx - 1) % self.chunksize
            if rel_idx >= len(self.chunk_data):
                logger.error(f"Index {idx} (relative {rel_idx}) out of bounds for chunk with {len(self.chunk_data)} rows")
                return None
                
            row = self.chunk_data.iloc[rel_idx]
            return row
            
        except Exception as e:
            logger.error(f"Error getting row by index {idx}: {e}")
            return None
            
    def _is_index_in_current_chunk(self, idx):
        """Check if the given index is in the current chunk"""
        if self.chunk_data is None:
            return False
            
        # Calculate which chunk should contain this index
        chunk_idx = ((idx - 1) // self.chunksize) * self.chunksize
        next_chunk_idx = chunk_idx + self.chunksize
        
        # Check if current chunk start matches the expected chunk start
        return (self.next_chunk_idx - self.chunksize) == chunk_idx
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return max(1, math.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, idx):
        """Get a batch of data with caching for efficiency"""
        # Check if batch is in cache
        if idx in self.cache:
            return self.cache[idx]
            
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Handle empty batches - return small valid batch instead of failing
        if not batch_indices:
            logger.warning(f"Empty batch requested at index {idx}. Creating small dummy batch.")
            # Create a minimal valid batch with the right shapes
            return self._create_dummy_batch()
        
        # Initialize batch arrays
        batch_5m = []
        batch_15m = []
        batch_1h = []
        batch_google = []
        batch_santiment = []
        batch_ta = []
        batch_ctx = []
        batch_y = []
        
        # Track memory usage before loading
        start_mem = 0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            start_mem = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
            
        # Process each row in the batch
        for i in batch_indices:
            try:
                # Get row by index using our efficient method
                row = self._get_row_by_index(i)
                
                if row is None:
                    continue
                    
                # Parse row data using optimized JSON handling
                try:
                    import json
                    from json import loads as json_loads
                    
                    # Parse arrays more efficiently
                    arr_5m = np.asarray(json_loads(row["arr_5m"])[0], dtype=np.float32)
                    arr_15m = np.asarray(json_loads(row["arr_15m"])[0], dtype=np.float32)
                    arr_1h = np.asarray(json_loads(row["arr_1h"])[0], dtype=np.float32)
                    arr_google_trend = np.asarray(json_loads(row["arr_google_trend"])[0], dtype=np.float32)
                    arr_santiment = np.asarray(json_loads(row["arr_santiment"])[0], dtype=np.float32)
                    arr_ta = np.asarray(json_loads(row["arr_ta_63"])[0], dtype=np.float32)
                    arr_ctx = np.asarray(json_loads(row["arr_ctx_11"])[0], dtype=np.float32)
                    
                    # Parse target values
                    targets = []
                    if not self.use_default_targets:
                        for col in self.y_columns:
                            targets.append(float(row[col]))
                    else:
                        # Use default target values (all zeros)
                        targets = [0.0] * self.default_target_count
                    
                    # Fill in missing targets if needed
                    while len(targets) < NUM_FUTURE_STEPS:
                        targets.append(0.0)
                    
                    # Apply scaling if available
                    if self.scaler:
                        arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx = prepare_for_model_inputs(
                            arr_5m, arr_15m, arr_1h,
                            arr_google_trend, arr_santiment,
                            arr_ta, arr_ctx,
                            self.scaler
                        )
                    
                    # Reshape time series data efficiently
                    arr_5m = arr_5m.reshape(1, arr_5m.shape[0], arr_5m.shape[1])
                    arr_15m = arr_15m.reshape(1, arr_15m.shape[0], arr_15m.shape[1])
                    arr_1h = arr_1h.reshape(1, arr_1h.shape[0], arr_1h.shape[1])
                    arr_google_trend = arr_google_trend.reshape(1, arr_google_trend.shape[0], 1)
                    
                    # Add to batch
                    batch_5m.append(arr_5m)
                    batch_15m.append(arr_15m)
                    batch_1h.append(arr_1h)
                    batch_google.append(arr_google_trend)
                    batch_santiment.append(arr_santiment)
                    batch_ta.append(arr_ta)
                    batch_ctx.append(arr_ctx)
                    batch_y.append(targets)
                    
                    # Memory optimizations: Remove references to variables we no longer need
                    del arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx, targets
                    
                except Exception as e:
                    logger.error(f"Error parsing row data: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing row index {i}: {e}")
                continue
        
        # Handle case where all rows failed to parse
        if not batch_5m:
            logger.warning(f"All rows in batch {idx} failed to parse. Creating a dummy batch.")
            return self._create_dummy_batch()
        
        # Convert to numpy arrays efficiently
        try:
            # Concatenate along batch dimension
            X_5m = np.concatenate(batch_5m, axis=0) if batch_5m else np.empty((0, 241, 9))
            X_15m = np.concatenate(batch_15m, axis=0) if batch_15m else np.empty((0, 241, 9))
            X_1h = np.concatenate(batch_1h, axis=0) if batch_1h else np.empty((0, 241, 9))
            X_google = np.concatenate(batch_google, axis=0) if batch_google else np.empty((0, 24, 1))
            X_santiment = np.array(batch_santiment) if batch_santiment else np.empty((0, 12))
            X_ta = np.array(batch_ta) if batch_ta else np.empty((0, 63))
            X_ctx = np.array(batch_ctx) if batch_ctx else np.empty((0, 11))
            Y = np.array(batch_y) if batch_y else np.empty((0, NUM_FUTURE_STEPS))
            
            # Clear references to batch arrays to free memory
            del batch_5m, batch_15m, batch_1h, batch_google, batch_santiment, batch_ta, batch_ctx, batch_y
            
            # Create result batch
            result = ([X_5m, X_15m, X_1h, X_google, X_santiment, X_ta, X_ctx], Y)
            
            # Store in cache if cache is enabled
            if self.cache_size > 0:
                # Manage cache size
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                # Add to cache
                self.cache[idx] = result
            
            # Track memory usage after loading
            end_mem = 0
            try:
                import psutil
                process = psutil.Process(os.getpid())
                end_mem = process.memory_info().rss / 1024 / 1024  # MB
                if (end_mem - start_mem) > 500:  # If memory increased by more than 500MB
                    logger.warning(f"Large memory increase processing batch {idx}: {end_mem - start_mem:.2f} MB")
            except ImportError:
                pass
                
            # Force garbage collection if memory usage increased significantly
            if (end_mem - start_mem) > 500:
                gc.collect()
            
            return result
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            # Return fallback batch
            return self._create_dummy_batch()
    
    def _create_dummy_batch(self):
        """Create a minimal valid batch with the correct shapes"""
        batch_size = 1
        X_5m = np.zeros((batch_size, 241, 9), dtype=np.float32)
        X_15m = np.zeros((batch_size, 241, 9), dtype=np.float32)
        X_1h = np.zeros((batch_size, 241, 9), dtype=np.float32)
        X_google = np.zeros((batch_size, 24, 1), dtype=np.float32)
        X_santiment = np.zeros((batch_size, 12), dtype=np.float32)
        X_ta = np.zeros((batch_size, 63), dtype=np.float32)
        X_ctx = np.zeros((batch_size, 11), dtype=np.float32)
        Y = np.zeros((batch_size, NUM_FUTURE_STEPS), dtype=np.float32)
        
        return [X_5m, X_15m, X_1h, X_google, X_santiment, X_ta, X_ctx], Y
                
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
            
            # Clear cache when shuffling
            self.cache.clear()
            
            # Reset chunk loading
            self.chunks_loaded = False
            self.chunk_data = None
            self.next_chunk_idx = 0
            
            # Force garbage collection
            gc.collect()
            
            # Restart prefetching
            if self.prefetch_thread is not None:
                self._start_prefetch_thread()


class Trainer:
    def __init__(
        self,
        training_csv=LSTM_DATA_FILE,
        model_out="models/advanced_lstm_model.keras",
        rl_transitions_csv=RL_TRANSITIONS_FILE,
        rl_model_out="models/rl_DQNAgent.weights.h5",
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
        batch_size=4,  # Smaller batch size for massive models
        rl_batch_size=64,
        apply_scaling=True,
        train_ratio=0.8,
        val_ratio=0.2,
        skip_lstm=False,
        skip_rl=False,
        max_rows=0,
        grad_accum=True,
        accum_steps=8,  # More accumulation steps for massive models
        model_size="massive",  # New default is massive
        reduce_precision=True,
        rl_epochs=10,
        cache_batches=16,  # Cache some batches for faster training
        use_mmap=True,  # Use memory-mapped files for large datasets
        lazy_loading=True,  # Enable lazy loading for large datasets
        chunk_size=10000,  # Chunk size for CSV reading
        prefetch_batches=2,  # Number of batches to prefetch
        monitor_memory=True,  # Enable memory usage monitoring
        tf_allow_growth=True,  # Allow TF to grow memory usage as needed
        aggressive_gc=True  # Enable aggressive garbage collection
    ):
        """
        Ultra-optimized Trainer for massive models (500MB+) and large datasets (6GB+)
        """
        self.logger = logger
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        os.makedirs("logs_training", exist_ok=True)
        
        self.training_csv = training_csv
        self.model_out = model_out
        self.rl_transitions_csv = rl_transitions_csv
        self.rl_model_out = rl_model_out
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
        self.rl_batch_size = rl_batch_size
        self.apply_scaling = apply_scaling
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.skip_lstm = skip_lstm
        self.skip_rl = skip_rl
        self.max_rows = max_rows
        self.grad_accum = grad_accum
        self.accum_steps = accum_steps
        self.model_size = model_size
        self.reduce_precision = reduce_precision
        self.rl_epochs = rl_epochs
        self.cache_batches = cache_batches
        
        # New memory optimization parameters
        self.use_mmap = use_mmap
        self.lazy_loading = lazy_loading
        self.chunk_size = chunk_size
        self.prefetch_batches = prefetch_batches
        self.monitor_memory = monitor_memory
        self.tf_allow_growth = tf_allow_growth
        self.aggressive_gc = aggressive_gc
        
        # Set base units based on model size
        base_units_map = {
            "tiny": 32,
            "small": 48,
            "medium": 64,
            "large": 96,
            "xlarge": 128,
            "massive": 512,  # New massive size
            "gigantic": 1024  # Ultra massive size
        }
        self.base_units = base_units_map.get(model_size, 512)
        
        # Set depth based on model size
        depth_map = {
            "tiny": 2,
            "small": 2,
            "medium": 3,
            "large": 4,
            "xlarge": 5,
            "massive": 6,
            "gigantic": 8  
        }
        self.depth = depth_map.get(model_size, 6)

        # Initialize model scaler
        self.model_scaler = None
        if self.apply_scaling:
            try:
                self.model_scaler = ModelScaler.load("models/scalers.pkl")
                self.logger.info("Loaded model scalers from models/scalers.pkl.")
            except:
                self.logger.warning("No scalers found. Will fit new scalers.")
                self.model_scaler = ModelScaler()

        # Verify training data file exists
        if not os.path.exists(self.training_csv):
            self.logger.error(f"Training data file not found: {self.training_csv}")
            print(f"ERROR: Training data file not found: {self.training_csv}")
        else:
            self.logger.info(f"Training data file found: {self.training_csv}")
            
            # Check file size for large files
            try:
                file_size_bytes = os.path.getsize(self.training_csv)
                file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
                if file_size_gb > 5.0:
                    self.logger.info(f"Large training file detected: {file_size_gb:.2f} GB")
                    print(f"Large training file detected: {file_size_gb:.2f} GB")
                    
                    # Auto-enable memory optimizations for very large files
                    if file_size_gb > 6.0:
                        if not self.use_mmap:
                            self.logger.info("Auto-enabling memory-mapped file support for large CSV")
                            self.use_mmap = True
                        if not self.lazy_loading:
                            self.logger.info("Auto-enabling lazy loading for large CSV")
                            self.lazy_loading = True
                        if not self.aggressive_gc:
                            self.logger.info("Auto-enabling aggressive garbage collection for large CSV")
                            self.aggressive_gc = True
            except Exception as e:
                self.logger.warning(f"Error checking file size: {e}")
                
        # Configure TensorFlow memory growth if requested
        if self.tf_allow_growth:
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info("TensorFlow memory growth enabled")
            except Exception as e:
                self.logger.warning(f"Failed to enable TensorFlow memory growth: {e}")
                
        # Set up aggressive garbage collection if requested
        if self.aggressive_gc:
            try:
                import gc
                gc.enable()
                gc.set_threshold(100, 5, 5)  # More aggressive thresholds
                self.logger.info("Aggressive garbage collection enabled")
            except Exception as e:
                self.logger.warning(f"Failed to set up aggressive garbage collection: {e}")
                
        # Set up memory monitoring if requested
        if self.monitor_memory:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
                print(f"Initial memory usage: {initial_memory:.2f} MB")
            except ImportError:
                self.logger.warning("psutil not available, memory monitoring disabled")
                self.monitor_memory = False
            
        # Initialize model if not skipping LSTM training
        if not skip_lstm:
            self.logger.info(f"Initializing MASSIVE model with {NUM_FUTURE_STEPS} outputs, {self.base_units} units, and depth {self.depth}")
            try:
                # Enable mixed precision if requested
                if reduce_precision:
                    try:
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        self.logger.info(f"Mixed precision enabled with policy: {policy.name}")
                    except:
                        self.logger.warning("Mixed precision not available, using default precision")
                        
                # Create model - add massive_model=True to enable the new architecture
                self.model = load_advanced_lstm_model(
                    model_5m_window=self.window_5m,
                    model_15m_window=self.window_15m,
                    model_1h_window=self.window_1h,
                    feature_dim=self.feature_5m,
                    santiment_dim=self.santiment_dim,
                    ta_dim=self.ta_dim,
                    signal_dim=self.signal_dim,
                    base_units=self.base_units,
                    depth=self.depth,
                    memory_efficient=True,
                    gradient_accumulation=self.grad_accum,
                    gradient_accumulation_steps=self.accum_steps,
                    mixed_precision=self.reduce_precision,
                    massive_model=True  # Enable massive model architecture
                )
                
                self.logger.info(f"Model created with {self.model.count_params():,} parameters")
                # Calculate approximate model size in MB
                model_size_mb = self.model.count_params() * 4 / (1024 * 1024)
                self.logger.info(f"Approximate model size: {model_size_mb:.2f} MB")
                
                # Enable fallback to CPU implementation for operations that don't have GPU optimizations
                try:
                    tf.config.experimental.set_synchronous_execution(False)
                    self.logger.info("Asynchronous execution enabled")
                except:
                    self.logger.warning("Failed to enable asynchronous execution")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize model: {e}")
                print(f"ERROR: Failed to initialize model: {e}")
                import traceback
                print(traceback.format_exc())
                raise
        else:
            self.logger.info("Skipping model initialization (skip_lstm=True).")
            self.model = None
            
        # Initialize DQN agent if not skipping RL training
        if not skip_rl:
            self.dqn_agent = DQNAgent(
                state_dim=NUM_FUTURE_STEPS + 3,  # 10 signals + atr + btc_frac + eur_frac
                batch_size=self.rl_batch_size
            )
        else:
            self.dqn_agent = None
            
        # Log memory status after initialization
        if self.monitor_memory:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.logger.info(f"Memory usage after initialization: {current_memory:.2f} MB")
                print(f"Memory usage after initialization: {current_memory:.2f} MB")
            except:
                pass

    def fit_scalers(self, gen):
        """Fit scalers on a subset of training data"""
        try:
            # Get first batch for fitting scalers
            X, _ = gen[0]
            X_5m, X_15m, X_1h, X_google, X_santiment, X_ta, X_ctx = X
            
            # Initialize scaler if needed
            if self.model_scaler is None:
                self.model_scaler = ModelScaler()
                
            # Fit all scalers
            self.model_scaler.fit_all(
                X_5m, X_15m, X_1h,
                X_google, X_santiment,
                X_ta, X_ctx
            )
            
            # Save scalers
            try:
                self.model_scaler.save("models/scalers.pkl")
                self.logger.info("Scalers fitted and saved to models/scalers.pkl")
            except Exception as e:
                self.logger.error(f"Error saving scalers: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error fitting scalers: {e}")
            return False

    def custom_gradient_accumulation_training(self, train_gen, val_gen=None):
        """
        Memory-efficient custom training function with gradient accumulation
        optimized for massive models and large CSV files
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
        
        if val_gen:
            val_loss = tf.keras.metrics.Mean()
            val_mae = tf.keras.metrics.MeanAbsoluteError()
        
        # TensorBoard logging
        current_time = int(time.time())
        log_dir = f'logs_training/gradient_accum_{current_time}'
        summary_writer = tf.summary.create_file_writer(log_dir)
        
        steps_per_epoch = len(train_gen)
        
        self.logger.info(f"Starting training with gradient accumulation")
        print(f"Starting training with gradient accumulation")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        print(f"Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"Batch size: {self.batch_size}, Accumulation steps: {self.accum_steps}")
        print(f"Batch size: {self.batch_size}, Accumulation steps: {self.accum_steps}")
        
        # Enable memory tracking if available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            enable_memory_tracking = True
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            print(f"Initial memory usage: {initial_memory:.2f} MB")
        except ImportError:
            enable_memory_tracking = False
            
        # CSV file validation before starting training
        self._validate_training_csv(train_gen.csv_file)
        
        # Track total training time
        total_training_start = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Reset metrics - using try/except for TF version compatibility
            try:
                train_loss.reset_states()
                train_mae.reset_states()
            except AttributeError:
                try:
                    # TF 2.18+ uses reset() instead of reset_states()
                    train_loss.reset()
                    train_mae.reset()
                except AttributeError:
                    # Last resort: create new metric instances
                    train_loss = tf.keras.metrics.Mean()
                    train_mae = tf.keras.metrics.MeanAbsoluteError()
            
            # Accumulated gradients
            accumulated_gradients = None
            accumulation_count = 0
            
            # Track error counts
            error_count = 0
            max_errors = 10  # Maximum number of errors before warning
            skip_steps = []
            
            # Track memory usage spike
            peak_memory = 0 if enable_memory_tracking else None
            
            # Process batches
            for step in range(steps_per_epoch):
                # Skip steps with previous errors
                if step in skip_steps:
                    continue
                    
                step_start_time = time.time()
                
                try:
                    # Get batch
                    X_batch, y_batch = train_gen[step]
                    
                    # Skip empty batches
                    if X_batch[0].shape[0] == 0:
                        continue
                    
                    # Check for NaN values in batch
                    nan_detected = False
                    for i, arr in enumerate(X_batch):
                        if np.isnan(arr).any():
                            self.logger.warning(f"NaN values detected in input {i} of batch {step}. Fixing...")
                            X_batch[i] = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
                            nan_detected = True
                    
                    if np.isnan(y_batch).any():
                        self.logger.warning(f"NaN values detected in target of batch {step}. Fixing...")
                        y_batch = np.nan_to_num(y_batch, nan=0.0)
                        nan_detected = True
                        
                    if nan_detected:
                        self.logger.info(f"Fixed NaN values in batch {step}")
                    
                    # Convert to tensors more efficiently - with try/except for better error handling
                    try:
                        X_5m_batch = tf.convert_to_tensor(X_batch[0], dtype=tf.float32)
                        X_15m_batch = tf.convert_to_tensor(X_batch[1], dtype=tf.float32)
                        X_1h_batch = tf.convert_to_tensor(X_batch[2], dtype=tf.float32)
                        X_gt_batch = tf.convert_to_tensor(X_batch[3], dtype=tf.float32)
                        X_sa_batch = tf.convert_to_tensor(X_batch[4], dtype=tf.float32)
                        X_ta_batch = tf.convert_to_tensor(X_batch[5], dtype=tf.float32)
                        X_ctx_batch = tf.convert_to_tensor(X_batch[6], dtype=tf.float32)
                        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
                    except Exception as e:
                        self.logger.error(f"Error converting tensors in batch {step}: {e}")
                        error_count += 1
                        skip_steps.append(step)
                        continue
                    
                    data_prep_time = time.time() - step_start_time
                    
                    # Initialize accumulated gradients if needed
                    if accumulated_gradients is None:
                        try:
                            accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
                        except Exception as e:
                            self.logger.error(f"Error initializing accumulated gradients: {e}")
                            error_count += 1
                            continue
                    
                    # Forward pass and calculate gradients - with memory tracking
                    if enable_memory_tracking:
                        before_forward = process.memory_info().rss / 1024 / 1024  # MB
                        
                    forward_start = time.time()
                    
                    try:
                        with tf.GradientTape() as tape:
                            y_pred = self.model(
                                [X_5m_batch, X_15m_batch, X_1h_batch, X_gt_batch, X_sa_batch, X_ta_batch, X_ctx_batch], 
                                training=True
                            )
                            batch_loss = loss_fn(y_batch, y_pred)
                    except Exception as e:
                        self.logger.error(f"Error in forward pass for batch {step}: {e}")
                        error_count += 1
                        
                        # Clear references to tensors
                        del X_5m_batch, X_15m_batch, X_1h_batch, X_gt_batch, X_sa_batch, X_ta_batch, X_ctx_batch
                        del y_batch
                        
                        # Force garbage collection
                        gc.collect()
                        continue
                        
                    forward_time = time.time() - forward_start
                    
                    if enable_memory_tracking:
                        after_forward = process.memory_info().rss / 1024 / 1024  # MB
                        forward_memory_increase = after_forward - before_forward
                        if forward_memory_increase > 500:  # More than 500MB increase
                            self.logger.warning(f"Large memory increase during forward pass: {forward_memory_increase:.2f} MB")
                    
                    # Calculate and accumulate gradients - with memory tracking
                    if enable_memory_tracking:
                        before_gradient = process.memory_info().rss / 1024 / 1024  # MB
                        
                    gradient_start = time.time()
                    
                    try:
                        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
                    except Exception as e:
                        self.logger.error(f"Error calculating gradients for batch {step}: {e}")
                        error_count += 1
                        
                        # Clear references to tensors
                        del X_5m_batch, X_15m_batch, X_1h_batch, X_gt_batch, X_sa_batch, X_ta_batch, X_ctx_batch
                        del y_batch, y_pred, tape
                        
                        # Force garbage collection
                        gc.collect()
                        continue
                    
                    # Check for NaN gradients and clip them
                    has_nans = False
                    for i, grad in enumerate(gradients):
                        if grad is not None:
                            if tf.reduce_any(tf.math.is_nan(grad)):
                                has_nans = True
                                gradients[i] = tf.zeros_like(grad)
                            else:
                                # Clip gradients to prevent explosion
                                gradients[i] = tf.clip_by_norm(grad, 1.0)
                    
                    if has_nans:
                        self.logger.warning(f"NaN gradients detected in step {step+1}, zeroing them out")
                    
                    # Accumulate gradients
                    try:
                        for i, grad in enumerate(gradients):
                            if grad is not None:
                                accumulated_gradients[i] += grad
                    except Exception as e:
                        self.logger.error(f"Error accumulating gradients for batch {step}: {e}")
                        error_count += 1
                        
                        # Skip gradient accumulation but continue training
                        
                    gradient_time = time.time() - gradient_start
                    
                    if enable_memory_tracking:
                        after_gradient = process.memory_info().rss / 1024 / 1024  # MB
                        gradient_memory_increase = after_gradient - before_gradient
                        if gradient_memory_increase > 500:  # More than 500MB increase
                            self.logger.warning(f"Large memory increase during gradient calculation: {gradient_memory_increase:.2f} MB")
                    
                    # Update metrics
                    train_loss.update_state(batch_loss)
                    train_mae.update_state(y_batch, y_pred)
                    
                    # Free memory - explicitly delete tensors
                    del X_5m_batch, X_15m_batch, X_1h_batch, X_gt_batch, X_sa_batch, X_ta_batch, X_ctx_batch
                    del y_batch, y_pred, gradients, tape
                    
                    # Increment accumulation counter
                    accumulation_count += 1
                    
                    # Apply accumulated gradients when needed
                    is_last_batch = (step == steps_per_epoch - 1)
                    if accumulation_count >= self.accum_steps or is_last_batch:
                        # Normalize gradients by the number of accumulation steps
                        try:
                            for i in range(len(accumulated_gradients)):
                                if accumulated_gradients[i] is not None and accumulation_count > 0:
                                    accumulated_gradients[i] = accumulated_gradients[i] / accumulation_count
                            
                            # Apply gradients
                            apply_start = time.time()
                            optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
                            apply_time = time.time() - apply_start
                            
                            # Reset accumulation
                            accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
                            accumulation_count = 0
                        except Exception as e:
                            self.logger.error(f"Error applying accumulated gradients: {e}")
                            # Reset accumulation and continue
                            accumulated_gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
                            accumulation_count = 0
                        
                        # Force garbage collection
                        gc.collect()
                    
                    # Calculate step time
                    step_time = time.time() - step_start_time
                    
                    # Track peak memory usage
                    if enable_memory_tracking:
                        current_mem = process.memory_info().rss / 1024 / 1024  # MB
                        peak_memory = max(peak_memory, current_mem)
                    
                    # Log timing details periodically
                    if (step + 1) % 10 == 0:
                        self.logger.info(
                            f"  Step {step+1}/{steps_per_epoch} - "
                            f"Loss: {train_loss.result():.4f} - "
                            f"Time: {step_time:.2f}s (data: {data_prep_time:.2f}s, "
                            f"forward: {forward_time:.2f}s, grad: {gradient_time:.2f}s)"
                        )
                        
                        # Log memory usage periodically
                        if enable_memory_tracking:
                            current_mem = process.memory_info().rss / 1024 / 1024  # MB
                            self.logger.info(f"  Memory usage: {current_mem:.2f} MB, Peak: {peak_memory:.2f} MB")
                            
                            # If memory usage is too high, trigger garbage collection
                            if current_mem > peak_memory * 0.9:  # If we're using 90% of peak
                                self.logger.info("  Triggering garbage collection to reduce memory usage")
                                gc.collect()
                                
                except Exception as e:
                    self.logger.error(f"Error in training step {step+1}: {e}")
                    print(f"Error in training step {step+1}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    error_count += 1
                    
                    # Force garbage collection
                    gc.collect()
                    continue
                    
            # Print error summary if errors occurred
            if error_count > 0:
                self.logger.warning(f"Encountered {error_count} errors during epoch {epoch+1}")
                print(f"Encountered {error_count} errors during epoch {epoch+1}")
            
            # Validation if available
            if val_gen:
                try:
                    # Reset validation metrics
                    try:
                        val_loss.reset_states()
                        val_mae.reset_states()
                    except AttributeError:
                        try:
                            # TF 2.18+ uses reset() instead of reset_states()
                            val_loss.reset()
                            val_mae.reset()
                        except AttributeError:
                            # Last resort: create new metric instances
                            val_loss = tf.keras.metrics.Mean()
                            val_mae = tf.keras.metrics.MeanAbsoluteError()
                    
                    # Process validation batches
                    self.logger.info(f"Running validation ({len(val_gen)} batches)...")
                    
                    # Track validation errors
                    val_error_count = 0
                    
                    for val_step in range(len(val_gen)):
                        try:
                            # Get validation batch
                            X_val_batch, y_val_batch = val_gen[val_step]
                            
                            # Skip empty batches
                            if X_val_batch[0].shape[0] == 0:
                                continue
                                
                            # Fix NaN values
                            for i, arr in enumerate(X_val_batch):
                                if np.isnan(arr).any():
                                    X_val_batch[i] = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
                                    
                            if np.isnan(y_val_batch).any():
                                y_val_batch = np.nan_to_num(y_val_batch, nan=0.0)
                                
                            # Convert to tensors
                            X_5m_val = tf.convert_to_tensor(X_val_batch[0], dtype=tf.float32)
                            X_15m_val = tf.convert_to_tensor(X_val_batch[1], dtype=tf.float32)
                            X_1h_val = tf.convert_to_tensor(X_val_batch[2], dtype=tf.float32)
                            X_gt_val = tf.convert_to_tensor(X_val_batch[3], dtype=tf.float32)
                            X_sa_val = tf.convert_to_tensor(X_val_batch[4], dtype=tf.float32)
                            X_ta_val = tf.convert_to_tensor(X_val_batch[5], dtype=tf.float32)
                            X_ctx_val = tf.convert_to_tensor(X_val_batch[6], dtype=tf.float32)
                            y_val = tf.convert_to_tensor(y_val_batch, dtype=tf.float32)
                            
                            # Forward pass with try/except for robustness
                            try:
                                y_pred = self.model(
                                    [X_5m_val, X_15m_val, X_1h_val, X_gt_val, 
                                    X_sa_val, X_ta_val, X_ctx_val], 
                                    training=False
                                )
                                
                                # Calculate loss and update metrics
                                val_batch_loss = loss_fn(y_val, y_pred)
                                val_loss.update_state(val_batch_loss)
                                val_mae.update_state(y_val, y_pred)
                            except Exception as e:
                                self.logger.error(f"Error in validation forward pass, step {val_step+1}: {e}")
                                val_error_count += 1
                                continue
                            
                            # Free memory
                            del X_5m_val, X_15m_val, X_1h_val, X_gt_val
                            del X_sa_val, X_ta_val, X_ctx_val, y_val, y_pred
                            
                        except Exception as e:
                            self.logger.error(f"Error in validation step {val_step+1}: {e}")
                            val_error_count += 1
                            continue
                            
                        # Log progress
                        if (val_step + 1) % 10 == 0:
                            self.logger.info(f"  Val Step {val_step+1}/{len(val_gen)}")
                            
                    # Print validation error summary if errors occurred
                    if val_error_count > 0:
                        self.logger.warning(f"Encountered {val_error_count} errors during validation for epoch {epoch+1}")
                    
                    # Log metrics
                    with summary_writer.as_default():
                        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                        tf.summary.scalar('train_mae', train_mae.result(), step=epoch)
                        tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
                        tf.summary.scalar('val_mae', val_mae.result(), step=epoch)
                        
                        # Log memory usage if tracking enabled
                        if enable_memory_tracking:
                            current_mem = process.memory_info().rss / 1024 / 1024  # MB
                            tf.summary.scalar('memory_usage_mb', current_mem, step=epoch)
                            tf.summary.scalar('peak_memory_mb', peak_memory, step=epoch)
                    
                    # Check for improvement
                    current_val_loss = val_loss.result()
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        # Save best model
                        try:
                            self.model.save(self.model_out)
                            self.logger.info(f"  Saved best model with val_loss: {best_val_loss:.4f}")
                            print(f"  Saved best model with val_loss: {best_val_loss:.4f}")
                        except Exception as e:
                            self.logger.error(f"Error saving best model: {e}")
                            print(f"  Error saving best model: {e}")
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
                    
                    # Log memory usage if tracking enabled
                    if enable_memory_tracking:
                        current_mem = process.memory_info().rss / 1024 / 1024  # MB
                        tf.summary.scalar('memory_usage_mb', current_mem, step=epoch)
                        tf.summary.scalar('peak_memory_mb', peak_memory, step=epoch)
                
                # Save checkpoint models periodically (every 5 epochs for massive models)
                if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                    # Use checkpoint directory to save multiple versions
                    try:
                        checkpoint_dir = f"models/checkpoints/epoch_{epoch+1}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(checkpoint_dir, "model.keras")
                        self.model.save(checkpoint_path)
                        self.logger.info(f"  Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
                        print(f"  Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {e}")
                        print(f"  Error saving checkpoint: {e}")
                    
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
                
                # Log memory usage if tracking enabled
                if enable_memory_tracking:
                    current_mem = process.memory_info().rss / 1024 / 1024  # MB
                    mem_increase = current_mem - initial_memory
                    self.logger.info(f"  Memory usage: {current_mem:.2f} MB (increase: {mem_increase:.2f} MB)")
                    print(f"  Memory usage: {current_mem:.2f} MB (increase: {mem_increase:.2f} MB)")
                
            # Force garbage collection between epochs
            gc.collect()
            
            # Sleep briefly to allow memory to stabilize
            time.sleep(1)
        
        # Total training time
        total_training_time = time.time() - total_training_start
        hours = total_training_time // 3600
        minutes = (total_training_time % 3600) // 60
        seconds = total_training_time % 60
        
        self.logger.info(f"Total training time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        print(f"Total training time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        
        # Save final model if not already saved
        if not val_gen or patience_counter < self.early_stop_patience:
            try:
                final_path = self.model_out
                self.model.save(final_path)
                self.logger.info(f"Saved final model to {final_path}")
                print(f"Saved final model to {final_path}")
                
                # Also save to a timestamp-based file to keep a record
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                timestamp_path = f"models/advanced_lstm_model_{timestamp}.keras"
                self.model.save(timestamp_path)
                self.logger.info(f"Saved timestamped model to {timestamp_path}")
                print(f"Saved timestamped model to {timestamp_path}")
            except Exception as e:
                self.logger.error(f"Error saving final model: {e}")
                print(f"Error saving final model: {e}")
                
        # Final memory report if tracking enabled
        if enable_memory_tracking:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = final_memory - initial_memory
            self.logger.info(f"Final memory usage: {final_memory:.2f} MB (net increase: {mem_increase:.2f} MB)")
            self.logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
            print(f"Final memory usage: {final_memory:.2f} MB (net increase: {mem_increase:.2f} MB)")
            print(f"Peak memory usage: {peak_memory:.2f} MB")
            
    def _validate_training_csv(self, csv_file):
        """Validate the training CSV file before training"""
        try:
            import pandas as pd
            
            self.logger.info(f"Validating training CSV file: {csv_file}")
            print(f"Validating training CSV file: {csv_file}")
            
            # Check file size
            file_size_bytes = os.path.getsize(csv_file)
            file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
            
            self.logger.info(f"CSV file size: {file_size_gb:.2f} GB")
            print(f"CSV file size: {file_size_gb:.2f} GB")
            
            # Get file information
            import pandas as pd
            
            # Check if file can be opened
            try:
                # Read just the header and first row to verify structure
                df_sample = pd.read_csv(csv_file, nrows=1)
                num_columns = len(df_sample.columns)
                self.logger.info(f"CSV has {num_columns} columns")
                
                # Check required columns
                required_columns = ["arr_5m", "arr_15m", "arr_1h", "arr_google_trend", 
                                   "arr_santiment", "arr_ta_63", "arr_ctx_11"]
                
                missing = [col for col in required_columns if col not in df_sample.columns]
                if missing:
                    self.logger.error(f"Missing required columns in CSV: {missing}")
                    print(f"ERROR: Missing required columns in CSV: {missing}")
                else:
                    self.logger.info("All required columns found in CSV")
                    
                # Check target columns
                target_cols = [col for col in df_sample.columns if col.startswith("y_") and col[2:].isdigit()]
                self.logger.info(f"Found {len(target_cols)} target columns")
                
                # For large files, provide recommendations
                if file_size_gb > 5.0:
                    self.logger.info(f"Large CSV file detected ({file_size_gb:.2f} GB). Recommendations:")
                    self.logger.info("1. Consider reducing batch size and increasing gradient accumulation steps")
                    self.logger.info("2. Enable checkpointing every few epochs to prevent data loss")
                    self.logger.info("3. Monitor memory usage during training")
                    print(f"Large CSV file detected ({file_size_gb:.2f} GB). Performance optimizations will be used.")
                
                # All checks passed
                self.logger.info("CSV validation successful")
                
            except Exception as e:
                self.logger.error(f"Error reading CSV file: {e}")
                print(f"WARNING: Error validating CSV file: {e}")
                
        except Exception as e:
            self.logger.error(f"Error validating CSV file: {e}")
            print(f"WARNING: Error validating CSV file: {e}")
            # Continue with training despite validation failure

    def train_lstm(self):
        """
        Train the multi-output LSTM with optimizations for massive models
        and large datasets (6.4GB+)
        """
        self.logger.info("Starting MASSIVE LSTM training with memory optimizations...")
        print("Starting MASSIVE LSTM training with memory optimizations...")
        
        # Validate training data first
        if not os.path.exists(self.training_csv):
            self.logger.error(f"Training data file not found: {self.training_csv}")
            print(f"ERROR: Training data file not found: {self.training_csv}")
            return
            
        if os.path.getsize(self.training_csv) == 0:
            self.logger.error(f"Training data file is empty: {self.training_csv}")
            print(f"ERROR: Training data file is empty: {self.training_csv}")
            return
        
        # Get file size for memory optimization tuning
        file_size_bytes = os.path.getsize(self.training_csv)
        file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
        self.logger.info(f"Training data file size: {file_size_gb:.2f} GB")
        print(f"Training data file size: {file_size_gb:.2f} GB")
        
        # Adjust batch size and accumulation steps for very large files
        if file_size_gb > 6.0 and self.batch_size > 2:
            old_batch_size = self.batch_size
            self.batch_size = 2
            self.logger.info(f"Auto-reducing batch size from {old_batch_size} to {self.batch_size} for large file")
            print(f"Auto-reducing batch size from {old_batch_size} to {self.batch_size} for large file")
            
            # Increase accumulation steps to compensate
            if self.grad_accum:
                old_accum_steps = self.accum_steps
                self.accum_steps = max(self.accum_steps, 4)
                if old_accum_steps != self.accum_steps:
                    self.logger.info(f"Auto-increasing accumulation steps from {old_accum_steps} to {self.accum_steps}")
                    print(f"Auto-increasing accumulation steps from {old_accum_steps} to {self.accum_steps}")
        
        # Validate CSV format first before creating generators
        try:
            with open(self.training_csv, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                required_columns = ["arr_5m", "arr_15m", "arr_1h", "arr_google_trend", 
                                   "arr_santiment", "arr_ta_63", "arr_ctx_11"]
                
                missing_cols = [col for col in required_columns if col not in header]
                if missing_cols:
                    self.logger.error(f"Training CSV is missing required columns: {missing_cols}")
                    print(f"ERROR: Training CSV is missing required columns: {missing_cols}")
                    return
                
                # Check for target columns (y_1, y_2, etc.)
                target_cols = [col for col in header if col.startswith("y_") and col[2:].isdigit()]
                if not target_cols and NUM_FUTURE_STEPS > 0:
                    self.logger.warning(f"No target columns (y_1, y_2, etc.) found in CSV. Using default zero targets.")
                    print(f"WARNING: No target columns found in CSV. Using default zero targets.")
                
                # Try to read first data row
                try:
                    first_row = next(reader)
                    if len(first_row) != len(header):
                        self.logger.error(f"CSV format error: Header has {len(header)} columns but data row has {len(first_row)} columns")
                        print(f"ERROR: CSV format error: Header/data column count mismatch")
                        return
                except StopIteration:
                    self.logger.error("CSV contains header but no data rows")
                    print("ERROR: CSV contains header but no data rows")
                    return
                
        except Exception as e:
            self.logger.error(f"Error validating CSV format: {e}")
            print(f"ERROR: Error validating CSV format: {e}")
            return
            
        # Force garbage collection before creating generators
        if self.aggressive_gc:
            gc.collect()
            self.logger.info("Performed garbage collection before creating data generators")
        
        # Create optimized data generators with appropriate settings for large files
        self.logger.info("Creating memory-efficient data generators...")
        print("Creating memory-efficient data generators...")
        train_gen = MemoryEfficientGenerator(
            csv_file=self.training_csv,
            batch_size=self.batch_size,
            shuffle=True,
            validation=False,
            val_split=self.val_ratio,
            max_rows=self.max_rows,
            scaler=None,  # Don't apply scaling yet
            cache_size=self.cache_batches,
            chunksize=self.chunk_size  # Use configurable chunk size for large files
        )
        
        val_gen = MemoryEfficientGenerator(
            csv_file=self.training_csv,
            batch_size=self.batch_size,
            shuffle=False,
            validation=True,
            val_split=self.val_ratio,
            max_rows=self.max_rows,
            scaler=None,  # Don't apply scaling yet
            cache_size=self.cache_batches,
            chunksize=self.chunk_size  # Use configurable chunk size for large files
        )
        
        # Verify we have enough samples
        if len(train_gen) < 2 or len(val_gen) < 1:
            self.logger.error(f"Not enough samples for training. Found {len(train_gen) * train_gen.batch_size} training and {len(val_gen) * val_gen.batch_size} validation samples")
            print(f"ERROR: Not enough samples for training")
            return
        
        # Fit scalers if needed - with memory optimization
        if self.apply_scaling and (self.model_scaler is None or not hasattr(self.model_scaler, 'fitted') or not self.model_scaler.fitted):
            self.logger.info("Fitting scalers on training data...")
            print("Fitting scalers on training data...")
            
            scaling_success = False
            try:
                scaling_success = self.fit_scalers(train_gen)
            except Exception as e:
                self.logger.error(f"Error fitting scalers: {e}")
                print(f"Error fitting scalers: {e}")
                
            # If scaling failed, try again with a subset of data
            if not scaling_success and file_size_gb > 1.0:
                self.logger.info("Trying to fit scalers on a subset of data...")
                try:
                    # Create a small generator just for scaling
                    scale_gen = MemoryEfficientGenerator(
                        csv_file=self.training_csv,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation=False,
                        val_split=0.01,  # Use only 1% for validation
                        max_rows=1000,   # Limit to 1000 rows for scaling
                        scaler=None,
                        cache_size=1
                    )
                    scaling_success = self.fit_scalers(scale_gen)
                    del scale_gen  # Remove reference to free memory
                    gc.collect()
                except Exception as e:
                    self.logger.error(f"Error fitting scalers on subset: {e}")
                    print(f"Error fitting scalers on subset: {e}")
                    
            if not scaling_success:
                self.logger.warning("Failed to fit scalers, will proceed without scaling")
                print("WARNING: Failed to fit scalers, will proceed without scaling")
            
        # Update generators to use scalers if available
        if self.model_scaler and hasattr(self.model_scaler, 'fitted') and self.model_scaler.fitted:
            train_gen.scaler = self.model_scaler
            val_gen.scaler = self.model_scaler
            self.logger.info("Data generators configured to use fitted scalers")
        else:
            self.logger.warning("Training without data scaling")
            print("WARNING: Training without data scaling")
        
        self.logger.info(f"Training with {len(train_gen)} batches, validation with {len(val_gen)} batches")
        print(f"Training with {len(train_gen)} batches, validation with {len(val_gen)} batches")

        # Create checkpoint directory
        os.makedirs("models/checkpoints", exist_ok=True)

        # Train model with enhanced gradient accumulation for large datasets
        self.logger.info(f"Using advanced gradient accumulation for massive model with {self.accum_steps} accumulation steps")
        print(f"Using advanced gradient accumulation for massive model with {self.accum_steps} accumulation steps")
        
        # Record start time
        train_start_time = time.time()
        
        # Train with optimized gradient accumulation method
        self.custom_gradient_accumulation_training(train_gen, val_gen)
        
        # Log total training time
        train_duration = time.time() - train_start_time
        hours = train_duration // 3600
        minutes = (train_duration % 3600) // 60
        seconds = train_duration % 60
        
        self.logger.info(f"Training completed in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        print(f"Training completed in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        
        # Clean up - explicitly delete generators and force GC
        del train_gen
        del val_gen
        gc.collect()
        
        # Report final memory
        if self.monitor_memory:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.logger.info(f"Final memory usage after LSTM training: {final_memory:.2f} MB")
                print(f"Final memory usage after LSTM training: {final_memory:.2f} MB")
            except:
                pass

    def train_rl(self):
        """Train the RL agent on offline transitions"""
        self.logger.info("Starting RL training...")
        print("Starting RL training...")
        
        # Check if transitions file exists
        if not os.path.exists(self.rl_transitions_csv):
            self.logger.error(f"RL transitions file not found: {self.rl_transitions_csv}")
            print(f"ERROR: RL transitions file not found: {self.rl_transitions_csv}")
            return
            
        # Load transitions
        transitions = self._load_rl_transitions()
        
        if not transitions:
            self.logger.error("No valid RL transitions found => abort RL training")
            print("ERROR: No valid RL transitions found => abort RL training")
            return
            
        self.logger.info(f"Loaded {len(transitions)} RL transitions")
        print(f"Loaded {len(transitions)} RL transitions")
        
        # Store transitions in agent memory
        for state, action, reward, next_state, done in transitions:
            self.dqn_agent.store_transition(state, action, reward, next_state, done)
            
        # Train agent
        self.logger.info(f"Training agent for {self.rl_epochs} epochs...")
        print(f"Training agent for {self.rl_epochs} epochs...")
        
        for epoch in range(self.rl_epochs):
            epoch_losses = []
            steps_per_epoch = min(len(transitions) // self.rl_batch_size * 5, 1000)
            
            for step in range(steps_per_epoch):
                loss = self.dqn_agent.train_step()
                if loss is not None:
                    epoch_losses.append(loss)
                
                # Log progress
                if (step + 1) % 100 == 0:
                    avg_recent_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                    self.logger.info(f"  Step {step+1}/{steps_per_epoch} - Avg Recent Loss: {avg_recent_loss:.6f}")
                    
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            
            self.logger.info(f"Epoch {epoch+1}/{self.rl_epochs} - Avg Loss: {avg_loss:.6f} - Epsilon: {self.dqn_agent.epsilon:.4f}")
            print(f"Epoch {epoch+1}/{self.rl_epochs} - Avg Loss: {avg_loss:.6f} - Epsilon: {self.dqn_agent.epsilon:.4f}")
            
            # Save checkpoint after each epoch
            if (epoch + 1) % 2 == 0 or epoch == self.rl_epochs - 1:
                checkpoint_path = f"models/rl_DQNAgent_epoch{epoch+1}.weights.h5"
                self.dqn_agent.save(checkpoint_path)
                self.logger.info(f"RL agent checkpoint saved to {checkpoint_path}")
            
        # Save final agent
        self.dqn_agent.save(self.rl_model_out)
        self.logger.info(f"RL agent saved to {self.rl_model_out}")
        print(f"RL agent saved to {self.rl_model_out}")
        
    def _load_rl_transitions(self):
        """Load RL transitions from CSV file with advanced progress reporting for large files"""
        transitions = []
        error_count = 0
        
        try:
            # First, count number of rows (without parsing)
            total_rows = 0
            with open(self.rl_transitions_csv, 'r') as f:
                for _ in f:
                    total_rows += 1
            
            # Skip header row
            total_rows -= 1
            
            self.logger.info(f"Found {total_rows} rows in RL transitions file")
            print(f"Found {total_rows} rows in RL transitions file")
            
            # Now parse with progress reporting
            with open(self.rl_transitions_csv, 'r') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader, 1):
                    try:
                        # Parse fields
                        old_state = np.array(json.loads(row['old_state']), dtype=np.float32)
                        action = row['action']
                        reward = float(row['reward'])
                        new_state = np.array(json.loads(row['new_state']), dtype=np.float32)
                        done = row['done'].lower() in ['1', 'true', 't', 'yes', 'y']
                        
                        # Validate dimensions
                        if old_state.shape[0] != NUM_FUTURE_STEPS + 3 or new_state.shape[0] != NUM_FUTURE_STEPS + 3:
                            error_count += 1
                            continue
                            
                        # Validate action
                        if action not in ACTIONS:
                            error_count += 1
                            continue
                            
                        # Add to transitions
                        transitions.append((old_state, action, reward, new_state, done))
                        
                        # Report progress
                        if i % 1000 == 0 or i == total_rows:
                            self.logger.info(f"Loaded {i}/{total_rows} transitions ({i/total_rows*100:.1f}%)")
                        
                    except Exception as e:
                        error_count += 1
                        if error_count < 10:  # Log only first few errors
                            self.logger.error(f"Error parsing transition {i}: {e}")
        except Exception as e:
            self.logger.error(f"Error reading RL transitions file: {e}")
            
        if error_count > 0:
            self.logger.warning(f"Skipped {error_count} invalid transitions")
            
        return transitions

    def run(self):
        """Run the training pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING TRAINING PIPELINE FOR MASSIVE MODEL")
        self.logger.info("=" * 80)
        
        # Train LSTM model
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

        # Train RL agent
        if not self.skip_rl:
            try:
                self.logger.info("=== Starting RL Training ===")
                print("=== Starting RL Training ===")
                self.train_rl()
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
            self.logger.info("Skipping RL training.")
            print("Skipping RL training.")
            
        self.logger.info("=" * 80)
        self.logger.info("TRAINING PIPELINE COMPLETE")
        self.logger.info("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ultra-optimized training script for massive (500MB+) models and large datasets (6GB+)"
    )
    parser.add_argument("--csv", type=str, default=LSTM_DATA_FILE,
                        help="Path to multi-output training data CSV.")
    parser.add_argument("--model_out", type=str, default="models/advanced_lstm_model.keras",
                        help="File path for the LSTM model.")
    parser.add_argument("--rl_csv", type=str, default=RL_TRANSITIONS_FILE,
                        help="Path to RL transitions CSV.")
    parser.add_argument("--rl_out", type=str, default="models/rl_DQNAgent.weights.h5",
                        help="Output file for RL weights.")
    parser.add_argument("--epochs", type=int, default=100, help="LSTM epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="early_stop_patience")
    parser.add_argument("--batch_size", type=int, default=4, help="LSTM batch size. Use smaller values (2-4) for massive models.")
    parser.add_argument("--rl_batch_size", type=int, default=64, help="RL batch size.")
    parser.add_argument("--no_scale", action="store_true",
                        help="Disable feature scaling.")
    parser.add_argument("--skip_lstm", action="store_true",
                        help="Skip LSTM training entirely.")
    parser.add_argument("--skip_rl", action="store_true",
                        help="Skip RL training entirely.")
    parser.add_argument("--max_rows", type=int, default=0, 
                        help="Load x rows from csv file. 0 is all.")
    parser.add_argument("--grad_accum", action="store_true", 
                        help="Use gradient accumulation for memory efficiency")
    parser.add_argument("--accum_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--model_size", type=str, default="massive", 
                        choices=["small", "medium", "large", "xlarge", "massive", "gigantic"],
                        help="Model size (small=48, medium=64, large=96, xlarge=128, massive=512, gigantic=1024 units)")
    parser.add_argument("--no_reduce_precision", action="store_true",
                        help="Don't use reduced precision (use full float32)")
    parser.add_argument("--rl_epochs", type=int, default=10,
                        help="Offline RL training epochs.")
    parser.add_argument("--cache_batches", type=int, default=16,
                        help="Number of batches to cache in memory (0 to disable)")
    parser.add_argument("--delete_nn_model", action="store_true",
                        help="Delete the existing models directory before starting")
    
    # Add new memory optimization parameters
    parser.add_argument("--no_mmap", action="store_true",
                        help="Disable memory-mapped file support for large CSV files")
    parser.add_argument("--no_lazy_loading", action="store_true",
                        help="Disable lazy loading for large datasets")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Chunk size for CSV reading (larger values use more memory)")
    parser.add_argument("--prefetch_batches", type=int, default=2,
                        help="Number of batches to prefetch (0 to disable)")
    parser.add_argument("--no_memory_monitor", action="store_true",
                        help="Disable memory usage monitoring")
    parser.add_argument("--no_tf_allow_growth", action="store_true",
                        help="Disable TensorFlow memory growth (use fixed allocation)")
    parser.add_argument("--no_aggressive_gc", action="store_true",
                        help="Disable aggressive garbage collection")

    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("ULTRA-OPTIMIZED TRAINING SCRIPT FOR MASSIVE MODELS (500MB+) AND LARGE DATASETS (6GB+)")
    print("="*80 + "\n")
    
    # Delete models directory if requested
    if args.delete_nn_model:
        try:
            models_dir = "models"
            if os.path.exists(models_dir):
                print(f"Deleting existing models directory: {models_dir}")
                shutil.rmtree(models_dir)
                print(f"Successfully deleted {models_dir} directory")
            else:
                print(f"No existing models directory found to delete")
        except Exception as e:
            print(f"Error deleting models directory: {e}")
    
    # Print system information
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Number of GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    
    # Print GPU information if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i}: {gpu_details.get('device_name', 'Unknown')} "
                      f"({gpu_details.get('compute_capability', 'Unknown')})")
            except:
                print(f"GPU {i}: {gpu.name}")
    
    print(f"Model size: {args.model_size.upper()}")
    print(f"Training file: {args.csv}")
    print(f"RL training file: {args.rl_csv}")
    
    # Print batch size and gradient accumulation settings
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum}, "
          f"Accumulation steps: {args.accum_steps}")
    print(f"Effective batch size: {args.batch_size * (args.accum_steps if args.grad_accum else 1)}")
    
    # Print memory optimization settings
    memory_optimizations = []
    if not args.no_mmap:
        memory_optimizations.append("memory-mapped files")
    if not args.no_lazy_loading:
        memory_optimizations.append("lazy loading")
    if not args.no_memory_monitor:
        memory_optimizations.append("memory monitoring")
    if not args.no_tf_allow_growth:
        memory_optimizations.append("TF memory growth")
    if not args.no_aggressive_gc:
        memory_optimizations.append("aggressive GC")
    
    # Check CSV file size
    try:
        file_size_bytes = os.path.getsize(args.csv)
        file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
        print(f"CSV file size: {file_size_gb:.2f} GB")
        
        if file_size_gb > 5.0:
            print("Large CSV file detected! Using optimized data loading techniques.")
    except:
        print("Could not determine CSV file size.")
    
    print(f"Memory optimizations: {', '.join(memory_optimizations)}")
    print(f"CSV chunk size: {args.chunk_size}")
    print("="*80 + "\n")

    # Initialize trainer
    try:
        trainer = Trainer(
            training_csv=args.csv,
            model_out=args.model_out,
            rl_transitions_csv=args.rl_csv,
            rl_model_out=args.rl_out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            rl_batch_size=args.rl_batch_size,
            early_stop_patience=args.early_stop_patience,
            apply_scaling=not args.no_scale,
            skip_lstm=args.skip_lstm,
            skip_rl=args.skip_rl,
            max_rows=args.max_rows,
            grad_accum=args.grad_accum,
            accum_steps=args.accum_steps,
            model_size=args.model_size,
            reduce_precision=not args.no_reduce_precision,
            rl_epochs=args.rl_epochs,
            cache_batches=args.cache_batches,
            use_mmap=not args.no_mmap,
            lazy_loading=not args.no_lazy_loading,
            chunk_size=args.chunk_size,
            prefetch_batches=args.prefetch_batches,
            monitor_memory=not args.no_memory_monitor,
            tf_allow_growth=not args.no_tf_allow_growth,
            aggressive_gc=not args.no_aggressive_gc
        )
        
        # Run training
        trainer.run()
        
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
    
    





# conda activate env3


# For small debug model:
# python fitter.py --csv training_data\training_data.csv --model_size small --batch_size 4 --grad_accum --accum_steps 2 --skip_rl --chunk_size 10000 --cache_batches 8 --delete_nn_model --no_reduce_precision

# For gigantic model:
# python fitter.py --model_size gigantic --batch_size 2 --grad_accum --accum_steps 16 --rl_epochs 5 --no_reduce_precision --delete_nn_model


# dir_print . -I .git .gitignore dir_print.txt 2022 2023 2024 2025 -O ^.png^ ^.csv^ ^.json^ ^.pyc^ ^input_cache^ ^logs_training^ ^.env^ ^config^ ^.txt^ ^.log^ -E dir_print.txt --sos --line-count

# tensorboard --logdir=logs_training
# nvidia-smi -l 1
