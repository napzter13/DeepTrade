"""
Advanced Neural Network Model for Multi-timeframe Trading Predictions

Robust implementation with:
- Efficient parameter scaling
- Clean architecture with parallel processing pathways
- Feature extraction with proven regularization techniques
- Compatible dimensions between layers
- Memory-optimized training
"""

import tensorflow as tf
import numpy as np
import typing as tp
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import time

# Explicit imports to avoid name resolution issues
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.mixed_precision import set_global_policy

from .environment import NUM_FUTURE_STEPS

# Set up constants
GPU_MEMORY_LIMIT = 0.90  # Use 90% of GPU memory
DEFAULT_DTYPE = 'float32'

# Type aliases for code readability
TensorType = Union[tf.Tensor, np.ndarray]

# =============================================================================
# GPU SETUP & OPTIMIZATION
# =============================================================================
def setup_gpu_memory():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found. Running on CPU.")
        return False
    
    try:
        # Enable memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable tensor cores if available
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
        except:
            pass
        
        # Enable XLA compilation 
        try:
            tf.config.optimizer.set_jit(True)
        except:
            pass
        
        print(f"GPU setup successful. Found {len(gpus)} GPU(s)")
        return True
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
        return False

# Setup GPU memory - call this at module import time
setup_gpu_memory()

# =============================================================================
# CUSTOM LAYERS & LOSS FUNCTIONS
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="botlib")
def safe_mse_loss(y_true: TensorType, y_pred: TensorType) -> tf.Tensor:
    """
    Numerically stable MSE loss function with safeguards against NaN values
    and mixed precision compatibility.
    """
    # Ensure both tensors have the same dtype
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, -0.999, 0.999)
    
    # Calculate squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Handle NaN values in ground truth
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, tf.float32)
    
    # Apply mask and calculate mean
    masked_error = squared_error * mask
    count = tf.reduce_sum(mask, axis=-1)
    count = tf.maximum(count, 1.0)  # Prevent division by zero
    
    return tf.reduce_sum(masked_error, axis=-1) / count


# =============================================================================
# TIME SERIES ENCODER
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="botlib")
class TimeSeriesEncoder(layers.Layer):
    """Simplified and robust time series encoder"""
    
    def __init__(self, 
                 units: int = 64,
                 depth: int = 2,
                 dropout_rate: float = 0.1,
                 name: str = None):
        super(TimeSeriesEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.name_prefix = name or "ts_encoder"
        
        # Initial projection
        self.initial_conv = layers.Conv1D(
            units, 
            kernel_size=3, 
            padding='same',
            activation='relu',
            name=f"{self.name_prefix}_init_conv"
        )
        
        # Stacked convolutional layers
        self.conv_layers = []
        for i in range(depth):
            conv = layers.Conv1D(
                units,
                kernel_size=5,
                padding='same',
                activation=None,
                name=f"{self.name_prefix}_conv{i+1}"
            )
            bn = layers.BatchNormalization(name=f"{self.name_prefix}_bn{i+1}")
            act = layers.Activation('relu', name=f"{self.name_prefix}_act{i+1}")
            drop = layers.SpatialDropout1D(dropout_rate, name=f"{self.name_prefix}_drop{i+1}")
            
            self.conv_layers.append((conv, bn, act, drop))
        
        # Bidirectional GRU for sequence modeling
        self.gru = layers.Bidirectional(
            layers.GRU(
                units // 2,  # Half the units since we'll concatenate forward and backward
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=0.0,  # Avoid recurrent dropout for CuDNN compatibility
                name=f"{self.name_prefix}_gru"
            ),
            name=f"{self.name_prefix}_bidirectional"
        )
        
        # Pooling layers
        self.global_avg_pool = layers.GlobalAveragePooling1D(name=f"{self.name_prefix}_avg_pool")
        self.global_max_pool = layers.GlobalMaxPooling1D(name=f"{self.name_prefix}_max_pool")
        
        # Final dense layer
        self.final_dense = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{self.name_prefix}_final"
        )
        
    def call(self, inputs, training=None):
        # Initial processing
        x = self.initial_conv(inputs)
        
        # Apply convolutional layers with residual connections
        for i, (conv, bn, act, drop) in enumerate(self.conv_layers):
            residual = x
            x = conv(x)
            x = bn(x, training=training)
            x = act(x)
            x = drop(x, training=training)
            
            # Add residual connection if shapes match
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
        
        # Apply GRU
        x = self.gru(x)
        
        # Global pooling - combine different pooling methods
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Combine features
        combined = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Final projection
        output = self.final_dense(combined)
        
        return output
    
    def get_config(self):
        config = super(TimeSeriesEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'name': self.name_prefix
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class TabularEncoder(layers.Layer):
    """Simplified encoder for tabular features"""
    
    def __init__(self, 
                 units: int = 64,
                 depth: int = 3,
                 dropout_rate: float = 0.2,
                 name: str = None):
        super(TabularEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.name_prefix = name or "tab_encoder"
        
        # Initial projection
        self.initial_dense = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{self.name_prefix}_init_dense"
        )
        self.initial_bn = layers.BatchNormalization(name=f"{self.name_prefix}_init_bn")
        self.initial_act = layers.Activation('relu', name=f"{self.name_prefix}_init_act")
        
        # Stacked dense layers with skip connections
        self.dense_layers = []
        for i in range(depth):
            # Dense layer with normalization and activation
            dense = layers.Dense(
                units,
                activation=None,
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"{self.name_prefix}_dense{i+1}"
            )
            bn = layers.BatchNormalization(name=f"{self.name_prefix}_bn{i+1}")
            act = layers.Activation('relu', name=f"{self.name_prefix}_act{i+1}")
            drop = layers.Dropout(dropout_rate, name=f"{self.name_prefix}_drop{i+1}")
            
            self.dense_layers.append((dense, bn, act, drop))
        
    def call(self, inputs, training=None):
        # Initial projection
        x = self.initial_dense(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_act(x)
        
        # Process through dense layers with skip connections
        for i, (dense, bn, act, drop) in enumerate(self.dense_layers):
            residual = x
            x = dense(x)
            x = bn(x, training=training)
            x = act(x)
            x = drop(x, training=training)
            
            # Add residual connection
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
            
        return x
    
    def get_config(self):
        config = super(TabularEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'name': self.name_prefix
        })
        return config


# =============================================================================
# MAIN MODEL FUNCTION
# =============================================================================

def build_ensemble_model(
    model_5m_window: int = 241,
    model_15m_window: int = 241,
    model_1h_window: int = 241,
    feature_dim: int = 9,
    santiment_dim: int = 12,
    ta_dim: int = 63,
    signal_dim: int = 11,
    base_units: int = 64,
    depth: int = 2,
    memory_efficient: bool = True,
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 8,
    mixed_precision: bool = True,
    massive_model: bool = False,  # Flag to switch between normal and massive model
    **kwargs
) -> keras.models.Model:
    """
    Create a robust multi-timeframe model for trading predictions
    """
    print("\n" + "="*80)
    print(" CREATING TRADING MODEL ")
    print("="*80)
    
    # Enable mixed precision if requested
    if mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"Mixed precision enabled with policy: {policy.name}")
        except Exception as e:
            print(f"Mixed precision not available: {e}")
    
    # Input layers - these are the same regardless of model size
    input_5m = layers.Input(shape=(model_5m_window, feature_dim), name="input_5m")
    input_15m = layers.Input(shape=(model_15m_window, feature_dim), name="input_15m")
    input_1h = layers.Input(shape=(model_1h_window, feature_dim), name="input_1h")
    input_google = layers.Input(shape=(24, 1), name="input_google_trend")
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    
    # Scale up units for massive model
    if massive_model:
        # Let's use a multiplier for the base units
        actual_units = base_units * 2
        print(f"Using massive model with {actual_units} units")
    else:
        actual_units = base_units
        print(f"Using standard model with {actual_units} units")
    
    # Time series encoders
    ts_encoder_5m = TimeSeriesEncoder(
        units=actual_units,
        depth=depth,
        dropout_rate=0.2,
        name="enc_5m"
    )
    
    ts_encoder_15m = TimeSeriesEncoder(
        units=actual_units,
        depth=depth,
        dropout_rate=0.2,
        name="enc_15m"
    )
    
    ts_encoder_1h = TimeSeriesEncoder(
        units=actual_units,
        depth=depth,
        dropout_rate=0.2,
        name="enc_1h"
    )
    
    ts_encoder_google = TimeSeriesEncoder(
        units=actual_units // 2,
        depth=2,
        dropout_rate=0.2,
        name="enc_google"
    )
    
    # Tabular data encoders
    tab_encoder_santiment = TabularEncoder(
        units=actual_units // 2,
        depth=2,
        dropout_rate=0.2,
        name="enc_santiment"
    )
    
    tab_encoder_ta = TabularEncoder(
        units=actual_units,
        depth=2,
        dropout_rate=0.2,
        name="enc_ta"
    )
    
    tab_encoder_signal = TabularEncoder(
        units=actual_units // 2,
        depth=2,
        dropout_rate=0.2,
        name="enc_signal"
    )
    
    # Process each input through its encoder
    feat_5m = ts_encoder_5m(input_5m)
    feat_15m = ts_encoder_15m(input_15m)
    feat_1h = ts_encoder_1h(input_1h)
    feat_google = ts_encoder_google(input_google)
    
    feat_santiment = tab_encoder_santiment(input_santiment)
    feat_ta = tab_encoder_ta(input_ta)
    feat_signal = tab_encoder_signal(input_signal)
    
    # Enhanced cross-attention mechanism
    # First, stack the time series features
    time_features = tf.stack([feat_5m, feat_15m, feat_1h], axis=1)
    
    # Apply attention mechanism
    attention_query = layers.Dense(actual_units, activation='tanh')(time_features)
    attention_weights = layers.Dense(1)(attention_query)
    attention_weights = layers.Softmax(axis=1)(attention_weights)
    
    # Apply attention to get weighted time series features
    cross_context = layers.Multiply()([time_features, attention_weights])
    cross_context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(cross_context)
    
    # Concatenate all features
    combined = layers.Concatenate(name="all_features")([
        cross_context, feat_google, feat_santiment, feat_ta, feat_signal
    ])
    
    # Shared trunk network with residual connections
    x = combined
    for i in range(2):  # Simplified trunk with just 2 blocks
        # First dense block
        residual = x
        x = layers.Dense(actual_units * 2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Second dense block 
        x = layers.Dense(actual_units * 2)(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual connection with projection if needed
        if residual.shape[-1] != x.shape[-1]:
            residual = layers.Dense(x.shape[-1])(residual)
            
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
    
    # Multiple output heads for different time horizons
    outputs = []
    for i in range(NUM_FUTURE_STEPS):
        head = layers.Dense(
            1, 
            activation='tanh',  # tanh for [-1,1] range 
            name=f"output_{i+1}"
        )(x)
        outputs.append(head)
    
    # Concatenate all outputs
    output = layers.Concatenate(name="all_predictions")(outputs)
    
    # Create model
    model = keras.Model(
        inputs=[input_5m, input_15m, input_1h, input_google, input_santiment, input_ta, input_signal],
        outputs=output,
        name="multi_timeframe_trading_model"
    )
    
    # Compile the model
    optimizer = optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,  # Add gradient clipping for stability
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss=safe_mse_loss,
        metrics=['mae']  # Mean Absolute Error
    )
    
    # Print model summary
    total_params = model.count_params()
    print(f"Model created with {total_params:,} parameters")
    print(f"Output dimension: {NUM_FUTURE_STEPS}")
    print(f"Approximate model size: {total_params * 4 / (1024 * 1024):.2f} MB")
    print("="*80 + "\n")
    
    return model


# Backwards compatibility function
def load_advanced_lstm_model(
    model_5m_window: int = 241,
    model_15m_window: int = 241,
    model_1h_window: int = 241,
    feature_dim: int = 9,
    santiment_dim: int = 12,
    ta_dim: int = 63,
    signal_dim: int = 11,
    base_units: int = 64,
    memory_efficient: bool = True,
    mixed_precision: bool = True,
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 8,
    depth: int = 2,
    massive_model: bool = False,
    **kwargs
) -> keras.models.Model:
    """Wrapper for backwards compatibility"""
    return build_ensemble_model(
        model_5m_window=model_5m_window,
        model_15m_window=model_15m_window,
        model_1h_window=model_1h_window,
        feature_dim=feature_dim,
        santiment_dim=santiment_dim,
        ta_dim=ta_dim,
        signal_dim=signal_dim,
        base_units=base_units,
        depth=depth,
        memory_efficient=memory_efficient,
        mixed_precision=mixed_precision,
        gradient_accumulation=gradient_accumulation,
        gradient_accumulation_steps=gradient_accumulation_steps,
        massive_model=massive_model,
        **kwargs
    )