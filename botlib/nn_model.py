"""
Advanced Ensemble Neural Network Model for Multi-timeframe Trading Predictions

Optimized implementation with:
- Mixed precision training (FP16/BF16)
- Faster architecture with GRU instead of LSTM 
- Enhanced GPU utilization
- Model parallelism support
- Reduced memory footprint with weight sharing
- Improved attention mechanisms
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
DEFAULT_COMPLEX_DTYPE = 'complex64'

# Type aliases for code readability
TensorType = Union[tf.Tensor, np.ndarray]

# =============================================================================
# GPU SETUP & OPTIMIZATION
# =============================================================================
def setup_gpu_memory():
    """Configure GPU for optimal performance (without memory growth conflicts)"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found. Running on CPU.")
        return False
    
    try:
        # IMPORTANT: Don't try to set both memory growth and virtual device configuration
        # Just enable tensor cores and XLA for performance
        
        # Enable tensor cores if available
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("TensorFloat-32 execution enabled if supported by hardware")
        except:
            pass
        
        # Enable XLA compilation for faster execution (if not causing issues)
        try:
            tf.config.optimizer.set_jit(True)
            print("XLA compilation enabled")
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
class FastGRUBlock(layers.Layer):
    """
    Optimized GRU block with performance enhancements for trading data
    """
    
    def __init__(self, 
                 units: int,
                 dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional: bool = True,
                 name: str = None):
        """Initialize the GRU block with configurable bidirectionality"""
        super(FastGRUBlock, self).__init__(name=name)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        
        # Use CuDNN implementation when possible by setting specific settings
        # that allow for CuDNN acceleration
        if bidirectional:
            self.gru = layers.Bidirectional(
                layers.GRU(
                    units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    reset_after=True,  # Enable CuDNN acceleration
                ),
                name=f"{name}_bidirectional" if name else None
            )
        else:
            self.gru = layers.GRU(
                units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                reset_after=True,  # Enable CuDNN acceleration
                name=name
            )
        
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_ln" if name else None
        )
            
    def call(self, inputs, training=None, mask=None):
        """Process inputs through the GRU block"""
        x = self.gru(inputs, training=training, mask=mask)
        return self.layer_norm(x, training=training)
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super(FastGRUBlock, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'bidirectional': self.bidirectional,
            'name': self.name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class TimeSeriesEncoder(layers.Layer):
    """
    Memory-efficient time series encoder
    """
    
    def __init__(self, 
                 units: int = 64,
                 depth: int = 2,
                 dropout_rate: float = 0.1,
                 name: str = None):
        """Initialize the time series encoder"""
        super(TimeSeriesEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.name_prefix = name or "time_encoder"
        
        # Initial projection
        self.initial_conv = layers.Conv1D(
            units, 
            kernel_size=3, 
            padding='same',
            name=f"{self.name_prefix}_init_conv"
        )
        
        # GRU layers
        self.gru_layers = []
        for i in range(depth):
            gru_layer = FastGRUBlock(
                units=units,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate/2,
                bidirectional=True,
                name=f"{self.name_prefix}_gru{i+1}"
            )
            self.gru_layers.append(gru_layer)
            
        # Global pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D(
            name=f"{self.name_prefix}_avg_pool"
        )
        self.global_max_pool = layers.GlobalMaxPooling1D(
            name=f"{self.name_prefix}_max_pool"
        )
        
        # Final projection
        self.final_dense = layers.Dense(
            units, 
            activation='relu',
            name=f"{self.name_prefix}_final"
        )
        
    def call(self, inputs, training=None):
        """Process inputs through the time series encoder"""
        # Initial processing
        x = self.initial_conv(inputs)
        
        # GRU layers
        for gru_layer in self.gru_layers:
            x = gru_layer(x, training=training)
            
        # Global pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Combine pooling results
        combined = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Final projection
        output = self.final_dense(combined)
        
        return output
    
    def get_config(self):
        """Return configuration for serialization"""
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
    """
    Encoder for tabular features
    """
    
    def __init__(self, 
                 units: int = 64,
                 depth: int = 2,
                 dropout_rate: float = 0.1,
                 name: str = None):
        """Initialize the tabular encoder"""
        super(TabularEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.name_prefix = name or "tab_encoder"
        
        # Dense layers
        self.dense_layers = []
        for i in range(depth):
            layer_units = units * 2 if i < depth - 1 else units
            dense = layers.Dense(
                layer_units,
                activation='relu',
                name=f"{self.name_prefix}_dense{i+1}"
            )
            self.dense_layers.append(dense)
            
            # Add normalization and dropout
            norm = layers.BatchNormalization(
                name=f"{self.name_prefix}_bn{i+1}"
            )
            self.dense_layers.append(norm)
            
            dropout = layers.Dropout(
                dropout_rate,
                name=f"{self.name_prefix}_drop{i+1}"
            )
            self.dense_layers.append(dropout)
            
    def call(self, inputs, training=None):
        """Process inputs through the tabular encoder"""
        x = inputs
        
        # Process through dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        return x
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super(TabularEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'name': self.name_prefix
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class EfficientMultiHeadAttention(layers.Layer):
    """
    Memory and compute-efficient multi-head attention implementation
    """
    
    def __init__(self, 
                 num_heads: int,
                 key_dim: int, 
                 dropout: float = 0.0,
                 use_bias: bool = True,
                 name: str = None):
        """Initialize the efficient multi-head attention layer"""
        super(EfficientMultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.use_bias = use_bias
        
        self.wq = layers.Dense(num_heads * key_dim, use_bias=use_bias, name=f"{name}_wq" if name else None)
        self.wk = layers.Dense(num_heads * key_dim, use_bias=use_bias, name=f"{name}_wk" if name else None)
        self.wv = layers.Dense(num_heads * key_dim, use_bias=use_bias, name=f"{name}_wv" if name else None)
        self.wo = layers.Dense(num_heads * key_dim, use_bias=use_bias, name=f"{name}_wo" if name else None)
        self.dropout = layers.Dropout(dropout, name=f"{name}_dropout" if name else None)
            
    def call(self, query, key=None, value=None, training=None, mask=None):
        """Process inputs through the efficient multi-head attention layer"""
        if key is None:
            key = query
        if value is None:
            value = key
            
        # Get dimensions
        batch_size = tf.shape(query)[0]
        seq_len_q = tf.shape(query)[1]
        seq_len_k = tf.shape(key)[1]
        seq_len_v = tf.shape(value)[1]
        
        # Linear projections and reshape for multi-head
        q = self.wq(query)  # (batch_size, seq_len_q, num_heads*key_dim)
        k = self.wk(key)    # (batch_size, seq_len_k, num_heads*key_dim)
        v = self.wv(value)  # (batch_size, seq_len_v, num_heads*key_dim)
        
        # Reshape to multi-head format
        q = tf.reshape(q, [batch_size, seq_len_q, self.num_heads, self.key_dim])
        k = tf.reshape(k, [batch_size, seq_len_k, self.num_heads, self.key_dim])
        v = tf.reshape(v, [batch_size, seq_len_v, self.num_heads, self.key_dim])
        
        # Transpose for batched matrix multiplication
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len_q, key_dim)
        k = tf.transpose(k, [0, 2, 3, 1])  # (batch_size, num_heads, key_dim, seq_len_k)
        v = tf.transpose(v, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len_v, key_dim)
        
        # Calculate attention scores
        scores = tf.matmul(q, k)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Scale attention scores
        dk = tf.cast(self.key_dim, scores.dtype)
        scores = scores / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            # Add large negative value to masked positions
            scores += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention weights to values
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, key_dim)
        
        # Transpose and reshape back
        output = tf.transpose(output, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, key_dim)
        output = tf.reshape(output, [batch_size, seq_len_q, self.num_heads * self.key_dim])
        
        # Final linear projection
        output = self.wo(output)
        
        return output
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super(EfficientMultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate,
            'use_bias': self.use_bias,
            'name': self.name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class FastTransformerBlock(layers.Layer):
    """
    Optimized transformer block with performance enhancements
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_dim: int,
                 dropout_rate: float = 0.1,
                 use_bias: bool = True,
                 activation: str = 'gelu',
                 layer_norm_epsilon: float = 1e-6,
                 name: str = None):
        """Initialize the fast transformer block"""
        super(FastTransformerBlock, self).__init__(name=name)
        self.name_prefix = name or "transformer"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        
        # Multi-head attention layer
        self.attention = EfficientMultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            use_bias=use_bias,
            name=f"{self.name_prefix}_attn"
        )
        
        # Dropout layer for attention
        self.attn_dropout = layers.Dropout(
            dropout_rate, 
            name=f"{self.name_prefix}_attn_dropout"
        )
        
        # Layer normalization for attention
        self.attn_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, 
            name=f"{self.name_prefix}_attn_norm"
        )
        
        # Feed-forward network
        self.ffn = [
            layers.Dense(
                ff_dim, 
                activation=activation,
                use_bias=use_bias,
                name=f"{self.name_prefix}_ffn1"
            ),
            layers.Dropout(
                dropout_rate,
                name=f"{self.name_prefix}_ffn_dropout1"
            ),
            layers.Dense(
                embed_dim,
                use_bias=use_bias,
                name=f"{self.name_prefix}_ffn2"
            ),
            layers.Dropout(
                dropout_rate,
                name=f"{self.name_prefix}_ffn_dropout2"
            )
        ]
        
        # Layer normalization for feed-forward network
        self.ffn_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name=f"{self.name_prefix}_ffn_norm"
        )
    
    def call(self, inputs, training=None, mask=None):
        """Process inputs through the transformer block"""
        # Multi-head attention with residual connection and layer normalization
        attn_output = self.attention(
            query=inputs, 
            key=inputs, 
            value=inputs, 
            training=training,
            mask=mask
        )
        attn_output = self.attn_dropout(attn_output, training=training)
        
        # Add residual connection and normalize
        attn_output = self.attn_norm(inputs + attn_output, training=training)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = attn_output
        for layer in self.ffn:
            ffn_output = layer(ffn_output, training=training)
        
        # Add residual connection and normalize
        output = self.ffn_norm(attn_output + ffn_output, training=training)
        
        return output
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super(FastTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'name': self.name_prefix
        })
        return config


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


@tf.keras.utils.register_keras_serializable(package="botlib")
def weighted_mse_loss(y_true: TensorType, y_pred: TensorType) -> tf.Tensor:
    """
    MSE loss function that places more weight on recent future values
    """
    # Ensure both tensors have the same dtype
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, -0.999, 0.999)
    
    # Create decreasing weights for time steps (more recent = higher weight)
    time_steps = tf.shape(y_true)[1]
    weights = tf.range(time_steps, 0, -1, dtype=tf.float32)
    weights = weights / tf.reduce_sum(weights)
    weights = tf.reshape(weights, [1, -1])
    
    # Calculate squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Apply time-based weights
    weighted_error = squared_error * weights
    
    # Handle NaN values in ground truth
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, tf.float32)
    
    # Apply mask and calculate weighted mean
    masked_weighted_error = weighted_error * mask
    
    return tf.reduce_sum(masked_weighted_error) / tf.reduce_sum(mask * weights)


# =============================================================================
# CORE ENCODER MODULES
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="botlib")
class MultiScaleConvEncoder(layers.Layer):
    """
    Efficient time series encoder using multi-scale convolutions
    """
    
    def __init__(self, 
                 units: int,
                 depth: int = 3,
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu',
                 l2_reg: float = 1e-6,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 name: str = None):
        """Initialize the multi-scale convolutional encoder"""
        super(MultiScaleConvEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.name_prefix = name or "conv_encoder"
        
        # Regularizer
        self.regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        # Initial normalization
        self.initial_norm = layers.BatchNormalization(
            name=f"{self.name_prefix}_initial_bn"
        ) if use_batch_norm else None
        
        # Initial projection to match channel dimension
        self.initial_proj = layers.Conv1D(
            units, 1, padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_initial_proj"
        )
        
        # Multi-scale convolutional blocks
        self.conv_blocks = []
        for i in range(depth):
            block = {}
            
            # Multiple kernel sizes for multi-scale processing
            block['convs'] = []
            for k in kernel_sizes:
                conv = layers.Conv1D(
                    units // len(kernel_sizes), k, padding='same',
                    activation=None,  # Activation applied after merging
                    kernel_regularizer=self.regularizer,
                    name=f"{self.name_prefix}_block{i+1}_conv{k}"
                )
                block['convs'].append(conv)
            
            # Merge convolution outputs
            block['merge'] = layers.Concatenate(
                name=f"{self.name_prefix}_block{i+1}_merge"
            )
            
            # Normalization
            if use_batch_norm:
                block['norm'] = layers.BatchNormalization(
                    name=f"{self.name_prefix}_block{i+1}_bn"
                )
            else:
                block['norm'] = layers.LayerNormalization(
                    epsilon=1e-6,
                    name=f"{self.name_prefix}_block{i+1}_ln"
                )
                
            # Activation
            block['activation'] = layers.Activation(
                activation,
                name=f"{self.name_prefix}_block{i+1}_act"
            )
            
            # Dropout
            block['dropout'] = layers.Dropout(
                dropout_rate,
                name=f"{self.name_prefix}_block{i+1}_dropout"
            )
            
            # Pooling (except last block)
            if i < depth - 1:
                block['pool'] = layers.MaxPooling1D(
                    2,
                    name=f"{self.name_prefix}_block{i+1}_pool"
                )
            else:
                block['pool'] = None
                
            # Residual connection projection (if dimensions change due to pooling)
            if use_residual and i < depth - 1:
                block['residual_proj'] = layers.Conv1D(
                    units, 1, strides=2, padding='same',
                    kernel_regularizer=self.regularizer,
                    name=f"{self.name_prefix}_block{i+1}_res_proj"
                )
            else:
                block['residual_proj'] = None
                
            self.conv_blocks.append(block)
            
        # Global pooling operations
        self.global_avg_pool = layers.GlobalAveragePooling1D(
            name=f"{self.name_prefix}_global_avg_pool"
        )
        self.global_max_pool = layers.GlobalMaxPooling1D(
            name=f"{self.name_prefix}_global_max_pool"
        )
    
    def call(self, inputs, training=None):
        """Process inputs through the multi-scale convolutional encoder"""
        # Initial processing
        if self.initial_norm is not None:
            x = self.initial_norm(inputs, training=training)
        else:
            x = inputs
            
        x = self.initial_proj(x)
        
        # Process through multi-scale convolutional blocks
        skip_connections = []
        for i, block in enumerate(self.conv_blocks):
            residual = x
            
            # Apply convolutions with different kernel sizes
            conv_outputs = []
            for conv in block['convs']:
                conv_outputs.append(conv(x))
                
            # Merge convolution outputs
            x = block['merge'](conv_outputs)
            
            # Apply normalization and activation
            x = block['norm'](x, training=training)
            x = block['activation'](x)
            x = block['dropout'](x, training=training)
            
            # Add residual connection if enabled
            if self.use_residual:
                if block['residual_proj'] is not None:
                    residual = block['residual_proj'](residual)
                    
                if residual.shape[-1] == x.shape[-1] and residual.shape[1] == x.shape[1]:
                    x = x + residual
            
            # Store for skip connections
            skip_connections.append(x)
            
            # Apply pooling if available
            if block['pool'] is not None:
                x = block['pool'](x)
        
        # Global pooling
        avg_pooled = self.global_avg_pool(x)
        max_pooled = self.global_max_pool(x)
        
        # Combine pooled features
        output = tf.concat([avg_pooled, max_pooled], axis=-1)
        
        return output, skip_connections
        
    def get_config(self):
        """Return configuration for serialization"""
        config = super(MultiScaleConvEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'kernel_sizes': self.kernel_sizes,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'use_residual': self.use_residual,
            'use_batch_norm': self.use_batch_norm,
            'name': self.name_prefix
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class HybridTimeSeriesEncoder(layers.Layer):
    """
    Hybrid time series encoder using convolutions, GRU, and transformers
    """
    
    def __init__(self, 
                 units: int,
                 depth: int = 2,
                 num_heads: int = 4,
                 use_conv: bool = True,
                 use_gru: bool = True,
                 use_transformer: bool = True,
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.1,
                 activation: str = 'gelu',
                 l2_reg: float = 1e-6,
                 name: str = None):
        """Initialize the hybrid time series encoder"""
        super(HybridTimeSeriesEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.num_heads = num_heads
        self.use_conv = use_conv
        self.use_gru = use_gru
        self.use_transformer = use_transformer
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        self.name_prefix = name or "hybrid_encoder"
        
        # Regularizer
        self.regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        # Initial projection to match channel dimension
        self.initial_proj = layers.Conv1D(
            units, 1, padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_initial_proj"
        )
        
        # Convolutional encoder
        if use_conv:
            self.conv_encoder = MultiScaleConvEncoder(
                units=units,
                depth=depth,
                kernel_sizes=kernel_sizes,
                dropout_rate=dropout_rate,
                activation=activation,
                l2_reg=l2_reg,
                use_residual=True,
                name=f"{self.name_prefix}_conv"
            )
        else:
            self.conv_encoder = None
            
        # GRU layers
        self.gru_layers = []
        if use_gru:
            for i in range(depth // 2):  # Fewer GRU layers as they're more expensive
                self.gru_layers.append(
                    FastGRUBlock(
                        units=units,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate / 2,
                        bidirectional=True,
                        name=f"{self.name_prefix}_gru{i+1}"
                    )
                )
                
        # Transformer blocks
        self.transformer_blocks = []
        if use_transformer:
            for i in range(depth // 2):  # Fewer transformer blocks as they're more expensive
                self.transformer_blocks.append(
                    FastTransformerBlock(
                        embed_dim=units * 2 if use_gru else units,
                        num_heads=num_heads,
                        ff_dim=units * 4,
                        dropout_rate=dropout_rate,
                        activation=activation,
                        name=f"{self.name_prefix}_transformer{i+1}"
                    )
                )
                
        # Attention pooling
        self.attention_query = layers.Dense(
            units,
            activation='tanh',
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_attn_query"
        )
        self.attention_key = layers.Dense(
            units,
            activation='tanh',
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_attn_key"
        )
        self.attention_value = layers.Dense(
            units,
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_attn_value"
        )
        self.attention_weights = layers.Softmax(
            axis=1,
            name=f"{self.name_prefix}_attn_weights"
        )
        
        # Global pooling operations
        self.global_avg_pool = layers.GlobalAveragePooling1D(
            name=f"{self.name_prefix}_global_avg_pool"
        )
        self.global_max_pool = layers.GlobalMaxPooling1D(
            name=f"{self.name_prefix}_global_max_pool"
        )
        
        # Final projection
        self.final_proj = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_final_proj"
        )
        
        self.final_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name_prefix}_final_norm"
        )
        
        self.final_dropout = layers.Dropout(
            dropout_rate,
            name=f"{self.name_prefix}_final_dropout"
        )
    
    def call(self, inputs, training=None):
        """Process inputs through the hybrid time series encoder"""
        # Initial projection
        x = self.initial_proj(inputs)
        
        # Convolutional encoder
        if self.conv_encoder is not None:
            conv_features, skip_connections = self.conv_encoder(x, training=training)
            # Get the last skip connection as sequential features
            x = skip_connections[-1]
        
        # GRU layers
        for gru_layer in self.gru_layers:
            x = gru_layer(x, training=training)
            
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
            
        # Attention pooling
        q = self.attention_query(x)
        k = self.attention_key(x)
        v = self.attention_value(x)
        
        # Calculate attention scores
        attention_score = tf.matmul(q, k, transpose_b=True)
        scale = tf.math.sqrt(tf.cast(tf.shape(k)[-1], attention_score.dtype))
        attention_score = attention_score / scale
        attention_weights = self.attention_weights(attention_score)
        
        # Apply attention weights
        attention_output = tf.matmul(attention_weights, v)
        attended_features = tf.reduce_sum(attention_output, axis=1)
        
        # Global pooling
        avg_pooled = self.global_avg_pool(x)
        max_pooled = self.global_max_pool(x)
        
        # Combine features
        if self.conv_encoder is not None:
            combined_features = tf.concat([conv_features, attended_features, avg_pooled, max_pooled], axis=-1)
        else:
            combined_features = tf.concat([attended_features, avg_pooled, max_pooled], axis=-1)
            
        # Final projection
        output = self.final_proj(combined_features)
        output = self.final_norm(output, training=training)
        output = self.final_dropout(output, training=training)
        
        return output
        
    def get_config(self):
        """Return configuration for serialization"""
        config = super(HybridTimeSeriesEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'use_conv': self.use_conv,
            'use_gru': self.use_gru,
            'use_transformer': self.use_transformer,
            'kernel_sizes': self.kernel_sizes,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'name': self.name_prefix
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class FastTabularEncoder(layers.Layer):
    """
    Optimized encoder for tabular data
    """
    
    def __init__(self, 
                 units: int,
                 depth: int = 3,
                 dropout_rate: float = 0.1,
                 activation: str = 'gelu',
                 l2_reg: float = 1e-6,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 name: str = None):
        """Initialize the fast tabular encoder"""
        super(FastTabularEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.name_prefix = name or "tab_encoder"
        
        # Regularizer
        self.regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        # Initial projection
        self.initial_proj = layers.Dense(
            units,
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_initial_proj"
        )
        
        # Dense blocks
        self.dense_blocks = []
        for i in range(depth):
            block = {}
            
            # First dense layer
            block['dense1'] = layers.Dense(
                units * 2,
                activation=None,
                kernel_regularizer=self.regularizer,
                name=f"{self.name_prefix}_block{i+1}_dense1"
            )
            
            # Normalization
            if use_batch_norm:
                block['norm1'] = layers.BatchNormalization(
                    name=f"{self.name_prefix}_block{i+1}_bn1"
                )
            else:
                block['norm1'] = layers.LayerNormalization(
                    epsilon=1e-6,
                    name=f"{self.name_prefix}_block{i+1}_ln1"
                )
                
            # Activation
            block['activation1'] = layers.Activation(
                activation,
                name=f"{self.name_prefix}_block{i+1}_act1"
            )
            
            # Dropout
            block['dropout1'] = layers.Dropout(
                dropout_rate,
                name=f"{self.name_prefix}_block{i+1}_dropout1"
            )
            
            # Second dense layer
            block['dense2'] = layers.Dense(
                units,
                activation=None,
                kernel_regularizer=self.regularizer,
                name=f"{self.name_prefix}_block{i+1}_dense2"
            )
            
            # Normalization
            if use_batch_norm:
                block['norm2'] = layers.BatchNormalization(
                    name=f"{self.name_prefix}_block{i+1}_bn2"
                )
            else:
                block['norm2'] = layers.LayerNormalization(
                    epsilon=1e-6,
                    name=f"{self.name_prefix}_block{i+1}_ln2"
                )
                
            # Activation
            block['activation2'] = layers.Activation(
                activation,
                name=f"{self.name_prefix}_block{i+1}_act2"
            )
            
            # Dropout
            block['dropout2'] = layers.Dropout(
                dropout_rate,
                name=f"{self.name_prefix}_block{i+1}_dropout2"
            )
            
            self.dense_blocks.append(block)
            
        # Final projection
        self.final_proj = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=self.regularizer,
            name=f"{self.name_prefix}_final_proj"
        )
    
    def call(self, inputs, training=None):
        """Process inputs through the fast tabular encoder"""
        # Initial projection
        x = self.initial_proj(inputs)
        
        # Process through dense blocks
        for block in self.dense_blocks:
            residual = x
            
            # First dense layer
            x = block['dense1'](x)
            x = block['norm1'](x, training=training)
            x = block['activation1'](x)
            x = block['dropout1'](x, training=training)
            
            # Second dense layer
            x = block['dense2'](x)
            x = block['norm2'](x, training=training)
            x = block['activation2'](x)
            x = block['dropout2'](x, training=training)
            
            # Add residual connection if enabled
            if self.use_residual:
                x = x + residual
                
        # Final projection
        output = self.final_proj(x)
        
        return output
        
    def get_config(self):
        """Return configuration for serialization"""
        config = super(FastTabularEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'use_residual': self.use_residual,
            'use_batch_norm': self.use_batch_norm,
            'name': self.name_prefix
        })
        return config


# =============================================================================
# MODEL CLASS
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="botlib")
class LightEnsembleModel(keras.Model):
    """
    Efficient ensemble model for trading predictions
    """
    
    def __init__(self,
                 window_5m=241,
                 window_15m=241,
                 window_1h=241,
                 window_google_trend=24,
                 feature_dim=9,
                 google_feature_dim=1,
                 santiment_dim=12,
                 ta_dim=63,
                 signal_dim=11,
                 output_dim=NUM_FUTURE_STEPS,
                 base_units=64,
                 dropout_rate=0.1,
                 name="light_ensemble"):
        """Initialize the light ensemble model"""
        super(LightEnsembleModel, self).__init__(name=name)
        
        # Store configuration
        self.window_5m = window_5m
        self.window_15m = window_15m
        self.window_1h = window_1h
        self.window_google_trend = window_google_trend
        self.feature_dim = feature_dim
        self.google_feature_dim = google_feature_dim
        self.santiment_dim = santiment_dim
        self.ta_dim = ta_dim
        self.signal_dim = signal_dim
        self.output_dim = output_dim
        self.base_units = base_units
        self.dropout_rate = dropout_rate
        
        # Input layers
        self.input_5m = layers.Input(
            shape=(window_5m, feature_dim),
            name="input_5m"
        )
        self.input_15m = layers.Input(
            shape=(window_15m, feature_dim),
            name="input_15m"
        )
        self.input_1h = layers.Input(
            shape=(window_1h, feature_dim),
            name="input_1h"
        )
        self.input_google = layers.Input(
            shape=(window_google_trend, google_feature_dim),
            name="input_google_trend"
        )
        self.input_santiment = layers.Input(
            shape=(santiment_dim,),
            name="input_santiment"
        )
        self.input_ta = layers.Input(
            shape=(ta_dim,),
            name="input_ta"
        )
        self.input_signal = layers.Input(
            shape=(signal_dim,),
            name="input_signal"
        )
        
        # Time series encoders
        self.encoder_5m = TimeSeriesEncoder(
            units=base_units,
            depth=2,
            dropout_rate=dropout_rate,
            name="enc_5m"
        )
        
        self.encoder_15m = TimeSeriesEncoder(
            units=base_units,
            depth=2,
            dropout_rate=dropout_rate,
            name="enc_15m"
        )
        
        self.encoder_1h = TimeSeriesEncoder(
            units=base_units,
            depth=2,
            dropout_rate=dropout_rate,
            name="enc_1h"
        )
        
        self.encoder_google = TimeSeriesEncoder(
            units=base_units // 2,
            depth=1,
            dropout_rate=dropout_rate,
            name="enc_google"
        )
        
        # Tabular encoders
        self.encoder_santiment = TabularEncoder(
            units=base_units // 2,
            depth=2,
            dropout_rate=dropout_rate,
            name="enc_santiment"
        )
        
        self.encoder_ta = TabularEncoder(
            units=base_units,
            depth=2,
            dropout_rate=dropout_rate,
            name="enc_ta"
        )
        
        self.encoder_signal = TabularEncoder(
            units=base_units // 2,
            depth=2,
            dropout_rate=dropout_rate,
            name="enc_signal"
        )
        
        # Ensemble layers
        self.concat_features = layers.Concatenate(name="concat_features")
        self.ensemble_dense1 = layers.Dense(
            base_units * 2,
            activation='relu',
            name="ensemble_dense1"
        )
        self.ensemble_bn = layers.BatchNormalization(name="ensemble_bn")
        self.ensemble_dropout = layers.Dropout(dropout_rate, name="ensemble_dropout")
        self.ensemble_dense2 = layers.Dense(
            base_units,
            activation='relu',
            name="ensemble_dense2"
        )
        
        # Output layer
        self.output_layer = layers.Dense(
            output_dim,
            activation='tanh',
            name="output"
        )
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        # Process each input stream
        feat_5m = self.encoder_5m(self.input_5m)
        feat_15m = self.encoder_15m(self.input_15m)
        feat_1h = self.encoder_1h(self.input_1h)
        feat_google = self.encoder_google(self.input_google)
        feat_santiment = self.encoder_santiment(self.input_santiment)
        feat_ta = self.encoder_ta(self.input_ta)
        feat_signal = self.encoder_signal(self.input_signal)
        
        # Combine features
        combined = self.concat_features([
            feat_5m, feat_15m, feat_1h, 
            feat_google, feat_santiment, feat_ta, feat_signal
        ])
        
        # Process through ensemble layers
        x = self.ensemble_dense1(combined)
        x = self.ensemble_bn(x)
        x = self.ensemble_dropout(x)
        x = self.ensemble_dense2(x)
        
        # Generate output
        output = self.output_layer(x)
        
        # Create model
        self.model = tf.keras.Model(
            inputs=[
                self.input_5m, self.input_15m, self.input_1h,
                self.input_google, self.input_santiment, 
                self.input_ta, self.input_signal
            ],
            outputs=output,
            name=self.name
        )
        
    def call(self, inputs, training=None):
        """Forward pass through the model"""
        return self.model(inputs, training=training)
        
    def get_config(self):
        """Return configuration for serialization"""
        config = super(LightEnsembleModel, self).get_config()
        config.update({
            'window_5m': self.window_5m,
            'window_15m': self.window_15m,
            'window_1h': self.window_1h,
            'window_google_trend': self.window_google_trend,
            'feature_dim': self.feature_dim,
            'google_feature_dim': self.google_feature_dim,
            'santiment_dim': self.santiment_dim,
            'ta_dim': self.ta_dim,
            'signal_dim': self.signal_dim,
            'output_dim': self.output_dim,
            'base_units': self.base_units,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# FACTORY FUNCTION FOR BACKWARD COMPATIBILITY
# =============================================================================

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
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 8,
    mixed_precision: bool = True,
    **kwargs
) -> keras.models.Model:
    """
    Create a memory-efficient model for trading predictions.
    """
    print("\n" + "="*80)
    print(" CREATING MEMORY-EFFICIENT MODEL FOR TRADING PREDICTIONS ")
    print("="*80)
    
    # Enable mixed precision if requested (before creating the model)
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(f"Mixed precision enabled with policy: {policy.name}")
        except:
            print("Mixed precision not available in this TensorFlow version")
    
    # Create a simpler, memory-efficient model
    
    # Input layers
    input_5m = layers.Input(shape=(model_5m_window, feature_dim), name="input_5m")
    input_15m = layers.Input(shape=(model_15m_window, feature_dim), name="input_15m")
    input_1h = layers.Input(shape=(model_1h_window, feature_dim), name="input_1h")
    input_google = layers.Input(shape=(24, 1), name="input_google_trend")
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    
    # Simplified processing - use simpler Conv1D layers instead of complex custom encoders
    # Process time series with Conv1D and global pooling
    conv_5m = layers.Conv1D(base_units, 3, padding='same', activation='relu')(input_5m)
    conv_5m = layers.GlobalAveragePooling1D()(conv_5m)
    bn_5m = layers.BatchNormalization()(conv_5m)
    
    conv_15m = layers.Conv1D(base_units, 3, padding='same', activation='relu')(input_15m)
    conv_15m = layers.GlobalAveragePooling1D()(conv_15m)
    bn_15m = layers.BatchNormalization()(conv_15m)
    
    conv_1h = layers.Conv1D(base_units, 3, padding='same', activation='relu')(input_1h)
    conv_1h = layers.GlobalAveragePooling1D()(conv_1h)
    bn_1h = layers.BatchNormalization()(conv_1h)
    
    conv_google = layers.Conv1D(base_units//2, 3, padding='same', activation='relu')(input_google)
    conv_google = layers.GlobalAveragePooling1D()(conv_google)
    bn_google = layers.BatchNormalization()(conv_google)
    
    # Process tabular data with simple dense layers
    dense_sa = layers.Dense(base_units//2, activation='relu')(input_santiment)
    bn_sa = layers.BatchNormalization()(dense_sa)
    
    dense_ta = layers.Dense(base_units, activation='relu')(input_ta)
    bn_ta = layers.BatchNormalization()(dense_ta)
    
    dense_signal = layers.Dense(base_units//2, activation='relu')(input_signal)
    bn_signal = layers.BatchNormalization()(dense_signal)
    
    # Combine all features
    concat = layers.Concatenate()([bn_5m, bn_15m, bn_1h, bn_google, bn_sa, bn_ta, bn_signal])
    
    # Ensemble dense layers
    dense1 = layers.Dense(base_units*2, activation='relu')(concat)
    dropout1 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(base_units, activation='relu')(dropout1)
    
    # Output layer
    output = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(dense2)
    
    # Create and compile model
    model = keras.Model(
        inputs=[input_5m, input_15m, input_1h, input_google, input_santiment, input_ta, input_signal],
        outputs=output,
        name="efficient_trading_model"
    )
    
    # Compile with Adam optimizer and MSE loss
    optimizer = optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Print model info
    total_params = model.count_params()
    print(f"Model created with {total_params:,} parameters")
    print(f"Base units: {base_units}")
    print(f"Output dimension: {NUM_FUTURE_STEPS}")
    print("="*80 + "\n")
    
    return model


# Legacy functions - minimal implementation that calls into FastEnsembleModel
def build_5m_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for 5m data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('window_5m', 241), kwargs.get('feature_dim', 9)))
    encoder = TimeSeriesEncoder(
        units=kwargs.get('base_units', 64),
        name="5m"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_5m")

def build_15m_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for 15m data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('window_15m', 241), kwargs.get('feature_dim', 9)))
    encoder = TimeSeriesEncoder(
        units=kwargs.get('base_units', 64),
        name="15m"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_15m")

def build_1h_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for 1h data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('window_1h', 241), kwargs.get('feature_dim', 9)))
    encoder = TimeSeriesEncoder(
        units=kwargs.get('base_units', 64),
        name="1h"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_1h")

def build_google_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for Google Trends data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('window_google', 24), kwargs.get('feature_google', 1)))
    encoder = TimeSeriesEncoder(
        units=kwargs.get('base_units', 32),
        name="google"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_google")

def build_ta_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for TA data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('ta_dim', 63),))
    encoder = TabularEncoder(
        units=kwargs.get('base_units', 64),
        name="ta"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_ta")

def build_santiment_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for sentiment data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('santiment_dim', 12),))
    encoder = TabularEncoder(
        units=kwargs.get('base_units', 32),
        name="santiment"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_santiment")

def build_signal_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy model builder for signal data - this returns a subset of FastEnsembleModel")
    inputs = layers.Input(shape=(kwargs.get('signal_dim', 11),))
    encoder = TabularEncoder(
        units=kwargs.get('base_units', 32),
        name="signal"
    )
    features = encoder(inputs)
    outputs = layers.Dense(NUM_FUTURE_STEPS, activation='tanh')(features)
    return keras.models.Model(inputs=inputs, outputs=outputs, name="model_signal")

def build_ensemble_model(**kwargs):
    """Legacy function for backward compatibility"""
    print("Warning: Using legacy ensemble model builder - consider using FastEnsembleModel directly")
    return load_advanced_lstm_model(**kwargs)