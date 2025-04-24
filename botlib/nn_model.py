"""
Advanced Ensemble Neural Network Model for Multi-timeframe Trading Predictions

Significantly enhanced implementation with:
- Massive parameter scaling for 500MB+ model size
- Advanced transformer architecture with multi-head attention
- Multiple parallel processing pathways with residual connections
- Deep feature extraction with sophisticated regularization
- Memory-optimized training for large model handling
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
class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention mechanism for time series data"""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 use_bias: bool = True,
                 name: str = None):
        super(MultiHeadSelfAttention, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
        
        self.depth = embed_dim // num_heads
        
        self.query_dense = layers.Dense(embed_dim, use_bias=use_bias)
        self.key_dense = layers.Dense(embed_dim, use_bias=use_bias)
        self.value_dense = layers.Dense(embed_dim, use_bias=use_bias)
        
        self.attention_dropout = layers.Dropout(dropout_rate)
        self.output_dense = layers.Dense(embed_dim, use_bias=use_bias)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        queries = self.query_dense(inputs)
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)
        
        # Split heads
        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = self._scaled_dot_product_attention(
            queries, keys, values, mask, training
        )
        
        # Transpose and reshape
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))
        
        # Final linear projection
        output = self.output_dense(concat_attention)
        
        return output
    
    def _scaled_dot_product_attention(self, q, k, v, mask, training):
        """Calculate the attention weights"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk - FIX for mixed precision
        dk = tf.shape(k)[-1]
        # Cast to match q's dtype instead of hardcoding to float32
        dk = tf.cast(dk, q.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.attention_dropout(attention_weights, training=training)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'name': self.name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="botlib")
class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network with mixed precision support"""
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_dim: int,
                 dropout_rate: float = 0.1,
                 use_bias: bool = True,
                 name: str = None):
        super(TransformerBlock, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name=f"{name}_mha" if name else None
        )
        
        self.ffn1 = layers.Dense(ff_dim, activation='gelu', use_bias=use_bias)
        self.ffn2 = layers.Dense(embed_dim, use_bias=use_bias)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention with residual connection and layer normalization
        attn_output = self.attention(inputs, training=training, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        
        # Make sure datatypes match for addition
        if attn_output.dtype != inputs.dtype:
            attn_output = tf.cast(attn_output, inputs.dtype)
            
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network with residual connection and layer normalization
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Make sure datatypes match for addition
        if ffn_output.dtype != out1.dtype:
            ffn_output = tf.cast(ffn_output, out1.dtype)
            
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'name': self.name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="botlib")
class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer inputs using TensorFlow operations"""
    
    def __init__(self, max_length, embed_dim, name=None):
        super(PositionalEncoding, self).__init__(name=name)
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Create fixed positional encodings for common dimensions
        self._create_fixed_encodings()
        
    def _create_fixed_encodings(self):
        """Pre-compute positional encodings for common dimensions"""
        # Create encodings for powers of 2 up to 4096
        self.fixed_encodings = {}
        dimensions = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        for dim in dimensions:
            positions = np.arange(self.max_length)[:, np.newaxis]
            div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
            
            pos_encoding = np.zeros((self.max_length, dim))
            pos_encoding[:, 0::2] = np.sin(positions * div_term)
            pos_encoding[:, 1::2] = np.cos(positions * div_term)
            
            # Add batch dimension
            pos_encoding = pos_encoding[np.newaxis, ...]
            
            # Store as TensorFlow constant
            self.fixed_encodings[dim] = tf.constant(pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        """Apply positional encoding using the closest pre-computed encoding"""
        # Get shape info at runtime
        input_shape = tf.shape(inputs)
        seq_length = input_shape[1]
        embed_dim = input_shape[2]
        
        # Use static dimension when available
        if hasattr(inputs, 'shape') and inputs.shape[2] is not None:
            static_dim = inputs.shape[2]
        else:
            # Use dynamic dimension
            static_dim = embed_dim
            
        # Convert to Python int during eager execution, or use closest fixed dimension
        if tf.executing_eagerly():
            dim_value = int(static_dim)
            # Find closest power of 2
            closest_dim = min(self.fixed_encodings.keys(), key=lambda x: abs(x - dim_value))
        else:
            # During graph building, use the expected dimension based on model config
            # For safety, we'll use the closest from our fixed dimensions
            if self.embed_dim in self.fixed_encodings:
                closest_dim = self.embed_dim
            else:
                closest_dim = min(self.fixed_encodings.keys(), key=lambda x: abs(x - self.embed_dim))
        
        # Get pre-computed encoding
        pos_encoding = self.fixed_encodings[closest_dim]
        
        # If dimensions don't match exactly, we need to adapt
        if closest_dim != static_dim:
            if tf.executing_eagerly():
                # During eager execution we can resize
                if closest_dim > static_dim:
                    # Truncate
                    pos_encoding = pos_encoding[:, :, :static_dim]
                else:
                    # Pad (repeat the pattern)
                    pad_size = static_dim - closest_dim
                    padding = tf.repeat(pos_encoding[:, :, :pad_size], tf.constant([1]), axis=2)
                    pos_encoding = tf.concat([pos_encoding, padding], axis=2)
            else:
                # During graph building, log a warning but continue
                tf.print("Warning: Positional encoding dimension mismatch. Using closest available.")
        
        # Cast to input dtype
        pos_encoding_cast = tf.cast(pos_encoding, dtype=inputs.dtype)
        
        # Apply slice for correct sequence length
        pos_encoding_slice = pos_encoding_cast[:, :seq_length, :]
        
        # Add to input
        return inputs + pos_encoding_slice
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_length': self.max_length,
            'embed_dim': self.embed_dim,
            'name': self.name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="botlib")
class FastGRUBlock(layers.Layer):
    """Optimized GRU block with performance enhancements for trading data"""
    
    def __init__(self, 
                 units: int,
                 dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional: bool = True,
                 name: str = None):
        super(FastGRUBlock, self).__init__(name=name)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        
        # Use CuDNN implementation when possible
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
        x = self.gru(inputs, training=training, mask=mask)
        return self.layer_norm(x, training=training)
    
    def get_config(self):
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
class ConvBlock(layers.Layer):
    """Convolutional block with multiple filter sizes for time series feature extraction"""
    
    def __init__(self,
                 filters: int,
                 kernel_sizes: List[int] = [1, 3, 5, 7],
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 name: str = None):
        super(ConvBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Create parallel convolutional layers with different kernel sizes
        self.conv_layers = []
        for kernel_size in kernel_sizes:
            conv = layers.Conv1D(
                filters=filters // len(kernel_sizes),
                kernel_size=kernel_size,
                padding='same',
                activation=None,
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"{name}_conv{kernel_size}" if name else None
            )
            self.conv_layers.append(conv)
        
        # Batch normalization and activation
        if use_batch_norm:
            self.batch_norm = layers.BatchNormalization(name=f"{name}_bn" if name else None)
        
        self.activation_layer = layers.Activation(activation, name=f"{name}_act" if name else None)
        self.dropout = layers.SpatialDropout1D(dropout_rate, name=f"{name}_drop" if name else None)
        
    def call(self, inputs, training=None):
        # Apply parallel convolutions
        conv_outputs = [conv(inputs) for conv in self.conv_layers]
        
        # Concatenate outputs
        x = layers.concatenate(conv_outputs, axis=-1)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
        
        # Apply activation and dropout
        x = self.activation_layer(x)
        x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'name': self.name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="botlib")
class ResidualBlock(layers.Layer):
    """Residual block for deep networks"""
    
    def __init__(self,
                 units: int,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 name: str = None):
        super(ResidualBlock, self).__init__(name=name)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # First dense layer
        self.dense1 = layers.Dense(
            units*2,
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{name}_dense1" if name else None
        )
        
        # Second dense layer (for residual path)
        self.dense2 = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{name}_dense2" if name else None
        )
        
        # Projection for input if needed
        self.projection = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{name}_proj" if name else None
        )
        
        # Batch normalization layers
        if use_batch_norm:
            self.bn1 = layers.BatchNormalization(name=f"{name}_bn1" if name else None)
            self.bn2 = layers.BatchNormalization(name=f"{name}_bn2" if name else None)
        
        # Activation and dropout
        self.activation1 = layers.Activation(activation, name=f"{name}_act1" if name else None)
        self.activation2 = layers.Activation(activation, name=f"{name}_act2" if name else None)
        self.dropout1 = layers.Dropout(dropout_rate, name=f"{name}_drop1" if name else None)
        self.dropout2 = layers.Dropout(dropout_rate, name=f"{name}_drop2" if name else None)
        
    def call(self, inputs, training=None):
        # Project input if needed
        input_projection = self.projection(inputs)
        
        # First dense layer
        x = self.dense1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        # Second dense layer
        x = self.dense2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        
        # Add residual connection
        x = x + input_projection
        
        # Final activation and dropout
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        
        return x
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'name': self.name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="botlib")
class TimeSeriesEncoder(layers.Layer):
    """Memory-efficient time series encoder with advanced architecture"""
    
    def __init__(self, 
                 units: int = 64,
                 depth: int = 2,
                 dropout_rate: float = 0.1,
                 use_transformer: bool = True,
                 use_conv: bool = True,
                 use_gru: bool = True,
                 name: str = None):
        super(TimeSeriesEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_transformer = use_transformer
        self.use_conv = use_conv
        self.use_gru = use_gru
        self.name_prefix = name or "time_encoder"
        
        # Initial projection with 1D convolution for efficiency
        self.initial_conv = layers.Conv1D(
            units, 
            kernel_size=3, 
            padding='same',
            activation='relu',
            name=f"{self.name_prefix}_init_conv"
        )
        
        # Convolutional blocks for feature extraction
        if use_conv:
            self.conv_blocks = []
            for i in range(depth):
                conv_block = ConvBlock(
                    filters=units * 2,
                    kernel_sizes=[1, 3, 5, 7],
                    dropout_rate=dropout_rate,
                    name=f"{self.name_prefix}_conv_block{i+1}"
                )
                self.conv_blocks.append(conv_block)
        
        # Transformer blocks for sequential relationships
        if use_transformer:
            # Note: We initialize with a default embed_dim, but it will adapt in call()
            self.positional_encoding = PositionalEncoding(
                max_length=1000,  # Large enough for most sequences
                embed_dim=units * 2,  # This will be dynamically updated
                name=f"{self.name_prefix}_pos_enc"
            )
            
            self.transformer_blocks = []
            for i in range(depth):
                transformer = TransformerBlock(
                    embed_dim=units * 2,  # Will need to match the conv output
                    num_heads=8,
                    ff_dim=units * 4,
                    dropout_rate=dropout_rate,
                    name=f"{self.name_prefix}_transformer{i+1}"
                )
                self.transformer_blocks.append(transformer)
        
        # GRU layers for sequential patterns
        if use_gru:
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
        
        # Global pooling - combine both average and max for better feature extraction
        self.global_avg_pool = layers.GlobalAveragePooling1D(
            name=f"{self.name_prefix}_avg_pool"
        )
        self.global_max_pool = layers.GlobalMaxPooling1D(
            name=f"{self.name_prefix}_max_pool"
        )
        
        # Attention pooling
        self.attention_query = layers.Dense(
            units, 
            activation='tanh',
            name=f"{self.name_prefix}_attn_query"
        )
        self.attention_value = layers.Dense(
            1, 
            activation=None,
            name=f"{self.name_prefix}_attn_value"
        )
        
        # Combine different representations
        self.final_dense = layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{self.name_prefix}_final"
        )
        
    def call(self, inputs, training=None):
        # Initial processing
        x = self.initial_conv(inputs)
        
        # Store original for residual
        original_x = x
        
        # Apply convolutional blocks
        if self.use_conv:
            for conv_block in self.conv_blocks:
                x = conv_block(x, training=training)
            
            # Add residual if dimensions match
            if original_x.shape[-1] == x.shape[-1]:
                # Ensure dtype consistency for addition
                if original_x.dtype != x.dtype:
                    original_x = tf.cast(original_x, x.dtype)
                x = x + original_x
        
        # Apply transformer blocks
        if self.use_transformer:
            # Add positional encoding - this will now dynamically adapt to input dimension
            x_transformer = self.positional_encoding(x)
            
            # Make sure transformer input has consistent dtype
            if x_transformer.dtype != x.dtype:
                x_transformer = tf.cast(x_transformer, x.dtype)
            
            # Apply transformer blocks
            for transformer in self.transformer_blocks:
                x_transformer = transformer(x_transformer, training=training)
            
            # Use transformer output
            x = x_transformer
        
        # Apply GRU layers
        if self.use_gru:
            x_gru = x
            for gru_layer in self.gru_layers:
                x_gru = gru_layer(x_gru, training=training)
            
            # Use GRU output
            x = x_gru
        
        # Global pooling - combine different types for better feature extraction
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Attention pooling
        attn_weights = self.attention_query(x)
        attn_weights = self.attention_value(attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=1)
        
        # Ensure consistent dtype for multiplication
        if attn_weights.dtype != x.dtype:
            attn_weights = tf.cast(attn_weights, x.dtype)
            
        context_vector = tf.reduce_sum(x * attn_weights, axis=1)
        
        # Combine pooling results for richer feature representation
        # Ensure all tensors have the same dtype before concatenation
        dtype = avg_pool.dtype
        if max_pool.dtype != dtype:
            max_pool = tf.cast(max_pool, dtype)
        if context_vector.dtype != dtype:
            context_vector = tf.cast(context_vector, dtype)
            
        combined = tf.concat([avg_pool, max_pool, context_vector], axis=-1)
        
        # Final projection
        output = self.final_dense(combined)
        
        return output
    
    def get_config(self):
        config = super(TimeSeriesEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'use_transformer': self.use_transformer,
            'use_conv': self.use_conv,
            'use_gru': self.use_gru,
            'name': self.name_prefix
        })
        return config


@tf.keras.utils.register_keras_serializable(package="botlib")
class TabularEncoder(layers.Layer):
    """Enhanced encoder for tabular features with residual connections"""
    
    def __init__(self, 
                 units: int = 64,
                 depth: int = 3,
                 dropout_rate: float = 0.2,
                 use_residual: bool = True,
                 name: str = None):
        super(TabularEncoder, self).__init__(name=name)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.name_prefix = name or "tab_encoder"
        
        # Initial projection
        self.initial_projection = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"{self.name_prefix}_init_proj"
        )
        self.initial_bn = layers.BatchNormalization(name=f"{self.name_prefix}_init_bn")
        self.initial_act = layers.Activation('relu', name=f"{self.name_prefix}_init_act")
        
        # Residual blocks
        if use_residual:
            self.residual_blocks = []
            for i in range(depth):
                res_block = ResidualBlock(
                    units=units,
                    dropout_rate=dropout_rate,
                    name=f"{self.name_prefix}_res{i+1}"
                )
                self.residual_blocks.append(res_block)
        else:
            # Dense layers with normalization and dropout
            self.dense_layers = []
            for i in range(depth):
                # First dense layer in the block
                layer_units = units * 2 if i < depth - 1 else units
                dense = layers.Dense(
                    layer_units,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-5),
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
        # Initial projection
        x = self.initial_projection(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_act(x)
        
        # Process through residual or dense layers
        if self.use_residual:
            for block in self.residual_blocks:
                x = block(x, training=training)
        else:
            for layer in self.dense_layers:
                x = layer(x, training=training)
            
        return x
    
    def get_config(self):
        config = super(TabularEncoder, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
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


# =============================================================================
# MASSIVELY SCALED UP MODEL CLASS - For 500MB+ model size
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="botlib")
class StackLayer(layers.Layer):
    """Custom layer to stack tensors along a specified axis"""
    
    def __init__(self, axis=1, name=None):
        super(StackLayer, self).__init__(name=name)
        self.axis = axis
        
    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)
        
    def get_config(self):
        config = super(StackLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'name': self.name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="botlib")
class MassiveEnsembleModel(keras.Model):
    """
    Massive ensemble model for trading predictions with 500MB+ parameter size
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
                 base_units=512,  # Significantly increased from 64-96
                 depth=6,  # Increased depth for more parameters
                 dropout_rate=0.2,
                 name="massive_ensemble"):
        """Initialize the massive ensemble model"""
        super(MassiveEnsembleModel, self).__init__(name=name)
        
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
        self.depth = depth
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
        
        # Advanced time series encoders with massive capacity
        self.encoder_5m = TimeSeriesEncoder(
            units=base_units,
            depth=depth,
            dropout_rate=dropout_rate,
            use_transformer=True,
            use_conv=True,
            use_gru=True,
            name="enc_5m"
        )
        
        self.encoder_15m = TimeSeriesEncoder(
            units=base_units,
            depth=depth,
            dropout_rate=dropout_rate,
            use_transformer=True, 
            use_conv=True,
            use_gru=True,
            name="enc_15m"
        )
        
        self.encoder_1h = TimeSeriesEncoder(
            units=base_units,
            depth=depth,
            dropout_rate=dropout_rate,
            use_transformer=True,
            use_conv=True,
            use_gru=True,
            name="enc_1h"
        )
        
        self.encoder_google = TimeSeriesEncoder(
            units=base_units // 2,
            depth=3,
            dropout_rate=dropout_rate,
            use_transformer=True,
            use_conv=True,
            use_gru=False,
            name="enc_google"
        )
        
        # Advanced tabular encoders with residual connections
        self.encoder_santiment = TabularEncoder(
            units=base_units // 2,
            depth=4,
            dropout_rate=dropout_rate,
            use_residual=True,
            name="enc_santiment"
        )
        
        self.encoder_ta = TabularEncoder(
            units=base_units,
            depth=4,
            dropout_rate=dropout_rate,
            use_residual=True,
            name="enc_ta"
        )
        
        self.encoder_signal = TabularEncoder(
            units=base_units // 2,
            depth=4,
            dropout_rate=dropout_rate,
            use_residual=True,
            name="enc_signal"
        )
        
        # Custom layer for stacking
        self.stack_layer = StackLayer(axis=1, name="time_features_stack")
        
        # Cross-attention between time series encodings
        self.cross_attention = MultiHeadSelfAttention(
            embed_dim=base_units,
            num_heads=8,
            dropout_rate=dropout_rate,
            name="cross_attention"
        )
        
        # Ensemble layers
        self.concat_features = layers.Concatenate(name="concat_features")
        
        # Deep ensemble network with stacked residual blocks
        self.ensemble_blocks = []
        for i in range(depth):
            res_block = ResidualBlock(
                units=base_units * 2,
                dropout_rate=dropout_rate,
                name=f"ensemble_res{i+1}"
            )
            self.ensemble_blocks.append(res_block)
        
        # Transformer block for final feature processing
        self.ensemble_transformer = TransformerBlock(
            embed_dim=base_units * 2,
            num_heads=8,
            ff_dim=base_units * 4,
            dropout_rate=dropout_rate,
            name="ensemble_transformer"
        )
        
        # Reshape layer for transformer
        self.reshape_layer = layers.Reshape((1, base_units * 2), name="reshape_for_transformer")
        
        # Flatten layer after transformer
        self.flatten_layer = layers.Flatten(name="flatten_after_transformer")
        
        # Multiple prediction heads for different time horizons
        self.prediction_heads = []
        for i in range(output_dim):
            head = layers.Dense(
                1,
                activation='tanh',  # tanh for [-1,1] range
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"output_head{i+1}"
            )
            self.prediction_heads.append(head)
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        # Process each input stream with advanced encoders
        feat_5m = self.encoder_5m(self.input_5m)
        feat_15m = self.encoder_15m(self.input_15m)
        feat_1h = self.encoder_1h(self.input_1h)
        feat_google = self.encoder_google(self.input_google)
        feat_santiment = self.encoder_santiment(self.input_santiment)
        feat_ta = self.encoder_ta(self.input_ta)
        feat_signal = self.encoder_signal(self.input_signal)
        
        # Apply cross-attention between time series features - use our custom stack layer
        time_features = self.stack_layer([feat_5m, feat_15m, feat_1h])
        cross_attended = self.cross_attention(time_features)
        
        # Use proper Keras operations for reduction
        cross_features = layers.GlobalAveragePooling1D()(cross_attended)
        
        # Combine all features
        combined = self.concat_features([
            cross_features, feat_google, feat_santiment, feat_ta, feat_signal
        ])
        
        # Process through deep ensemble network
        x = combined
        for block in self.ensemble_blocks:
            x = block(x)
            
        # Reshape for transformer (add sequence dimension) using proper Keras layers
        x_transformer = self.reshape_layer(x)
        x_transformer = self.ensemble_transformer(x_transformer)
        x = self.flatten_layer(x_transformer)
        
        # Generate outputs from multiple prediction heads
        outputs = []
        for head in self.prediction_heads:
            out = head(x)
            outputs.append(out)
            
        # Concatenate all outputs
        output = self.concat_features(outputs)
        
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
        
        # Compile with robust loss
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,  # Gradient clipping for stability
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=safe_mse_loss,
            metrics=['mae']
        )
        
    def call(self, inputs, training=None):
        """Forward pass through the model"""
        return self.model(inputs, training=training)
        
    def get_config(self):
        """Return configuration for serialization"""
        config = super(MassiveEnsembleModel, self).get_config()
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
            'depth': self.depth,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# FACTORY FUNCTION FOR BACKWARD COMPATIBILITY
# =============================================================================

def build_ensemble_model(
    model_5m_window: int = 241,
    model_15m_window: int = 241,
    model_1h_window: int = 241,
    feature_dim: int = 9,
    santiment_dim: int = 12,
    ta_dim: int = 63,
    signal_dim: int = 11,
    base_units: int = 512,  # Significantly increased
    depth: int = 6,  # New parameter for depth control
    memory_efficient: bool = True,
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 8,
    mixed_precision: bool = True,
    massive_model: bool = True,  # New parameter to enable the massive model
    **kwargs
) -> keras.models.Model:
    """
    Create a memory-efficient, massive-scale model for trading predictions.
    
    Args:
        massive_model: If True, use the MassiveEnsembleModel for 500MB+ size
    """
    print("\n" + "="*80)
    print(" CREATING MASSIVELY SCALED TRADING MODEL ")
    print("="*80)
    
    # Enable mixed precision if requested (before creating the model)
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(f"Mixed precision enabled with policy: {policy.name}")
        except Exception as e:
            print(f"Mixed precision not available in this TensorFlow version: {e}")
    
    if massive_model:
        # Use the new massive model architecture
        model = MassiveEnsembleModel(
            window_5m=model_5m_window,
            window_15m=model_15m_window, 
            window_1h=model_1h_window,
            window_google_trend=24,
            feature_dim=feature_dim,
            google_feature_dim=1,
            santiment_dim=santiment_dim,
            ta_dim=ta_dim,
            signal_dim=signal_dim,
            output_dim=NUM_FUTURE_STEPS,
            base_units=base_units,
            depth=depth,
            dropout_rate=0.2
        ).model
        
        # Print model info
        total_params = model.count_params()
        print(f"Massive model created with {total_params:,} parameters")
        print(f"Base units: {base_units}, Depth: {depth}")
        print(f"Output dimension: {NUM_FUTURE_STEPS}")
        print(f"Approximate model size: {total_params * 4 / (1024 * 1024):.2f} MB")
        print("="*80 + "\n")
        
        return model
    
    # Legacy implementation with increased parameters (legacy support)
    # Input layers
    input_5m = layers.Input(shape=(model_5m_window, feature_dim), name="input_5m")
    input_15m = layers.Input(shape=(model_15m_window, feature_dim), name="input_15m")
    input_1h = layers.Input(shape=(model_1h_window, feature_dim), name="input_1h")
    input_google = layers.Input(shape=(24, 1), name="input_google_trend")
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    
    # Process time series with multi-scale convolutions + attention
    def process_timeseries(inputs, name):
        # Multi-scale convolutions
        conv1 = layers.Conv1D(base_units//4, 1, padding='same', activation='relu')(inputs)
        conv3 = layers.Conv1D(base_units//4, 3, padding='same', activation='relu')(inputs)
        conv5 = layers.Conv1D(base_units//4, 5, padding='same', activation='relu')(inputs)
        conv7 = layers.Conv1D(base_units//4, 7, padding='same', activation='relu')(inputs)
        
        # Concatenate multi-scale features
        multi_scale = layers.Concatenate()([conv1, conv3, conv5, conv7])
        multi_scale = layers.BatchNormalization()(multi_scale)
        
        # Self-attention mechanism
        attention_query = layers.Dense(base_units//2, activation='tanh')(multi_scale)
        attention_key = layers.Dense(1)(attention_query)
        attention_weights = layers.Softmax(axis=1)(attention_key)
        context_vector = layers.Multiply()([multi_scale, attention_weights])
        context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        
        # Also use global pooling
        avg_pool = layers.GlobalAveragePooling1D()(multi_scale)
        max_pool = layers.GlobalMaxPooling1D()(multi_scale)
        
        # Combine features
        combined = layers.Concatenate()([context_vector, avg_pool, max_pool])
        return layers.Dense(base_units, activation='relu')(combined)
    
    # Enhanced time series processing
    feat_5m = process_timeseries(input_5m, "5m")
    feat_15m = process_timeseries(input_15m, "15m")
    feat_1h = process_timeseries(input_1h, "1h")
    
    # Process Google trends (simpler)
    conv_google = layers.Conv1D(base_units//2, 3, padding='same', activation='relu')(input_google)
    pool_google = layers.GlobalAveragePooling1D()(conv_google)
    bn_google = layers.BatchNormalization()(pool_google)
    
    # Process tabular data with deep residual blocks
    def create_residual_block(inputs, units):
        x = layers.Dense(units*2, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        residual = layers.Dense(units, activation=None)(x)
        
        x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual connection
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        return x
    
    # Deep residual processing for tabular data    
    x_sa = layers.Dense(base_units//2, activation='relu')(input_santiment)
    x_sa = layers.BatchNormalization()(x_sa)
    x_sa = create_residual_block(x_sa, base_units//2)
    
    x_ta = layers.Dense(base_units, activation='relu')(input_ta)
    x_ta = layers.BatchNormalization()(x_ta)
    x_ta = create_residual_block(x_ta, base_units)
    
    x_signal = layers.Dense(base_units//2, activation='relu')(input_signal)
    x_signal = layers.BatchNormalization()(x_signal)
    x_signal = create_residual_block(x_signal, base_units//2)
    
    # Combine all features with proper Keras layers
    # Create a custom layer for stacking
    class StackLayer(layers.Layer):
        def __init__(self, axis=1, **kwargs):
            super(StackLayer, self).__init__(**kwargs)
            self.axis = axis
            
        def call(self, inputs):
            return tf.stack(inputs, axis=self.axis)
            
        def get_config(self):
            config = super(StackLayer, self).get_config()
            config.update({'axis': self.axis})
            return config
    
    # Use proper Keras layers for these operations
    time_features = StackLayer(axis=1)([feat_5m, feat_15m, feat_1h])
    
    # Simple cross-attention implementation with attention
    attention_query = layers.Dense(base_units, activation='tanh')(time_features)
    attention_weights = layers.Dense(1)(attention_query)
    attention_weights = layers.Softmax(axis=1)(attention_weights)
    
    # Use proper Keras operation for multiplication and reduction
    cross_context = layers.Multiply()([time_features, attention_weights])
    cross_context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(cross_context)
    
    # Combine everything
    concat = layers.Concatenate()([cross_context, bn_google, x_sa, x_ta, x_signal])
    
    # Deep ensemble network with multiple residual blocks
    x = concat
    for _ in range(depth):
        x = create_residual_block(x, base_units*2)
    
    # Multiple output heads for different time horizons
    outputs = []
    for i in range(NUM_FUTURE_STEPS):
        head = layers.Dense(1, activation='tanh')(x)
        outputs.append(head)
    
    # Concatenate all outputs
    output = layers.Concatenate()(outputs)
    
    # Create and compile model
    model = keras.Model(
        inputs=[input_5m, input_15m, input_1h, input_google, input_santiment, input_ta, input_signal],
        outputs=output,
        name="optimized_massive_trading_model"
    )
    
    # Compile with Adam optimizer and safe MSE loss
    optimizer = optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,  # Add gradient clipping for stability
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss=safe_mse_loss,  # Use custom loss function
        metrics=['mae']
    )
    
    # Print model info
    total_params = model.count_params()
    print(f"Legacy model created with {total_params:,} parameters")
    print(f"Base units: {base_units}, Depth: {depth}")
    print(f"Output dimension: {NUM_FUTURE_STEPS}")
    print(f"Approximate model size: {total_params * 4 / (1024 * 1024):.2f} MB")
    print("="*80 + "\n")
    
    return model


# Legacy functions that call into build_ensemble_model for backward compatibility
def load_advanced_lstm_model(
    model_5m_window: int = 241,
    model_15m_window: int = 241,
    model_1h_window: int = 241,
    feature_dim: int = 9,
    santiment_dim: int = 12,
    ta_dim: int = 63,
    signal_dim: int = 11,
    base_units: int = 512,  # Significantly increased
    memory_efficient: bool = True,
    mixed_precision: bool = True,
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 8,
    **kwargs
) -> keras.models.Model:
    """Primary model creation function - calls into build_ensemble_model for compatibility"""
    return build_ensemble_model(
        model_5m_window=model_5m_window,
        model_15m_window=model_15m_window,
        model_1h_window=model_1h_window,
        feature_dim=feature_dim,
        santiment_dim=santiment_dim,
        ta_dim=ta_dim,
        signal_dim=signal_dim,
        base_units=base_units,
        depth=6,  # Increased depth
        memory_efficient=memory_efficient,
        mixed_precision=mixed_precision,
        gradient_accumulation=gradient_accumulation,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **kwargs
    )