import tensorflow as tf
from tensorflow.keras import layers, regularizers
from .environment import (
    NUM_FUTURE_STEPS,
)

################################################################################
# 1) Enhanced loss function with better training properties
################################################################################

def weighted_mse_loss(y_true, y_pred):
    """
    Enhanced MSE with better weighting for financial time series.
    Less aggressive clipping to allow model to learn from outliers.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Strong focus on near-term predictions
    weights = tf.linspace(1.8, 0.8, NUM_FUTURE_STEPS)
    weights = tf.reshape(weights, (1, -1))
    
    # Standard squared error without excessive clipping
    squared_error = tf.square(y_true - y_pred)
    
    # Apply weights
    weighted_squared_error = squared_error * weights
    
    # Handle NaN values
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, tf.float32)
    
    # Apply mask and mean
    masked_weighted_error = weighted_squared_error * mask
    num_valid = tf.reduce_sum(mask, axis=-1)
    num_valid = tf.maximum(num_valid, 1.0)  # Prevent division by zero
    
    return tf.reduce_sum(masked_weighted_error, axis=-1) / num_valid

################################################################################
# 2) Improved GRU block with better gradient flow
################################################################################

def improved_gru_block(
    x, 
    units, 
    dropout_rate=0.25,  # Reduced dropout
    l2_reg=0.0001,      # Reduced L2
    name_prefix="gru_block",
    apply_scaling=True  # Allow disabling scaling
):
    """
    Improved GRU block with better balance of regularization.
    """
    # Store shortcut
    shortcut = x
    
    # Normalize input with safe epsilon
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_ln")(x)
    
    # GRU layer with better initialization
    x = layers.GRU(
        units,
        return_sequences=True,
        dropout=0.0,             # For CuDNN compatibility
        recurrent_dropout=0.0,   # For CuDNN compatibility
        activation='tanh',
        recurrent_activation='sigmoid',
        reset_after=True,        # For CuDNN compatibility
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        kernel_regularizer=regularizers.l2(l2_reg),
        recurrent_regularizer=regularizers.l2(l2_reg/2),
        bias_regularizer=None,  # No bias regularization
        activity_regularizer=None,
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),  # Less restrictive
        recurrent_constraint=tf.keras.constraints.MaxNorm(5.0),  # Less restrictive
        name=f"{name_prefix}_gru"
    )(x)
    
    # Apply dropout
    if dropout_rate > 0:
        x = layers.SpatialDropout1D(dropout_rate, name=f"{name_prefix}_dropout")(x)
    
    # Project shortcut if dimensions don't match
    if shortcut.shape[-1] != units:
        shortcut = layers.Conv1D(
            units, 1, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
            name=f"{name_prefix}_shortcut"
        )(shortcut)
    
    # Scale outputs for better gradient flow - but less aggressively
    if apply_scaling:
        scale_factor = 0.5  # Larger scale factor to allow more gradient flow
        x = layers.Lambda(lambda x: x * scale_factor, name=f"{name_prefix}_scale")(x)
    
    # Add residual connection
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    
    return x

################################################################################
# 3) Time Series Encoder adapted for various sequence lengths
################################################################################

def ts_encoder_5m(
    input_tensor,
    base_units=384,
    n_blocks=2,
    dropout_rate=0.25,
    l2_reg=0.0001,
    name_prefix="enc_5m"
):
    """Encoder specifically for 5-minute data (241 timesteps)."""
    # Multi-scale convolution block
    conv_constraint = tf.keras.constraints.MaxNorm(5.0)
    
    n_filters = base_units // 3
    
    conv1 = layers.Conv1D(
        n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_s"
    )(input_tensor)
    
    conv2 = layers.Conv1D(
        n_filters,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_m"
    )(input_tensor)
    
    conv3 = layers.Conv1D(
        n_filters,
        kernel_size=9,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_l"
    )(input_tensor)
    
    # Merge convolutions
    x = layers.Concatenate(name=f"{name_prefix}_conv_concat")([conv1, conv2, conv3])
    
    # Batch normalization with stable parameters
    x = layers.BatchNormalization(
        momentum=0.99, 
        epsilon=1e-5, 
        name=f"{name_prefix}_bn"
    )(x)
    
    # Apply improved GRU blocks
    for i in range(n_blocks):
        # Apply different dropout rates at different depths
        block_dropout = dropout_rate * (1.0 - 0.1 * i)  # Reduce dropout slightly in deeper layers
        
        # Apply scaling only to the last block
        apply_scaling = (i == n_blocks - 1)
        
        x = improved_gru_block(
            x,
            base_units,
            dropout_rate=block_dropout,
            l2_reg=l2_reg,
            name_prefix=f"{name_prefix}_block{i+1}",
            apply_scaling=apply_scaling
        )
        
        # Add batch norm between blocks
        if i < n_blocks - 1:
            x = layers.BatchNormalization(
                momentum=0.99, 
                epsilon=1e-5, 
                name=f"{name_prefix}_bn{i+1}"
            )(x)
    
    # Multi-view pooling with fixed indices for 5m data
    # 1. Global average pooling
    avg_pool = layers.GlobalAveragePooling1D(name=f"{name_prefix}_avg_pool")(x)
    
    # 2. Global max pooling
    max_pool = layers.GlobalMaxPooling1D(name=f"{name_prefix}_max_pool")(x)
    
    # 3. Last step
    last_step = layers.Lambda(lambda x: x[:, -1, :], name=f"{name_prefix}_last_step")(x)
    
    # 4. Quarter and three-quarter points (indices 60 and 180 for 241-length sequences)
    quarter_step = layers.Lambda(lambda x: x[:, 60, :], name=f"{name_prefix}_quarter_step")(x)
    three_quarter_step = layers.Lambda(lambda x: x[:, 180, :], name=f"{name_prefix}_three_quarter_step")(x)
    
    # Concatenate pooling methods
    pooled = layers.Concatenate(name=f"{name_prefix}_pools")([avg_pool, max_pool, last_step, quarter_step, three_quarter_step])
    
    # Final dense encoding with ELU activation
    x = layers.Dense(
        base_units,
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name=f"{name_prefix}_dense"
    )(pooled)
    
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name=f"{name_prefix}_final_bn"
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_drop")(x)
    
    return x

def ts_encoder_15m(
    input_tensor,
    base_units=384,
    n_blocks=2,
    dropout_rate=0.25,
    l2_reg=0.0001,
    name_prefix="enc_15m"
):
    """Encoder specifically for 15-minute data (241 timesteps)."""
    # Multi-scale convolution block
    conv_constraint = tf.keras.constraints.MaxNorm(5.0)
    
    n_filters = base_units // 3
    
    conv1 = layers.Conv1D(
        n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_s"
    )(input_tensor)
    
    conv2 = layers.Conv1D(
        n_filters,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_m"
    )(input_tensor)
    
    conv3 = layers.Conv1D(
        n_filters,
        kernel_size=9,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_l"
    )(input_tensor)
    
    # Merge convolutions
    x = layers.Concatenate(name=f"{name_prefix}_conv_concat")([conv1, conv2, conv3])
    
    # Batch normalization with stable parameters
    x = layers.BatchNormalization(
        momentum=0.99, 
        epsilon=1e-5, 
        name=f"{name_prefix}_bn"
    )(x)
    
    # Apply improved GRU blocks
    for i in range(n_blocks):
        # Apply different dropout rates at different depths
        block_dropout = dropout_rate * (1.0 - 0.1 * i)
        
        # Apply scaling only to the last block
        apply_scaling = (i == n_blocks - 1)
        
        x = improved_gru_block(
            x,
            base_units,
            dropout_rate=block_dropout,
            l2_reg=l2_reg,
            name_prefix=f"{name_prefix}_block{i+1}",
            apply_scaling=apply_scaling
        )
        
        # Add batch norm between blocks
        if i < n_blocks - 1:
            x = layers.BatchNormalization(
                momentum=0.99, 
                epsilon=1e-5, 
                name=f"{name_prefix}_bn{i+1}"
            )(x)
    
    # Multi-view pooling with fixed indices for 15m data (same as 5m)
    # 1. Global average pooling
    avg_pool = layers.GlobalAveragePooling1D(name=f"{name_prefix}_avg_pool")(x)
    
    # 2. Global max pooling
    max_pool = layers.GlobalMaxPooling1D(name=f"{name_prefix}_max_pool")(x)
    
    # 3. Last step
    last_step = layers.Lambda(lambda x: x[:, -1, :], name=f"{name_prefix}_last_step")(x)
    
    # 4. Quarter and three-quarter points (indices 60 and 180 for 241-length sequences)
    quarter_step = layers.Lambda(lambda x: x[:, 60, :], name=f"{name_prefix}_quarter_step")(x)
    three_quarter_step = layers.Lambda(lambda x: x[:, 180, :], name=f"{name_prefix}_three_quarter_step")(x)
    
    # Concatenate pooling methods
    pooled = layers.Concatenate(name=f"{name_prefix}_pools")([avg_pool, max_pool, last_step, quarter_step, three_quarter_step])
    
    # Final dense encoding with ELU activation
    x = layers.Dense(
        base_units,
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name=f"{name_prefix}_dense"
    )(pooled)
    
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name=f"{name_prefix}_final_bn"
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_drop")(x)
    
    return x

def ts_encoder_1h(
    input_tensor,
    base_units=384,
    n_blocks=1,
    dropout_rate=0.25,
    l2_reg=0.0001,
    name_prefix="enc_1h"
):
    """Encoder specifically for 1-hour data (241 timesteps)."""
    # Multi-scale convolution block
    conv_constraint = tf.keras.constraints.MaxNorm(5.0)
    
    n_filters = base_units // 4  # Slightly smaller for 1h data
    
    conv1 = layers.Conv1D(
        n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_s"
    )(input_tensor)
    
    conv2 = layers.Conv1D(
        n_filters,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_m"
    )(input_tensor)
    
    conv3 = layers.Conv1D(
        n_filters,
        kernel_size=9,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_l"
    )(input_tensor)
    
    # Merge convolutions
    x = layers.Concatenate(name=f"{name_prefix}_conv_concat")([conv1, conv2, conv3])
    
    # Batch normalization
    x = layers.BatchNormalization(
        momentum=0.99, 
        epsilon=1e-5, 
        name=f"{name_prefix}_bn"
    )(x)
    
    # Apply improved GRU blocks
    for i in range(n_blocks):
        # Apply scaling only to the last block
        apply_scaling = (i == n_blocks - 1)
        
        x = improved_gru_block(
            x,
            int(base_units * 0.85),  # Slightly smaller for 1h
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"{name_prefix}_block{i+1}",
            apply_scaling=apply_scaling
        )
    
    # Multi-view pooling with fixed indices for 1h data (same as others)
    # 1. Global average pooling
    avg_pool = layers.GlobalAveragePooling1D(name=f"{name_prefix}_avg_pool")(x)
    
    # 2. Global max pooling
    max_pool = layers.GlobalMaxPooling1D(name=f"{name_prefix}_max_pool")(x)
    
    # 3. Last step
    last_step = layers.Lambda(lambda x: x[:, -1, :], name=f"{name_prefix}_last_step")(x)
    
    # 4. Quarter and three-quarter points (indices 60 and 180 for 241-length sequences)
    quarter_step = layers.Lambda(lambda x: x[:, 60, :], name=f"{name_prefix}_quarter_step")(x)
    three_quarter_step = layers.Lambda(lambda x: x[:, 180, :], name=f"{name_prefix}_three_quarter_step")(x)
    
    # Concatenate pooling methods
    pooled = layers.Concatenate(name=f"{name_prefix}_pools")([avg_pool, max_pool, last_step, quarter_step, three_quarter_step])
    
    # Final dense encoding
    x = layers.Dense(
        int(base_units * 0.85),
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name=f"{name_prefix}_dense"
    )(pooled)
    
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name=f"{name_prefix}_final_bn"
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_drop")(x)
    
    return x

def ts_encoder_google(
    input_tensor,
    base_units=384,
    n_blocks=1,
    dropout_rate=0.25,
    l2_reg=0.0001,
    name_prefix="enc_google"
):
    """Encoder specifically for Google trend data (24 timesteps)."""
    # Multi-scale convolution block with smaller kernel sizes
    conv_constraint = tf.keras.constraints.MaxNorm(5.0)
    
    n_filters = base_units // 5  # Even smaller for Google trends
    
    conv1 = layers.Conv1D(
        n_filters,
        kernel_size=2,  # Smaller kernels for shorter sequences
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_s"
    )(input_tensor)
    
    conv2 = layers.Conv1D(
        n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_m"
    )(input_tensor)
    
    conv3 = layers.Conv1D(
        n_filters,
        kernel_size=5,  # Max kernel size of 5 for 24-timestep sequence
        strides=1,
        padding='same',
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_l"
    )(input_tensor)
    
    # Merge convolutions
    x = layers.Concatenate(name=f"{name_prefix}_conv_concat")([conv1, conv2, conv3])
    
    # Batch normalization
    x = layers.BatchNormalization(
        momentum=0.99, 
        epsilon=1e-5, 
        name=f"{name_prefix}_bn"
    )(x)
    
    # Apply GRU blocks
    for i in range(n_blocks):
        x = improved_gru_block(
            x,
            int(base_units * 0.7),  # Smaller units for Google trends
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"{name_prefix}_block{i+1}",
            apply_scaling=True
        )
    
    # Multi-view pooling with fixed indices for Google trend data
    # 1. Global average pooling
    avg_pool = layers.GlobalAveragePooling1D(name=f"{name_prefix}_avg_pool")(x)
    
    # 2. Global max pooling
    max_pool = layers.GlobalMaxPooling1D(name=f"{name_prefix}_max_pool")(x)
    
    # 3. Last step
    last_step = layers.Lambda(lambda x: x[:, -1, :], name=f"{name_prefix}_last_step")(x)
    
    # 4. Quarter and three-quarter points (indices 6 and 18 for 24-length sequences)
    quarter_step = layers.Lambda(lambda x: x[:, 6, :], name=f"{name_prefix}_quarter_step")(x)
    three_quarter_step = layers.Lambda(lambda x: x[:, 18, :], name=f"{name_prefix}_three_quarter_step")(x)
    
    # Concatenate pooling methods
    pooled = layers.Concatenate(name=f"{name_prefix}_pools")([avg_pool, max_pool, last_step, quarter_step, three_quarter_step])
    
    # Final dense encoding
    x = layers.Dense(
        int(base_units * 0.7),
        activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name=f"{name_prefix}_dense"
    )(pooled)
    
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name=f"{name_prefix}_final_bn"
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_drop")(x)
    
    return x

################################################################################
# 4) Enhanced TA feature processor
################################################################################

def enhanced_ta_processor(
    input_tensor,
    base_units=384,
    depth=3,
    dropout_rate=0.25,  # Reduced dropout
    l2_reg=0.0001,      # Reduced L2
    name_prefix="ta_proc"
):
    """
    Enhanced technical analysis processor with better input sensitivity.
    """
    # Initial normalization
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name=f"{name_prefix}_bn"
    )(input_tensor)
    
    # Initial projection with ELU activation
    x = layers.Dense(
        base_units,
        activation='elu',  # Changed to ELU
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name=f"{name_prefix}_proj"
    )(x)
    
    # Deep residual tower
    for i in range(depth):
        skip = x
        
        # First dense with ELU
        x = layers.Dense(
            base_units,
            activation='elu',  # Changed to ELU
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
            name=f"{name_prefix}_dense{i+1}_1"
        )(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            epsilon=1e-5,
            name=f"{name_prefix}_bn{i+1}_1"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop{i+1}_1")(x)
        
        # Second dense with less reduction in units
        bottleneck_units = int(base_units * 0.8)  # Less reduction
        x = layers.Dense(
            bottleneck_units,
            activation='elu',  # Changed to ELU
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
            name=f"{name_prefix}_dense{i+1}_2"
        )(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            epsilon=1e-5,
            name=f"{name_prefix}_bn{i+1}_2"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop{i+1}_2")(x)
        
        # Project skip connection to match dimensions if needed
        if skip.shape[-1] != bottleneck_units:
            skip = layers.Dense(
                bottleneck_units,
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2_reg/2),
                kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
                name=f"{name_prefix}_skip_proj{i+1}"
            )(skip)
        
        # Only apply scaling on the final layer
        if i == depth - 1:
            # Less aggressive scaling
            x = layers.Lambda(
                lambda x: x * 0.7, 
                name=f"{name_prefix}_scale{i+1}"
            )(x)
        
        # Residual connection
        x = layers.Add(name=f"{name_prefix}_add{i+1}")([x, skip])
    
    return x

################################################################################
# 5) Standard scalar processor with improved sensitivity
################################################################################

def scalar_processor(
    input_tensor,
    base_units=256,
    depth=2,
    dropout_rate=0.25,  # Reduced dropout
    l2_reg=0.0001,      # Reduced L2
    name_prefix="scalar_proc"
):
    """Enhanced scalar feature processor with better input sensitivity."""
    # Input normalization
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name=f"{name_prefix}_bn"
    )(input_tensor)
    
    # Initial projection with ELU activation
    x = layers.Dense(
        base_units,
        activation='elu',  # Changed to ELU
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name=f"{name_prefix}_proj"
    )(x)
    
    # Residual blocks
    for i in range(depth):
        skip = x
        
        # Dense layer with ELU activation
        x = layers.Dense(
            base_units,
            activation='elu',  # Changed to ELU
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
            name=f"{name_prefix}_dense{i+1}"
        )(x)
        x = layers.BatchNormalization(
            momentum=0.99,
            epsilon=1e-5,
            name=f"{name_prefix}_bn{i+1}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop{i+1}")(x)
        
        # Only apply scaling on the final layer and less aggressively
        if i == depth - 1:
            x = layers.Lambda(
                lambda x: x * 0.7, 
                name=f"{name_prefix}_scale{i+1}"
            )(x)
        
        # Residual connection
        x = layers.Add(name=f"{name_prefix}_add{i+1}")([x, skip])
    
    return x

################################################################################
# 6) Improved high-capacity model with better input sensitivity
################################################################################

def build_improved_model(
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
    # Balanced hyperparameters
    base_units=384,
    dropout_rate=0.25,   # Reduced dropout
    l2_reg=0.0001,       # Reduced L2
    use_gradient_clipping=True,
):
    """
    Improved high-capacity multi-timeframe model with:
    1. Better input sensitivity
    2. Less excessive regularization
    3. Better activation functions
    4. Balanced feature importance
    5. More effective skip connections
    6. Fixed shape tensors for all Lambda operations
    """
    # === Input branches with specialized encoders for each timeframe ===
    
    # 5m branch (highest importance)
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = ts_encoder_5m(
        input_5m,
        base_units=base_units,
        n_blocks=2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_5m"
    )
    
    # 15m branch (high importance)
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = ts_encoder_15m(
        input_15m,
        base_units=base_units,
        n_blocks=2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_15m"
    )
    
    # 1h branch (medium importance)
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = ts_encoder_1h(
        input_1h,
        base_units=base_units,
        n_blocks=1,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_1h"
    )
    
    # Google trend branch (low importance)
    input_google_trend = layers.Input(
        shape=(window_google_trend, feature_google_trend),
        name="input_google_trend"
    )
    x_google = ts_encoder_google(
        input_google_trend,
        base_units=base_units,
        n_blocks=1,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_google"
    )
    
    # === Scalar feature branches with enhanced processing ===
    
    # Technical analysis processing (high importance)
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    x_ta = enhanced_ta_processor(
        input_ta,
        base_units=base_units,
        depth=3,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="ta"
    )
    
    # Santiment processing (medium importance)
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    x_santiment = scalar_processor(
        input_santiment,
        base_units=base_units // 2,
        depth=2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="santiment"
    )
    
    # Signal processing (medium importance)
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    x_signal = scalar_processor(
        input_signal,
        base_units=base_units // 2,
        depth=2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="signal"
    )
    
    # === More balanced feature weighting based on priorities ===
    # Less extreme weights
    x_5m_scaled = layers.Lambda(lambda x: x * 1.25, name="scale_5m")(x_5m)
    x_15m_scaled = layers.Lambda(lambda x: x * 1.20, name="scale_15m")(x_15m)
    x_1h_scaled = layers.Lambda(lambda x: x * 1.0, name="scale_1h")(x_1h)
    x_google_scaled = layers.Lambda(lambda x: x * 0.8, name="scale_google")(x_google)
    x_santiment_scaled = layers.Lambda(lambda x: x * 0.9, name="scale_santiment")(x_santiment)
    x_ta_scaled = layers.Lambda(lambda x: x * 1.15, name="scale_ta")(x_ta)
    x_signal_scaled = layers.Lambda(lambda x: x * 0.95, name="scale_signal")(x_signal)
    
    # === Merge all features ===
    merged = layers.Concatenate(name="concat_all")([
        x_5m_scaled, x_15m_scaled, x_1h_scaled,
        x_google_scaled, x_santiment_scaled, 
        x_ta_scaled, x_signal_scaled
    ])
    
    # === High-capacity fusion network with improved gradient flow ===
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name="merged_bn"
    )(merged)
    
    # First fusion layer with ELU activation
    x = layers.Dense(
        base_units * 3,
        activation='elu',  # Changed to ELU
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name="fusion_dense1"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name="fusion_bn1"
    )(x)
    x = layers.Dropout(dropout_rate, name="fusion_drop1")(x)
    
    # Residual block 1 with ELU activation
    skip1 = x
    x = layers.Dense(
        base_units * 3,
        activation='elu',  # Changed to ELU
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name="fusion_res1_dense1"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name="fusion_res1_bn1"
    )(x)
    x = layers.Dropout(dropout_rate, name="fusion_res1_drop1")(x)
    
    # Less bottlenecking in second dense
    bottleneck_units = int(base_units * 2.5)  # Less reduction
    x = layers.Dense(
        bottleneck_units,
        activation='elu',  # Changed to ELU
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name="fusion_res1_dense2"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name="fusion_res1_bn2"
    )(x)
    
    # Project skip1 to match bottleneck
    skip1_proj = layers.Dense(
        bottleneck_units,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name="fusion_res1_skip_proj"
    )(skip1)
    
    # Less aggressive scaling
    x = layers.Lambda(lambda x: x * 0.7, name="fusion_res1_scale")(x)
    
    # Add skip connection
    x = layers.Add(name="fusion_res1_add")([x, skip1_proj])
    
    # Direct connection from raw input features to improve sensitivity
    # Project these important features to match dimensions but with less scaling
    priority_features = layers.Concatenate(name="priority_concat")([
        x_5m, x_15m, x_ta  # Use raw unscaled features for more diversity
    ])
    
    # Project to match dimensions
    priority_projection = layers.Dense(
        bottleneck_units,
        activation='elu',  # ELU activation
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name="priority_proj"
    )(priority_features)
    
    # Less aggressive scaling of priority features
    priority_projection = layers.Lambda(
        lambda x: x * 0.5, 
        name="priority_scale"
    )(priority_projection)
    
    # Add direct connection
    x = layers.Add(name="fusion_priority_add")([x, priority_projection])
    
    # Final layer before output
    x = layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-5,
        name="fusion_final_bn"
    )(x)
    
    x = layers.Dense(
        base_units,
        activation='elu',  # Changed to ELU
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(5.0),
        name="fusion_dense_final"
    )(x)
    
    x = layers.Dropout(dropout_rate * 0.5, name="fusion_drop_final")(x)
    
    # Add noise layer to break symmetry if the model gets stuck
    # This helps prevent the model from outputting identical values
    x = layers.GaussianNoise(0.01, name="noise_layer")(x)
    
    # Final prediction layer with careful initialization and NO CONSTRAINT
    # Constraints on the final layer can cause identical outputs
    predictions = layers.Dense(
        NUM_FUTURE_STEPS,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
        kernel_regularizer=regularizers.l2(l2_reg/4),  # Less regularization
        kernel_constraint=None,  # No constraint on output layer
        name="output"
    )(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=[
            input_5m,
            input_15m,
            input_1h,
            input_google_trend,
            input_santiment,
            input_ta,
            input_signal,
        ],
        outputs=predictions,
        name="improved_responsive_model"
    )
    
    # Optimizer setup with less aggressive gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0 if use_gradient_clipping else None,  # Less aggressive norm clipping
        clipvalue=5.0 if use_gradient_clipping else None  # Less aggressive value clipping
    )
    
    # Compile with improved loss
    model.compile(optimizer=optimizer, loss=weighted_mse_loss)
    
    return model

################################################################################
# 7) Model loader function (keeping the same API)
################################################################################

def load_advanced_lstm_model(
    model_5m_window=241,
    model_15m_window=241,
    model_1h_window=241,
    feature_dim=9,
    santiment_dim=12,
    ta_dim=63,
    signal_dim=11,
    **kwargs  # Accept additional kwargs for compatibility
):
    """
    Factory function that creates and returns the improved model
    with better input sensitivity and response to different features.
    """
    print("Creating specialized time series encoders for different timestep sequences...")
    
    model = build_improved_model(
        window_5m=model_5m_window,
        feature_5m=feature_dim,
        window_15m=model_15m_window,
        feature_15m=feature_dim,
        window_1h=model_1h_window,
        feature_1h=feature_dim,
        window_google_trend=24,
        feature_google_trend=1,
        santiment_dim=santiment_dim,
        ta_dim=ta_dim,
        signal_dim=signal_dim
    )
    
    return model