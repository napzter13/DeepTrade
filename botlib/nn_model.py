import tensorflow as tf
from tensorflow.keras import layers, regularizers
from .environment import (
    NUM_FUTURE_STEPS,
)

################################################################################
# 1) Numerically stable loss function to prevent NaN values
################################################################################

def weighted_mse_loss(y_true, y_pred):
    """
    Enhanced numerically stable MSE with proper handling of financial data.
    Includes safeguards against NaN/Inf values.
    """
    # Cast to float32 for better numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Strong focus on near-term predictions with smoother gradient
    weights = tf.linspace(1.8, 0.8, NUM_FUTURE_STEPS)
    weights = tf.reshape(weights, (1, -1))
    
    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, -1.0, 1.0)
    
    # Squared error with safe handling of numerical issues
    squared_error = tf.square(tf.clip_by_value(y_true - y_pred, -10.0, 10.0))
    
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
# 2) Stabilized GRU block with safeguards against numerical issues
################################################################################

def stable_gru_block(
    x, 
    units, 
    dropout_rate=0.35,  # Increased dropout
    l2_reg=0.0003,      # Increased L2
    name_prefix="gru_block"
):
    """
    Numerically stable GRU block with residual connection and 
    safeguards against gradient explosion.
    """
    # Store shortcut
    shortcut = x
    
    # Normalize input with safe epsilon
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_ln")(x)
    
    # GRU layer with conservative initialization
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
        recurrent_regularizer=regularizers.l2(l2_reg/3),
        bias_regularizer=regularizers.l2(l2_reg/3),
        activity_regularizer=None,
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        recurrent_constraint=tf.keras.constraints.MaxNorm(3.0),
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
            kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
            name=f"{name_prefix}_shortcut"
        )(shortcut)
    
    # Scale outputs for more stable gradient flow
    scale_factor = 0.2  # Small scale factor to prevent extreme gradients
    x = layers.Lambda(lambda x: x * scale_factor, name=f"{name_prefix}_scale")(x)
    
    # Add residual connection
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    
    return x

################################################################################
# 3) Enhanced Time Series Encoder with anti-overfitting measures
################################################################################

def enhanced_ts_encoder(
    input_tensor,
    base_units=384,  # Slightly reduced to prevent overfitting
    n_blocks=2,
    importance='medium', # 'high' for 5m/15m, 'medium' for 1h, 'low' for others
    dropout_rate=0.35,   # Increased dropout
    l2_reg=0.0003,       # Increased L2
    name_prefix="ts_encoder"
):
    """
    Enhanced time series encoder with anti-overfitting measures and
    safeguards against numerical instability.
    """
    # Scale capacity based on importance with less extreme differences
    if importance == 'high':
        scale_factor = 1.0
        n_filters = base_units // 4
    elif importance == 'medium':
        scale_factor = 0.8  # Less reduction
        n_filters = base_units // 5
    else:  # 'low'
        scale_factor = 0.6  # Less reduction
        n_filters = base_units // 6
    
    actual_units = int(base_units * scale_factor)
    
    # Normalize the input first for better stability
    x_norm = layers.LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_input_ln")(input_tensor)
    
    # Multi-scale convolution block with weight constraints
    conv_constraint = tf.keras.constraints.MaxNorm(3.0)
    
    conv1 = layers.Conv1D(
        n_filters,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_s"
    )(x_norm)
    
    conv2 = layers.Conv1D(
        n_filters,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_m"
    )(x_norm)
    
    conv3 = layers.Conv1D(
        n_filters,
        kernel_size=9,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=conv_constraint,
        name=f"{name_prefix}_conv_l"
    )(x_norm)
    
    # Merge convolutions
    x = layers.Concatenate(name=f"{name_prefix}_conv_concat")([conv1, conv2, conv3])
    
    # Batch normalization with safe parameters
    x = layers.BatchNormalization(
        momentum=0.9, 
        epsilon=1e-5, 
        name=f"{name_prefix}_bn"
    )(x)
    
    # Apply stabilized GRU blocks
    for i in range(n_blocks):
        # Apply different dropout rates at different depths
        block_dropout = dropout_rate * (1.0 - 0.1 * i)  # Reduce dropout slightly in deeper layers
        x = stable_gru_block(
            x,
            actual_units,
            dropout_rate=block_dropout,
            l2_reg=l2_reg,
            name_prefix=f"{name_prefix}_block{i+1}"
        )
        
        # Add batch norm between blocks for more stable gradients
        if i < n_blocks - 1:
            x = layers.BatchNormalization(
                momentum=0.9, 
                epsilon=1e-5, 
                name=f"{name_prefix}_bn{i+1}"
            )(x)
    
    # Multi-view pooling with simpler, more stable operations
    # 1. Global average pooling (most stable)
    avg_pool = layers.GlobalAveragePooling1D(name=f"{name_prefix}_avg_pool")(x)
    
    # 2. Global max pooling (useful but can be unstable, so we clip)
    x_for_max = layers.Lambda(
        lambda x: tf.clip_by_value(x, -5.0, 5.0), 
        name=f"{name_prefix}_clip_for_max"
    )(x)
    max_pool = layers.GlobalMaxPooling1D(name=f"{name_prefix}_max_pool")(x_for_max)
    
    # 3. Last step
    last_step = layers.Lambda(lambda x: x[:, -1, :], name=f"{name_prefix}_last_step")(x)
    
    # Concatenate pooling methods
    pooled = layers.Concatenate(name=f"{name_prefix}_pools")([avg_pool, max_pool, last_step])
    
    # Final dense encoding
    x = layers.Dense(
        actual_units,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name=f"{name_prefix}_dense"
    )(pooled)
    
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name_prefix}_final_bn"
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_drop")(x)
    
    return x

################################################################################
# 4) Enhanced TA feature processor with stability improvements
################################################################################

def enhanced_ta_processor(
    input_tensor,
    base_units=384,  # Slightly reduced
    depth=3,
    dropout_rate=0.35,  # Increased dropout
    l2_reg=0.0003,      # Increased L2
    name_prefix="ta_proc"
):
    """
    Enhanced technical analysis processor with stability measures
    and anti-overfitting regularization.
    """
    # Initial normalization for better stability
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name_prefix}_bn"
    )(input_tensor)
    
    # Initial projection
    x = layers.Dense(
        base_units,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name=f"{name_prefix}_proj"
    )(x)
    
    # Deep residual tower
    for i in range(depth):
        skip = x
        
        # Progressive dropout reduction for better information flow
        layer_dropout = dropout_rate * (1.0 - 0.1 * i)
        
        # First dense
        x = layers.Dense(
            base_units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
            name=f"{name_prefix}_dense{i+1}_1"
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            name=f"{name_prefix}_bn{i+1}_1"
        )(x)
        x = layers.Dropout(layer_dropout, name=f"{name_prefix}_drop{i+1}_1")(x)
        
        # Second dense with smaller units for bottleneck effect
        bottleneck_units = int(base_units * 0.75)
        x = layers.Dense(
            bottleneck_units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
            name=f"{name_prefix}_dense{i+1}_2"
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            name=f"{name_prefix}_bn{i+1}_2"
        )(x)
        x = layers.Dropout(layer_dropout, name=f"{name_prefix}_drop{i+1}_2")(x)
        
        # Project skip connection to match dimensions if needed
        if skip.shape[-1] != bottleneck_units:
            skip = layers.Dense(
                bottleneck_units,
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(l2_reg/2),
                kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
                name=f"{name_prefix}_skip_proj{i+1}"
            )(skip)
        
        # Scale factor for more stable gradients
        x = layers.Lambda(
            lambda x: x * 0.2, 
            name=f"{name_prefix}_scale{i+1}"
        )(x)
        
        # Residual connection
        x = layers.Add(name=f"{name_prefix}_add{i+1}")([x, skip])
    
    return x

################################################################################
# 5) Standard scalar processor with stability improvements
################################################################################

def scalar_processor(
    input_tensor,
    base_units=256,  # Slightly reduced
    depth=2,
    dropout_rate=0.35,  # Increased dropout
    l2_reg=0.0003,      # Increased L2
    name_prefix="scalar_proc"
):
    """Enhanced standard scalar feature processor with stability improvements."""
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name_prefix}_bn"
    )(input_tensor)
    
    # Initial projection
    x = layers.Dense(
        base_units,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name=f"{name_prefix}_proj"
    )(x)
    
    # Residual blocks
    for i in range(depth):
        skip = x
        
        # Single dense layer with BN and activation
        x = layers.Dense(
            base_units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
            name=f"{name_prefix}_dense{i+1}"
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5,
            name=f"{name_prefix}_bn{i+1}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop{i+1}")(x)
        
        # Scale output for stable gradients
        x = layers.Lambda(
            lambda x: x * 0.3, 
            name=f"{name_prefix}_scale{i+1}"
        )(x)
        
        # Residual connection
        x = layers.Add(name=f"{name_prefix}_add{i+1}")([x, skip])
    
    return x

################################################################################
# 6) Stabilized high-capacity model with prioritized 5m/15m/TA data
################################################################################

def build_prioritized_model(
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
    # Optimized hyperparameters
    base_units=384,      # Reduced units to prevent overfitting
    dropout_rate=0.35,   # Increased dropout
    l2_reg=0.0003,       # Increased L2
    use_gradient_clipping=True,
):
    """
    High-capacity multi-timeframe model with:
    1. Prioritized 5m, 15m, and TA data processing
    2. Enhanced numerical stability
    3. Anti-overfitting measures
    4. Improved gradient flow
    5. Safeguards against NaN/Inf values
    """
    # === Input branches with prioritized encoders ===
    
    # 5m branch (highest importance)
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = enhanced_ts_encoder(
        input_5m,
        base_units=base_units,
        n_blocks=2,
        importance='high',  # Highest priority
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_5m"
    )
    
    # 15m branch (high importance)
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = enhanced_ts_encoder(
        input_15m,
        base_units=base_units,
        n_blocks=2,
        importance='high',  # Highest priority
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_15m"
    )
    
    # 1h branch (medium importance)
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = enhanced_ts_encoder(
        input_1h,
        base_units=base_units,
        n_blocks=1,
        importance='medium',  # Medium priority
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_1h"
    )
    
    # Google trend branch (low importance)
    input_google_trend = layers.Input(
        shape=(window_google_trend, feature_google_trend),
        name="input_google_trend"
    )
    x_google = enhanced_ts_encoder(
        input_google_trend,
        base_units=base_units,
        n_blocks=1,
        importance='low',  # Low priority
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_google"
    )
    
    # === Scalar feature branches with prioritized processing ===
    
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
    
    # === Feature importance weighting based on priorities ===
    # Explicit high weights for 5m, 15m, and TA with constraints
    x_5m_scaled = layers.Lambda(lambda x: x * 1.5, name="scale_5m")(x_5m)
    x_15m_scaled = layers.Lambda(lambda x: x * 1.4, name="scale_15m")(x_15m)
    x_1h_scaled = layers.Lambda(lambda x: x * 1.0, name="scale_1h")(x_1h)
    x_google_scaled = layers.Lambda(lambda x: x * 0.7, name="scale_google")(x_google)
    x_santiment_scaled = layers.Lambda(lambda x: x * 0.8, name="scale_santiment")(x_santiment)
    x_ta_scaled = layers.Lambda(lambda x: x * 1.3, name="scale_ta")(x_ta)
    x_signal_scaled = layers.Lambda(lambda x: x * 0.9, name="scale_signal")(x_signal)
    
    # === Merge all features ===
    merged = layers.Concatenate(name="concat_all")([
        x_5m_scaled, x_15m_scaled, x_1h_scaled,
        x_google_scaled, x_santiment_scaled, 
        x_ta_scaled, x_signal_scaled
    ])
    
    # === High-capacity fusion network with stability improvements ===
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name="merged_bn"
    )(merged)
    
    # First fusion layer with modest width
    x = layers.Dense(
        base_units * 3,  # Slightly reduced width
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name="fusion_dense1"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name="fusion_bn1"
    )(x)
    x = layers.Dropout(dropout_rate, name="fusion_drop1")(x)
    
    # Residual block 1 with stability improvements
    skip1 = x
    x = layers.Dense(
        base_units * 3,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name="fusion_res1_dense1"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name="fusion_res1_bn1"
    )(x)
    x = layers.Dropout(dropout_rate, name="fusion_res1_drop1")(x)
    
    # Second dense with bottleneck
    bottleneck_units = int(base_units * 2.5)
    x = layers.Dense(
        bottleneck_units,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name="fusion_res1_dense2"
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name="fusion_res1_bn2"
    )(x)
    
    # Project skip1 to match bottleneck
    skip1_proj = layers.Dense(
        bottleneck_units,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name="fusion_res1_skip_proj"
    )(skip1)
    
    # Scale output for gradient stability
    x = layers.Lambda(lambda x: x * 0.2, name="fusion_res1_scale")(x)
    
    # Add skip connection
    x = layers.Add(name="fusion_res1_add")([x, skip1_proj])
    
    # Direct connection from 5m, 15m, and TA to improve gradient flow
    # Project these important features to match dimensions
    priority_features = layers.Concatenate(name="priority_concat")([
        x_5m_scaled, x_15m_scaled, x_ta_scaled
    ])
    
    # Project to match dimensions with stability constraints
    priority_projection = layers.Dense(
        bottleneck_units,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        activation='relu',
        name="priority_proj"
    )(priority_features)
    
    # Add direct connection with scaling
    priority_projection = layers.Lambda(
        lambda x: x * 0.3, 
        name="priority_scale"
    )(priority_projection)
    
    x = layers.Add(name="fusion_priority_add")([x, priority_projection])
    
    # Final layer before output
    x = layers.BatchNormalization(
        momentum=0.9,
        epsilon=1e-5,
        name="fusion_final_bn"
    )(x)
    
    x = layers.Dense(
        base_units,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
        name="fusion_dense_final"
    )(x)
    
    x = layers.Dropout(dropout_rate * 0.5, name="fusion_drop_final")(x)
    
    # Final prediction layer with careful initialization
    predictions = layers.Dense(
        NUM_FUTURE_STEPS,
        activation='tanh',  # Correct for [-1,1] target range
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        kernel_regularizer=regularizers.l2(l2_reg/2),
        kernel_constraint=tf.keras.constraints.MaxNorm(3.0),
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
        name="stable_prioritized_model"
    )
    
    # Optimizer setup with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=0.8 if use_gradient_clipping else None,  # Tighter norm clipping
        clipvalue=3.0 if use_gradient_clipping else None  # Tighter value clipping
    )
    
    # Compile with enhanced numerically stable loss
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
    Factory function that creates and returns the stabilized high-capacity model
    with prioritized 5m, 15m, and TA data processing.
    """
    print("Creating stabilized high-capacity model with prioritized 5m, 15m, and TA data...")
    
    model = build_prioritized_model(
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