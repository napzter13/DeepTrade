import tensorflow as tf
from tensorflow.keras import layers, regularizers
from .environment import (
    NUM_FUTURE_STEPS,
)

################################################################################
# 1) Better balanced loss function with proper normalization
################################################################################

def weighted_mse_loss(y_true, y_pred):
    """
    Balanced MSE with proper normalization and handling of time-series prediction.
    Better suited for financial data with values in [-1,1] range.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # More balanced weights for future predictions (less aggressive decay)
    weights = tf.linspace(1.5, 1.0, NUM_FUTURE_STEPS)
    weights = tf.reshape(weights, (1, -1))  # shape(1, NUM_FUTURE_STEPS)
    
    # Calculate error with gentler gradient clipping
    errors = y_true - y_pred
    # Use less aggressive clipping to allow larger gradients through
    clipped_errors = tf.clip_by_value(errors, -3.0, 3.0)
    err_sq = tf.square(clipped_errors)
    
    # Apply weights
    weighted_err_sq = err_sq * weights
    
    # Handle NaN values
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, tf.float32)
    
    # Apply mask
    masked_weighted_err_sq = weighted_err_sq * mask
    
    # Calculate mean
    num_valid = tf.reduce_sum(mask, axis=-1)
    total_err = tf.reduce_sum(masked_weighted_err_sq, axis=-1)
    
    # Prevent division by zero
    num_valid = tf.maximum(num_valid, 1.0)
    
    return total_err / num_valid

################################################################################
# 2) Refactored residual block with better regularization
################################################################################

def residual_block(
    x, 
    units, 
    dropout_rate=0.0,
    l2_reg=0.0001,
    use_lstm=True,
    activation='selu',
    name_prefix="res_block"
):
    """
    Improved residual block with stronger regularization and better gradient stability.
    """
    # Create shortcut connection
    shortcut = x
    
    # Normalize inputs before processing (helps training stability)
    x_norm = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name_prefix}_input_norm"
    )(x)
    
    if use_lstm:
        # LSTM path
        main_path = layers.LSTM(
            units,
            return_sequences=True,
            dropout=0.0,  # CuDNN compatibility
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg/2),  # Less aggressive on recurrent weights
            name=f"{name_prefix}_lstm",
            # Use recurrent activation that works better with regularization
            recurrent_activation='sigmoid'
        )(x_norm)
        
        # Apply dropout after LSTM
        if dropout_rate > 0:
            main_path = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(main_path)
    else:
        # Dense path with better regularization
        main_path = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='glorot_normal',
            name=f"{name_prefix}_dense"
        )(x_norm)
        
        # Add batch normalization (helps with dense layers)
        main_path = layers.BatchNormalization(name=f"{name_prefix}_bn")(main_path)
        
        if dropout_rate > 0:
            main_path = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(main_path)
    
    # Project shortcut if dimensions don't match
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(
            units, 
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name_prefix}_shortcut"
        )(shortcut)
    
    # Add skip connection
    output = layers.Add(name=f"{name_prefix}_add")([main_path, shortcut])
    
    return output

################################################################################
# 3) Simplified time-series encoder with better regularization
################################################################################

def time_series_encoder(
    input_tensor,
    base_units=64,
    n_lstm_blocks=2,
    dropout_rate=0.3,  # Increased dropout
    l2_reg=0.0005,     # Stronger L2
    name_prefix="ts_encoder"
):
    """
    Simplified time-series encoder with better regularization and
    reduced complexity for better generalization.
    """
    # Input normalization
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name_prefix}_input_norm"
    )(input_tensor)
    
    # Initial convolution to extract patterns (with stronger regularization)
    x = layers.Conv1D(
        base_units, 
        kernel_size=3, 
        padding='same',
        activation='relu',  # Use ReLU for better gradient flow
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv1d"
    )(x)
    
    # Layer normalization after convolution
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name_prefix}_conv_norm"
    )(x)
    
    # Stack fewer LSTM blocks (reduced complexity)
    for i in range(n_lstm_blocks):
        # Less aggressive scaling of units
        block_units = base_units * (min(i+1, 2))  # More conservative unit scaling
        x = residual_block(
            x,
            units=block_units,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            use_lstm=True,
            name_prefix=f"{name_prefix}_lstm_block{i+1}"
        )
    
    # Simplified pooling (just use average and last-step for stability)
    avg_pool = layers.GlobalAveragePooling1D(name=f"{name_prefix}_avg_pool")(x)
    last_step = layers.Lambda(lambda x: x[:, -1, :], name=f"{name_prefix}_last_step")(x)
    
    # Concatenate different views
    x = layers.Concatenate(name=f"{name_prefix}_multi_view")([avg_pool, last_step])
    
    # Final encoding with stronger dropout
    x = layers.Dense(
        base_units,  # Reduced capacity
        activation='relu',  # ReLU for more stable gradients
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_encoding"
    )(x)
    
    x = layers.BatchNormalization(name=f"{name_prefix}_final_bn")(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_dropout")(x)
    
    return x

################################################################################
# 4) Optimized multi-timeframe model with better regularization
################################################################################

def build_multi_timeframe_model(
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
    # Improved hyperparameters
    base_units=48,            # Reduced base units to prevent overfitting
    dropout_rate=0.35,        # Increased dropout for better regularization
    l2_reg=0.0005,            # Stronger L2 regularization
    use_gradient_clipping=True,
):
    """
    Redesigned multi-timeframe model with:
    1. Simplified architecture with fewer parameters
    2. Stronger regularization
    3. Better balanced feature importance
    4. Optimized for [-1,1] output range with tanh activation
    """
    # === Input branches (simplified structure) ===
    
    # 5m branch (high importance)
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = time_series_encoder(
        input_5m,
        base_units=base_units * 1.5,  # More balanced scaling
        n_lstm_blocks=2,  # Reduced depth
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_5m"
    )
    
    # 15m branch
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = time_series_encoder(
        input_15m,
        base_units=base_units * 1.5,
        n_lstm_blocks=2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_15m"
    )
    
    # 1h branch
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = time_series_encoder(
        input_1h,
        base_units=base_units,
        n_lstm_blocks=1,  # Simplified
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_1h"
    )
    
    # Google trend branch (simplified)
    input_google_trend = layers.Input(
        shape=(window_google_trend, feature_google_trend),
        name="input_google_trend"
    )
    x_google = time_series_encoder(
        input_google_trend,
        base_units=base_units // 2,
        n_lstm_blocks=1,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_google"
    )
    
    # === Scalar feature branches (simplified) ===
    
    # Santiment processing
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    x_santiment = layers.BatchNormalization(name="santiment_bn")(input_santiment)
    x_santiment = layers.Dense(
        base_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="santiment_dense1"
    )(x_santiment)
    x_santiment = layers.Dropout(dropout_rate, name="santiment_dropout")(x_santiment)
    
    # Technical analysis processing (more important)
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    x_ta = layers.BatchNormalization(name="ta_bn")(input_ta)
    x_ta = layers.Dense(
        base_units * 1.5,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="ta_dense1"
    )(x_ta)
    x_ta = layers.BatchNormalization(name="ta_bn2")(x_ta)
    x_ta = layers.Dropout(dropout_rate, name="ta_dropout")(x_ta)
    
    # Signal processing
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    x_signal = layers.BatchNormalization(name="signal_bn")(input_signal)
    x_signal = layers.Dense(
        base_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="signal_dense1"
    )(x_signal)
    x_signal = layers.Dropout(dropout_rate, name="signal_dropout")(x_signal)
    
    # === Feature scaling (more balanced approach) ===
    # Less extreme scaling differences between features
    x_5m_scaled = layers.Lambda(lambda x: x * 1.3, name="scale_5m")(x_5m)
    x_15m_scaled = layers.Lambda(lambda x: x * 1.2, name="scale_15m")(x_15m)
    x_1h_scaled = layers.Lambda(lambda x: x * 1.0, name="scale_1h")(x_1h)
    x_google_scaled = layers.Lambda(lambda x: x * 0.7, name="scale_google")(x_google)
    x_santiment_scaled = layers.Lambda(lambda x: x * 0.7, name="scale_santiment")(x_santiment)
    x_ta_scaled = layers.Lambda(lambda x: x * 1.2, name="scale_ta")(x_ta)
    x_signal_scaled = layers.Lambda(lambda x: x * 1.0, name="scale_signal")(x_signal)
    
    # === Merge all features ===
    merged = layers.Concatenate(name="concat_all")([
        x_5m_scaled, x_15m_scaled, x_1h_scaled,
        x_google_scaled, x_santiment_scaled,
        x_ta_scaled, x_signal_scaled
    ])
    
    # === Simplified final processing (fewer layers) ===
    x = layers.BatchNormalization(name="merged_bn")(merged)
    
    # First dense layer
    x = layers.Dense(
        base_units * 4,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="final_dense1"
    )(x)
    x = layers.BatchNormalization(name="final_bn1")(x)
    x = layers.Dropout(dropout_rate, name="final_dropout1")(x)
    
    # Second dense layer with skip connection
    skip = layers.Dense(base_units * 2, name="skip_proj")(merged)
    x = layers.Dense(
        base_units * 2,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="final_dense2"
    )(x)
    x = layers.Add(name="final_skip")([x, skip])
    x = layers.BatchNormalization(name="final_bn2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="final_dropout2")(x) # Lighter dropout before output
    
    # Output layer with tanh activation (correct for [-1,1] range)
    predictions = layers.Dense(
        NUM_FUTURE_STEPS,
        activation='tanh',  # Tanh is correct for [-1,1] target range
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
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
        name="improved_multi_timeframe_model"
    )
    
    # Improved optimizer setup
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,  # Slightly higher starting LR for faster convergence
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=0.8 if use_gradient_clipping else None,
        clipvalue=1.0 if use_gradient_clipping else None  # More generous clipping
    )
    
    # Compile with balanced loss
    model.compile(optimizer=optimizer, loss=weighted_mse_loss)
    
    return model

################################################################################
# 5) Model loader function (keeping the same API)
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
    Factory function that creates and returns the optimized model.
    """
    print("Creating improved multi-timeframe model with better regularization...")
    
    model = build_multi_timeframe_model(
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