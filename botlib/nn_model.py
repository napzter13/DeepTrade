# nn_model.py

import tensorflow as tf
from tensorflow.keras import layers

################################################################################
# 1) Residual LSTM block
################################################################################

def residual_lstm_block(
    x,
    units_lstm1=512,
    units_lstm2=256,
    dropout_rate=0.0,   # Turn off dropout for overfitting test
    name_prefix="lstm_block"
):
    """
    A residual block that applies two LSTM layers in sequence,
    then merges back via skip connection.
    """
    # First LSTM (return_sequences=True for next layers)
    lstm_1 = layers.LSTM(
        units_lstm1,
        return_sequences=True,
        # no kernel_regularizer
        name=f"{name_prefix}_lstm1"
    )(x)

    # Second LSTM
    lstm_2 = layers.LSTM(
        units_lstm2,
        return_sequences=True,
        name=f"{name_prefix}_lstm2"
    )(lstm_1)

    # Residual / skip connection from the input x (if shapes match).
    if x.shape[-1] != lstm_2.shape[-1]:
        x = layers.Dense(
            units_lstm2,
            name=f"{name_prefix}_skip_proj"
        )(x)
    out = layers.Add(name=f"{name_prefix}_skip_add")([x, lstm_2])
    return out

################################################################################
# 2) Simple Transformer block
################################################################################

def transformer_block(
    x,
    num_heads=4,
    ff_dim=512,
    dropout_rate=0.0,  # Turn off dropout for overfitting test
    name_prefix="transformer_block"
):
    """
    A simplified Transformer-style block:
      1) MultiHeadAttention (self-attention)
      2) Residual + LayerNorm
      3) Feed-forward network
      4) Residual + LayerNorm
    """
    # Self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1],
        name=f"{name_prefix}_mha"
    )(x, x)
    # Residual + LayerNorm
    out1 = layers.Add(name=f"{name_prefix}_attn_add")([x, attn_output])
    out1 = layers.LayerNormalization(name=f"{name_prefix}_attn_norm")(out1)

    # Feed Forward
    ffn = layers.Dense(ff_dim, activation='relu', name=f"{name_prefix}_ffn_dense1")(out1)
    ffn = layers.Dense(out1.shape[-1], name=f"{name_prefix}_ffn_dense2")(ffn)

    # Residual + LayerNorm
    out2 = layers.Add(name=f"{name_prefix}_ffn_add")([out1, ffn])
    out2 = layers.LayerNormalization(name=f"{name_prefix}_ffn_norm")(out2)

    return out2

################################################################################
# 3) Single timeframe encoder (residual LSTM + Transformer)
################################################################################

def single_timeframe_encoder(
    input_tensor,
    units_lstm1=512,
    units_lstm2=256,
    num_heads=4,
    ff_dim=512,
    dropout_rate=0.0,
    name_prefix="timeframe"
):
    """
    Encodes a single time-series branch by:
      1) Residual LSTM block
      2) Transformer block
      3) GlobalAveragePooling
      4) Dense projection
    """
    # 1) Residual LSTM block
    x = residual_lstm_block(
        input_tensor,
        units_lstm1=units_lstm1,
        units_lstm2=units_lstm2,
        dropout_rate=dropout_rate,
        name_prefix=f"{name_prefix}_reslstm"
    )

    # 2) Transformer block
    x = transformer_block(
        x,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        name_prefix=f"{name_prefix}_transformer"
    )

    # 3) Global average pooling => (batch, features)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)

    # 4) Dense projection
    x = layers.Dense(ff_dim, activation='relu', name=f"{name_prefix}_final_dense")(x)
    return x

################################################################################
# 4) Full multi-timeframe model
################################################################################

def build_multi_timeframe_model(
    # More extended windows by default
    window_5m=241,
    feature_5m=9,
    window_15m=240,
    feature_15m=9,
    window_1h=240,
    feature_1h=9,
    window_google_trend=24,
    feature_google_trend=1,
    santiment_dim=12,
    ta_dim=63,
    signal_dim=11
):
    """
    A multi-branch model with:
      - Big Residual LSTM layers + Transformer block per timeframe
      - Additional sub-networks for Santiment, TA, and signals
      - Tanh final output
      - Minimally no dropout / no L2 to test overfitting
    """

    dropout_rate = 0.0  # no dropout for debugging overfit
    # You can reintroduce dropout/L2 after confirming overfit

    # === 5m branch ===
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = single_timeframe_encoder(
        input_5m,
        units_lstm1=512,
        units_lstm2=256,
        num_heads=8,    # more heads for bigger capacity
        ff_dim=512,
        dropout_rate=dropout_rate,
        name_prefix="enc_5m"
    )

    # === 15m branch ===
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = single_timeframe_encoder(
        input_15m,
        units_lstm1=512,
        units_lstm2=256,
        num_heads=8,
        ff_dim=512,
        dropout_rate=dropout_rate,
        name_prefix="enc_15m"
    )

    # === 1h branch ===
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = single_timeframe_encoder(
        input_1h,
        units_lstm1=512,
        units_lstm2=256,
        num_heads=8,
        ff_dim=512,
        dropout_rate=dropout_rate,
        name_prefix="enc_1h"
    )

    # === google_trend branch ===
    input_google_trend = layers.Input(
        shape=(window_google_trend, feature_google_trend),
        name="input_google_trend"
    )
    x_google_trend = single_timeframe_encoder(
        input_google_trend,
        units_lstm1=256,
        units_lstm2=128,
        num_heads=4,   # smaller than the bigger branches
        ff_dim=256,
        dropout_rate=dropout_rate,
        name_prefix="enc_google_trend"
    )

    # === Santiment context (12-dim) ===
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    x_santiment = layers.Dense(
        128, activation='relu',
        name="santiment_dense1"
    )(input_santiment)
    x_santiment = layers.Dense(
        64, activation='relu',
        name="santiment_dense2"
    )(x_santiment)

    # === TA context (63-dim) ===
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    x_ta = layers.Dense(
        128, activation='relu',
        name="ta_dense1"
    )(input_ta)
    x_ta = layers.Dense(
        64, activation='relu',
        name="ta_dense2"
    )(x_ta)

    # === Merge the four big time-series encoders + Santiment + TA ===
    merged_lstm_ta = layers.concatenate(
        [x_5m, x_15m, x_1h, x_google_trend, x_santiment, x_ta],
        name="concat_lstm_ta"
    )

    # A deeper dense block on the merged features
    x = layers.Dense(512, activation='relu', name="merged_dense1")(merged_lstm_ta)
    x = layers.Dense(256, activation='relu', name="merged_dense2")(x)
    x = layers.Dense(128, activation='relu', name="merged_dense3")(x)

    # === Final signals context (11-dim) ===
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    x_sig = layers.Dense(
        128, activation='relu',
        name="signal_dense1"
    )(input_signal)
    x_sig = layers.Dense(
        64, activation='relu',
        name="signal_dense2"
    )(x_sig)

    # === Merge partial network x with x_sig ===
    x_merged_signal = layers.concatenate([x, x_sig], name="concat_signal_branch")

    # Additional dense block -> output
    x2 = layers.Dense(256, activation='relu', name="final_dense1")(x_merged_signal)
    x2 = layers.Dense(128, activation='relu', name="final_dense2")(x2)

    # Output layer: 'tanh'
    out = layers.Dense(
        1, activation='tanh',
        name="output"
    )(x2)

    # Build the model
    model = tf.keras.Model(
        inputs=[
            input_5m,
            input_15m,
            input_1h,
            input_google_trend,
            input_santiment,
            input_ta,
            input_signal
        ],
        outputs=out,
        name="big_multi_timeframe_model_no_dropout"
    )

    # Use a higher LR to quickly see overfitting on 500 samples
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse')

    return model
