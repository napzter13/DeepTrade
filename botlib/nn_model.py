import tensorflow as tf
from tensorflow.keras import layers


def residual_lstm_block(
    x,
    units_lstm1=128,
    units_lstm2=64,
    dropout_rate=0.2,
    l2_reg=None,
    name_prefix="lstm_block"
):
    """
    A residual block that applies two LSTM layers (optionally in sequence),
    with dropout, then merges back via skip connection.
    """
    # First LSTM (return sequences to allow next layer or attention)
    lstm_1 = layers.LSTM(
        units_lstm1,
        return_sequences=True,
        kernel_regularizer=l2_reg,
        name=f"{name_prefix}_lstm1"
    )(x)
    lstm_1 = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop1")(lstm_1)

    # Second LSTM
    lstm_2 = layers.LSTM(
        units_lstm2,
        return_sequences=True,  # keep sequences for attention block
        kernel_regularizer=l2_reg,
        name=f"{name_prefix}_lstm2"
    )(lstm_1)
    lstm_2 = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop2")(lstm_2)

    # Residual / skip connection from the input x (if shapes match).
    if x.shape[-1] != lstm_2.shape[-1]:
        x = layers.Dense(
            units_lstm2, kernel_regularizer=l2_reg,
            name=f"{name_prefix}_skip_proj"
        )(x)
    out = layers.Add(name=f"{name_prefix}_skip_add")([x, lstm_2])
    return out


def transformer_block(
    x,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.2,
    l2_reg=None,
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
        kernel_regularizer=l2_reg,
        name=f"{name_prefix}_mha"
    )(x, x)
    attn_output = layers.Dropout(dropout_rate, name=f"{name_prefix}_attn_drop")(attn_output)
    # Residual + LayerNorm
    out1 = layers.Add(name=f"{name_prefix}_attn_add")([x, attn_output])
    out1 = layers.LayerNormalization(name=f"{name_prefix}_attn_norm")(out1)

    # Feed Forward
    ffn = layers.Dense(ff_dim, activation='relu', kernel_regularizer=l2_reg,
                       name=f"{name_prefix}_ffn_dense1")(out1)
    ffn = layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_drop1")(ffn)
    ffn = layers.Dense(out1.shape[-1], kernel_regularizer=l2_reg,
                       name=f"{name_prefix}_ffn_dense2")(ffn)
    ffn = layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_drop2")(ffn)

    # Residual + LayerNorm
    out2 = layers.Add(name=f"{name_prefix}_ffn_add")([out1, ffn])
    out2 = layers.LayerNormalization(name=f"{name_prefix}_ffn_norm")(out2)

    return out2


def single_timeframe_encoder(
    input_tensor,
    units_lstm1=128,
    units_lstm2=64,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.2,
    l2_reg=None,
    name_prefix="timeframe"
):
    """
    Encodes a single time-series branch (e.g. 5m, 15m, 1h, google_trend) by:
      1) Two-layer LSTM with skip-connection
      2) One Transformer block with self-attention
      3) GlobalAveragePooling to get a single vector
      4) Dense projection
    """
    # 1) Residual LSTM block
    x = residual_lstm_block(
        input_tensor,
        units_lstm1=units_lstm1,
        units_lstm2=units_lstm2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix=f"{name_prefix}_reslstm"
    )

    # 2) Transformer block
    x = transformer_block(
        x,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix=f"{name_prefix}_transformer"
    )

    # 3) Global average pooling along the time dimension => (batch, features)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)

    # 4) Final projection (dense) to unify dimension
    x = layers.Dense(
        ff_dim, activation='relu',
        kernel_regularizer=l2_reg,
        name=f"{name_prefix}_final_dense"
    )(x)
    return x


def build_multi_timeframe_model(
    window_5m=60,
    feature_5m=9,
    window_15m=60,
    feature_15m=9,
    window_1h=60,
    feature_1h=9,
    window_google_trend=8,
    feature_google_trend=1,
    santiment_dim=12,
    ta_dim=63,
    signal_dim=11
):
    """
    A significantly more advanced multi-input model combining:
      - LSTM layers with residual connections per timeframe
      - Transformer-style self-attention blocks
      - Additional feed-forward sub-networks for Santiment, TA, and signals
      - Potential for large parameter counts
    """

    # Optional small weight decay (L2 regularization)
    l2_reg = tf.keras.regularizers.l2(1e-6)
    dropout_rate = 0.2

    # === 5m branch ===
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = single_timeframe_encoder(
        input_5m,
        units_lstm1=128,
        units_lstm2=64,
        num_heads=4,
        ff_dim=128,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_5m"
    )

    # === 15m branch ===
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = single_timeframe_encoder(
        input_15m,
        units_lstm1=128,
        units_lstm2=64,
        num_heads=4,
        ff_dim=128,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_15m"
    )

    # === 1h branch ===
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = single_timeframe_encoder(
        input_1h,
        units_lstm1=128,
        units_lstm2=64,
        num_heads=4,
        ff_dim=128,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_1h"
    )

    # === google_trend branch ===
    input_google_trend = layers.Input(
        shape=(window_google_trend, feature_google_trend),
        name="input_google_trend"
    )
    x_google_trend = single_timeframe_encoder(
        input_google_trend,
        units_lstm1=64,
        units_lstm2=32,
        num_heads=2,      # maybe fewer heads for smaller input
        ff_dim=64,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_google_trend"
    )

    # === Santiment context (12-dim) ===
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    x_santiment = layers.Dense(
        64, activation='relu', kernel_regularizer=l2_reg,
        name="santiment_dense1"
    )(input_santiment)
    x_santiment = layers.Dropout(dropout_rate, name="santiment_drop1")(x_santiment)
    x_santiment = layers.Dense(
        32, activation='relu', kernel_regularizer=l2_reg,
        name="santiment_dense2"
    )(x_santiment)
    x_santiment = layers.Dropout(dropout_rate, name="santiment_drop2")(x_santiment)

    # === TA context (63-dim) ===
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    x_ta = layers.Dense(
        64, activation='relu', kernel_regularizer=l2_reg,
        name="ta_dense1"
    )(input_ta)
    x_ta = layers.Dropout(dropout_rate, name="ta_drop1")(x_ta)
    x_ta = layers.Dense(
        32, activation='relu', kernel_regularizer=l2_reg,
        name="ta_dense2"
    )(x_ta)
    x_ta = layers.Dropout(dropout_rate, name="ta_drop2")(x_ta)

    # === Merge the four big time-series encoders + Santiment + TA ===
    merged_lstm_ta = layers.concatenate(
        [x_5m, x_15m, x_1h, x_google_trend, x_santiment, x_ta],
        name="concat_lstm_ta"
    )

    # A deeper dense block on the merged features
    x = layers.Dense(
        256, activation='relu', kernel_regularizer=l2_reg,
        name="merged_dense1"
    )(merged_lstm_ta)
    x = layers.Dropout(dropout_rate, name="merged_drop1")(x)
    x = layers.Dense(
        128, activation='relu', kernel_regularizer=l2_reg,
        name="merged_dense2"
    )(x)
    x = layers.Dropout(dropout_rate, name="merged_drop2")(x)
    x = layers.Dense(
        64, activation='relu', kernel_regularizer=l2_reg,
        name="merged_dense3"
    )(x)
    x = layers.Dropout(dropout_rate, name="merged_drop3")(x)

    # === Final signals context (11-dim) ===
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    x_sig = layers.Dense(
        64, activation='relu', kernel_regularizer=l2_reg,
        name="signal_dense1"
    )(input_signal)
    x_sig = layers.Dropout(dropout_rate, name="signal_drop1")(x_sig)
    x_sig = layers.Dense(
        32, activation='relu', kernel_regularizer=l2_reg,
        name="signal_dense2"
    )(x_sig)
    x_sig = layers.Dropout(dropout_rate, name="signal_drop2")(x_sig)

    # === Merge partial network x with x_sig ===
    x_merged_signal = layers.concatenate([x, x_sig], name="concat_signal_branch")

    # Additional dense block -> output
    x2 = layers.Dense(
        128, activation='relu', kernel_regularizer=l2_reg,
        name="final_dense1"
    )(x_merged_signal)
    x2 = layers.Dropout(dropout_rate, name="final_drop1")(x2)
    x2 = layers.Dense(
        64, activation='relu', kernel_regularizer=l2_reg,
        name="final_dense2"
    )(x2)
    x2 = layers.Dropout(dropout_rate, name="final_drop2")(x2)

    # Output layer: 'tanh'
    out = layers.Dense(
        1, activation='tanh', kernel_regularizer=l2_reg,
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
        name="multi_timeframe_lstm_transformer_x1000"
    )

    # Use Adam with a smaller LR, or try AdamW, RectifiedAdam, etc.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse')

    return model
