import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers
from .environment import (
    NUM_FUTURE_STEPS,
)

################################################################################
# 0) Optional weighted loss function
################################################################################

def weighted_mse_loss(y_true, y_pred):
    """
    Example: Weighted MSE giving higher importance to nearer-term horizons.
    For NUM_FUTURE_STEPS=10, a simple static weighting:
       [2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0]

    Adjust these to suit preference (e.g. exponential decay).
    """
    # Define any weighting scheme you like
    weights = tf.constant([2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0],
                          dtype=tf.float32)
    # Reshape to broadcast along batch dimension
    weights = tf.reshape(weights, (1, -1))

    # Elementwise MSE
    err_sq = tf.square(y_true - y_pred)

    # Weighted along the horizon dimension
    weighted_err_sq = err_sq * weights
    return tf.reduce_mean(weighted_err_sq, axis=-1)  # mean across horizons

################################################################################
# 1) Residual LSTM block (with optional dropout & L2)
################################################################################

def residual_lstm_block(
    x,
    units_lstm=512,
    return_sequences=True,
    dropout_rate=0.0,
    l2_reg=1e-6,
    name_prefix="lstm_block"
):
    """
    A residual block that applies an LSTM layer + skip connection + LayerNorm.
    Includes optional dropout and L2 regularization.
    """
    # LSTM
    lstm_out = layers.LSTM(
        units_lstm,
        return_sequences=return_sequences,
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_lstm",
    )(x)

    # Optional dropout on the LSTM output
    if dropout_rate > 1e-7:
        lstm_out = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(lstm_out)

    # Residual / skip connection
    if x.shape[-1] != lstm_out.shape[-1]:
        x = layers.Dense(
            units_lstm,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name_prefix}_skip_proj"
        )(x)

    out = layers.Add(name=f"{name_prefix}_skip_add")([x, lstm_out])
    out = layers.LayerNormalization(name=f"{name_prefix}_layernorm")(out)
    return out

################################################################################
# 2) Stacked LSTM blocks
################################################################################

def stacked_residual_lstm(
    x,
    n_blocks=2,
    units_lstm1=512,
    units_lstm2=256,
    dropout_rate=0.0,
    l2_reg=1e-6,
    name_prefix="stacked_lstm"
):
    """
    Example: 2 consecutive residual LSTM blocks.
    """
    # 1st block
    out = residual_lstm_block(
        x,
        units_lstm=units_lstm1,
        return_sequences=True,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix=f"{name_prefix}_block1"
    )

    # 2nd block
    out = residual_lstm_block(
        out,
        units_lstm=units_lstm2,
        return_sequences=True,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix=f"{name_prefix}_block2"
    )

    return out

################################################################################
# 3) Transformer block (with L2 regularization)
################################################################################

def transformer_block(
    x,
    num_heads=4,
    ff_dim=512,
    dropout_rate=0.0,
    l2_reg=1e-6,
    name_prefix="transformer_block"
):
    """
    A single Transformer-style block with MHA + FFN + skip connections + LN.
    Includes optional dropout and L2 regularization on Dense layers.
    """
    # Self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1],
        kernel_regularizer=regularizers.l2(l2_reg),  # MHA can accept a reg
        name=f"{name_prefix}_mha"
    )(x, x)

    if dropout_rate > 1e-7:
        attn_output = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_mha_dropout"
        )(attn_output)

    # Residual + LayerNorm
    out1 = layers.Add(name=f"{name_prefix}_attn_add")([x, attn_output])
    out1 = layers.LayerNormalization(name=f"{name_prefix}_attn_norm")(out1)

    # Feed Forward
    ffn = layers.Dense(
        ff_dim,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_ffn_dense1"
    )(out1)
    if dropout_rate > 1e-7:
        ffn = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_ffn_dropout1"
        )(ffn)

    ffn = layers.Dense(
        out1.shape[-1],
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_ffn_dense2"
    )(ffn)
    if dropout_rate > 1e-7:
        ffn = layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_ffn_dropout2"
        )(ffn)

    # Residual + LayerNorm
    out2 = layers.Add(name=f"{name_prefix}_ffn_add")([out1, ffn])
    out2 = layers.LayerNormalization(name=f"{name_prefix}_ffn_norm")(out2)

    return out2

################################################################################
# 4) Multiple Transformer blocks
################################################################################

def stacked_transformer_blocks(
    x,
    n_transformer=2,
    num_heads=4,
    ff_dim=512,
    dropout_rate=0.0,
    l2_reg=1e-6,
    name_prefix="stacked_transformer"
):
    out = x
    for i in range(n_transformer):
        out = transformer_block(
            out,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"{name_prefix}_block{i+1}"
        )
    return out

################################################################################
# 5) Single timeframe encoder
################################################################################

def single_timeframe_encoder(
    input_tensor,
    # LSTM config
    lstm1_units=512,
    lstm2_units=256,
    # Transformer config
    n_transformer=2,
    num_heads=4,
    ff_dim=512,
    # Others
    dropout_rate=0.0,
    l2_reg=1e-6,
    name_prefix="timeframe"
):
    """
    Encodes a single time-series branch by:
      1) Stacked residual LSTMs
      2) Stacked Transformer blocks
      3) GlobalAveragePooling
      4) Dense projection
    """
    # 1) Stacked residual LSTMs
    x = stacked_residual_lstm(
        input_tensor,
        n_blocks=2,
        units_lstm1=lstm1_units,
        units_lstm2=lstm2_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix=f"{name_prefix}_lstmstack"
    )

    # 2) Stacked Transformer blocks
    x = stacked_transformer_blocks(
        x,
        n_transformer=n_transformer,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix=f"{name_prefix}_transformerstack"
    )

    # 3) Global average pooling => (batch, features)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)

    # 4) Dense projection
    x = layers.Dense(
        ff_dim,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_final_dense"
    )(x)
    if dropout_rate > 1e-7:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_final_dropout")(x)

    return x

################################################################################
# 6) Full multi-timeframe model
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
    # hyperparams
    dropout_rate=0.1,        # 10% dropout
    l2_reg=1e-6,             # L2 regularization
    lstm_units_5m=(512,256),
    lstm_units_15m=(512,256),
    lstm_units_1h=(512,256),
    lstm_units_google=(256,128),
    n_transformer=2,
    transformer_heads=8,
    transformer_ff=512,
    use_weighted_loss=True
):
    """
    Pimped multi-branch model with:
      - Stacked Residual LSTM blocks
      - Stacked Transformer blocks
      - Additional dropout
      - L2 regularization
      - Optional weighted loss
    """

    # === 5m branch ===
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = single_timeframe_encoder(
        input_5m,
        lstm1_units=lstm_units_5m[0],
        lstm2_units=lstm_units_5m[1],
        n_transformer=n_transformer,
        num_heads=transformer_heads,
        ff_dim=transformer_ff,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_5m"
    )

    # === 15m branch ===
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = single_timeframe_encoder(
        input_15m,
        lstm1_units=lstm_units_15m[0],
        lstm2_units=lstm_units_15m[1],
        n_transformer=n_transformer,
        num_heads=transformer_heads,
        ff_dim=transformer_ff,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_15m"
    )

    # === 1h branch ===
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = single_timeframe_encoder(
        input_1h,
        lstm1_units=lstm_units_1h[0],
        lstm2_units=lstm_units_1h[1],
        n_transformer=n_transformer,
        num_heads=transformer_heads,
        ff_dim=transformer_ff,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_1h"
    )

    # === google_trend branch ===
    input_google_trend = layers.Input(
        shape=(window_google_trend, feature_google_trend),
        name="input_google_trend"
    )
    x_google = single_timeframe_encoder(
        input_google_trend,
        lstm1_units=lstm_units_google[0],
        lstm2_units=lstm_units_google[1],
        n_transformer=n_transformer,
        num_heads=transformer_heads//2,  # smaller
        ff_dim=transformer_ff//2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name_prefix="enc_google"
    )

    # === Santiment (12-dim) ===
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")
    x_santiment = layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="santiment_dense1"
    )(input_santiment)
    if dropout_rate > 1e-7:
        x_santiment = layers.Dropout(dropout_rate)(x_santiment)
    x_santiment = layers.Dense(
        64, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="santiment_dense2"
    )(x_santiment)

    # === TA (63-dim) ===
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")
    x_ta = layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="ta_dense1"
    )(input_ta)
    if dropout_rate > 1e-7:
        x_ta = layers.Dropout(dropout_rate)(x_ta)
    x_ta = layers.Dense(
        64, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="ta_dense2"
    )(x_ta)

    # === Merge time-series + sentiment + TA ===
    merged_lstm_ta = layers.concatenate(
        [x_5m, x_15m, x_1h, x_google, x_santiment, x_ta],
        name="concat_lstm_ta"
    )

    # Deeper aggregator
    x = layers.Dense(
        1024, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="merged_dense1"
    )(merged_lstm_ta)
    if dropout_rate > 1e-7:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        512, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="merged_dense2"
    )(x)
    if dropout_rate > 1e-7:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        256, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="merged_dense3"
    )(x)

    # === final signal input (11-dim) ===
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    x_sig = layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="signal_dense1"
    )(input_signal)
    x_sig = layers.Dense(
        64, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="signal_dense2"
    )(x_sig)

    # === Merge aggregator with signals
    x_merged_signal = layers.concatenate([x, x_sig], name="concat_signal_branch")

    # Final dense block -> output
    x2 = layers.Dense(
        256, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="final_dense1"
    )(x_merged_signal)
    x2 = layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name="final_dense2"
    )(x2)

    out = layers.Dense(
        NUM_FUTURE_STEPS,
        activation='tanh',  # or 'linear' if you prefer
        kernel_regularizer=regularizers.l2(l2_reg),
        name="output"
    )(x2)

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
        outputs=out,
        name="mega_multi_timeframe_model"
    )

    # AdamW optimizer (from tensorflow_addons)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )

    if use_weighted_loss:
        model.compile(optimizer=optimizer, loss=weighted_mse_loss)
    else:
        # Plain MSE
        model.compile(optimizer=optimizer, loss='mse')

    return model
