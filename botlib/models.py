#!/usr/bin/env python3
"""
Local/Transformers model + OpenAI fallback + advanced LSTM model building/loading.
"""

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .environment import (
    MIXTRAL_MODEL_PATH,
    DEEPSEEK_MODEL_PATH,
    ADVANCED_MODEL_PATH,
    OPENAI_API_KEY,
    get_logger
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import datetime

logger = get_logger("Models")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

###############################################################################
# Local HF Models + OpenAI fallback
###############################################################################
def load_local_hf_model():
    """
    Try loading local Mixtral model. If not found, try DeepSeek.
    Return (tokenizer, model, model_type) or (None, None, None) if no success.
    """
    if os.path.exists(MIXTRAL_MODEL_PATH):
        try:
            tok = AutoTokenizer.from_pretrained(MIXTRAL_MODEL_PATH)
            mod = AutoModelForCausalLM.from_pretrained(MIXTRAL_MODEL_PATH)
            logger.info(f"Loaded local Mixtral model from {MIXTRAL_MODEL_PATH}")
            return tok, mod, "mixtral"
        except Exception as e:
            logger.error(f"Error loading Mixtral model: {e}")

    if os.path.exists(DEEPSEEK_MODEL_PATH):
        try:
            tok = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
            mod = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL_PATH)
            logger.info(f"Loaded local DeepSeek model from {DEEPSEEK_MODEL_PATH}")
            return tok, mod, "deepseek"
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {e}")

    # None found
    logger.warning("No local Mixtral/DeepSeek found; fallback to ChatGPT 3.5.")
    return None, None, None


def get_local_model_assessment(
    gpt_cache,
    tokenizer,
    model,
    model_type,
    type,
    prompt,
    temperature=0.2,
    timestamp = None,
    use_real_gpt = True
):
    """
    Cache local GPT calls to avoid re-calling multiple times during backtest.
    """
    if timestamp == None:
        key_dt = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    else:
        key_dt = timestamp.replace(minute=0, second=0, microsecond=0)
    p_hash = str(key_dt) + "|" + str(type)
    if timestamp != None:
        if p_hash in gpt_cache:
            cached_val = gpt_cache[p_hash]
            try:
                return float(cached_val)
            except:
                logger.warning("Cached GPT response not numeric => 0.0: ", cached_val)
        
    """
    Query local HF model if available; otherwise fallback to OpenAI.
    Return integer in [-100,100].
    """
    if tokenizer and model:
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            output_ids = model.generate(input_ids, max_length=256, temperature=temperature)
            out_txt = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            match = re.search(r'-?\d+(\.\d+)?', out_txt)
            if match:
                val = float(match.group(0))
                return max(min(val, 100), -100)
            else:
                logger.warning("No numeric in local model output. Fallback to OpenAI.")
        except Exception as e:
            logger.error(f"Local model error: {e}")
            
    if use_real_gpt == False:
        return np.nan

    # fallback
    val = get_openai_assessment(type, prompt, temperature)
    gpt_cache[p_hash] = str(val)
    return val


def get_openai_assessment(type, prompt, temperature=0.2):
    """
    Use OpenAI ChatGPT (3.5-turbo) to get an integer in [-100,100].
    """
    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a specialized crypto analyst AI.
You receive updated market and sentiment data for BTC every hour. Your role is to forecast BTC price movement over the next 3 hours and recommend a position size (buy or sell) based on short-term (5m, 15m, 1h) technical analysis signals and sentiment indicators.

**Output Requirements**:
- Output must be exactly ONE integer.
- This integer must be in the range [-100, 100].
- A positive integer (e.g., 50) means “Buy that percentage of available EUR in BTC” (i.e., go long).
- A negative integer (e.g., -30) means “Sell that percentage of BTC” (i.e., go short if you have BTC, or remain 0 if you have none).
- Do not include any text, explanations, disclaimers, or any other characters besides this integer.

If you are unsure, provide your best estimate within the specified range."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=20   # Low since we just want an integer as response.
        )
        txt = resp.choices[0].message.content.strip()

        # Debug store
        with open(f"debug/prompt_"+type+".txt", "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n\n{prompt}\n\n\n\nRESPONSE:\n\n{txt}")
            
        match = re.search(r'-?\d+(\.\d+)?', txt)
        if match:
            val = float(match.group(0))
            return max(min(val, 100), -100)
        return 0.0
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return 0.0

###############################################################################
# Advanced LSTM Model
###############################################################################

def build_multi_timeframe_model(
    window_5m=60,    # number of timesteps for 5m input
    feature_5m=9,    # # of features per bar in 5m
    window_15m=60,   
    feature_15m=9,
    window_1h=60,
    feature_1h=9,
    window_google_trend=8,
    feature_google_trend=1,
    santiment_dim=12,
    ta_dim=63,       # size of TA context vector
    signal_dim=11    # size of "signals" context vector
):
    """
    Multi-input model with:
      1) 5m LSTM branch
      2) 15m LSTM branch
      3) 1h LSTM branch
      4) google_trend LSTM branch
      5) santiment input (12-dim)
      6) TA context input (63-dim)
      7) Signal context input (11-dim)

    Steps:
      - Each LSTM branch -> a small embedding (dense) -> we merge them with the TA context branch
      - Then we have an intermediate dense block
      - Then we incorporate the final 'signal context' (the 11-dim) in a second merge
      - Then produce a single float in [-1,1]
    """

    # === 5m branch ===
    input_5m = layers.Input(shape=(window_5m, feature_5m), name="input_5m")
    x_5m = layers.LSTM(64, return_sequences=True)(input_5m)
    x_5m = layers.Dropout(0.2)(x_5m)
    x_5m = layers.LSTM(32, return_sequences=False)(x_5m)
    x_5m = layers.Dropout(0.2)(x_5m)
    x_5m = layers.Dense(32, activation='relu')(x_5m)  # final 5m embedding

    # === 15m branch ===
    input_15m = layers.Input(shape=(window_15m, feature_15m), name="input_15m")
    x_15m = layers.LSTM(64, return_sequences=True)(input_15m)
    x_15m = layers.Dropout(0.2)(x_15m)
    x_15m = layers.LSTM(32, return_sequences=False)(x_15m)
    x_15m = layers.Dropout(0.2)(x_15m)
    x_15m = layers.Dense(32, activation='relu')(x_15m)  # final 15m embedding

    # === 1h branch ===
    input_1h = layers.Input(shape=(window_1h, feature_1h), name="input_1h")
    x_1h = layers.LSTM(64, return_sequences=True)(input_1h)
    x_1h = layers.Dropout(0.2)(x_1h)
    x_1h = layers.LSTM(32, return_sequences=False)(x_1h)
    x_1h = layers.Dropout(0.2)(x_1h)
    x_1h = layers.Dense(32, activation='relu')(x_1h)  # final 1h embedding

    # === google_trend branch ===
    input_google_trend = layers.Input(shape=(window_google_trend, feature_google_trend), name="input_google_trend")
    x_google_trend = layers.LSTM(64, return_sequences=True)(input_google_trend)
    x_google_trend = layers.Dropout(0.2)(x_google_trend)
    x_google_trend = layers.LSTM(32, return_sequences=False)(x_google_trend)
    x_google_trend = layers.Dropout(0.2)(x_google_trend)
    x_google_trend = layers.Dense(32, activation='relu')(x_google_trend)  # final google_trend embedding

    # === Santiment context (12-dim) ===
    input_santiment = layers.Input(shape=(santiment_dim,), name="input_santiment")  
    x_santiment = layers.Dense(32, activation='relu')(input_santiment)
    x_santiment = layers.Dropout(0.2)(x_santiment)
    x_santiment = layers.Dense(16, activation='relu')(x_santiment)
    
    # === TA context (63-dim) ===
    input_ta = layers.Input(shape=(ta_dim,), name="input_ta")  
    x_ta = layers.Dense(32, activation='relu')(input_ta)
    x_ta = layers.Dropout(0.2)(x_ta)
    x_ta = layers.Dense(16, activation='relu')(x_ta)

    # Merge the three LSTM branches + the TA context
    merged_lstm_ta = layers.concatenate([x_5m, x_15m, x_1h, x_google_trend, x_santiment, x_ta], name="concat_lstm_ta")

    # A few dense layers on that merged block
    x = layers.Dense(64, activation='relu')(merged_lstm_ta)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # === Final signals context (11-dim) ===
    input_signal = layers.Input(shape=(signal_dim,), name="input_signal")
    x_sig = layers.Dense(16, activation='relu')(input_signal)
    x_sig = layers.Dropout(0.2)(x_sig)

    # Merge the partial network x with x_sig
    x_merged_signal = layers.concatenate([x, x_sig], name="concat_signal")

    # Additional dense -> output
    x2 = layers.Dense(64, activation='relu')(x_merged_signal)
    x2 = layers.Dropout(0.2)(x2)
    out = layers.Dense(1, activation='tanh', name="output")(x2)

    # Build the model with inputs => 1 output
    model = tf.keras.Model(
        inputs=[input_5m, input_15m, input_1h, input_google_trend, input_santiment, input_ta, input_signal],
        outputs=out
    )
    model.compile(optimizer='adam', loss='mse')
    return model



def load_advanced_lstm_model(model_5m_window=60, model_15m_window=60, model_1h_window=60, feature_dim=9, santiment_dim=12, ta_dim=63, signal_dim=11):
    if os.path.exists(ADVANCED_MODEL_PATH):
        try:
            loaded = tf.keras.models.load_model(ADVANCED_MODEL_PATH)
            logger.info("Loaded advanced multi-input LSTM model from disk.")
            return loaded
        except Exception as e:
            logger.error(f"Error loading advanced multi-input LSTM model: {e}")
    logger.warning("Creating a new advanced multi-input LSTM model from scratch...")
    m = build_multi_timeframe_model(
            window_5m=model_5m_window,   feature_5m=feature_dim,
            window_15m=model_15m_window, feature_15m=feature_dim,
            window_1h=model_1h_window,   feature_1h=feature_dim,
            santiment_dim=santiment_dim,
            ta_dim=ta_dim,      signal_dim=signal_dim)
    try:
        m.save(ADVANCED_MODEL_PATH)
    except Exception as e:
        logger.error(f"Could not save new advanced multi-input LSTM model: {e}")
    return m
