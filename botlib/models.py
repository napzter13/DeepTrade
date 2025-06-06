#!/usr/bin/env python3
"""
Local/Transformers model + OpenAI fallback + advanced LSTM model building/loading.
"""

import os
import re
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import datetime
from tensorflow.keras.utils import plot_model
from .environment import (
    MIXTRAL_MODEL_PATH,
    DEEPSEEK_MODEL_PATH,
    ADVANCED_MODEL_PATH,
    OPENAI_API_KEY,
    get_logger
)
from .nn_model import build_ensemble_model, safe_mse_loss
# Add these imports to expose the classes needed by fitter.py
from .nn_model import TimeSeriesEncoder, TabularEncoder
# LightEnsembleModel was referenced but doesn't exist in nn_model.py
# Remove or comment out this import

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
    Cache GPT calls to avoid re-calling multiple times during backtest.
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
- A positive integer (e.g., 50) means "Buy that percentage of available EUR in BTC" (i.e., go long).
- A negative integer (e.g., -30) means "Sell that percentage of BTC" (i.e., go short if you have BTC, or remain 0 if you have none).
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
    

def load_advanced_lstm_model(model_5m_window=241, 
                             model_15m_window=241, 
                             model_1h_window=241, 
                             feature_dim=9, 
                             santiment_dim=12, 
                             ta_dim=63, 
                             signal_dim=11,
                             base_units=256,
                             memory_efficient=True,
                             mixed_precision=True,
                             gradient_accumulation=False,
                             gradient_accumulation_steps=8,
                             depth=6,  # Add depth parameter with default value
                             massive_model=True  # Add massive_model parameter with default value
                             ):
    if os.path.exists(ADVANCED_MODEL_PATH):
        try:
            loaded = tf.keras.models.load_model(
                ADVANCED_MODEL_PATH,
                custom_objects={"safe_mse_loss": safe_mse_loss}
            )
            logger.info("Loaded advanced multi-input LSTM model from disk.")
            return loaded
        except Exception as e:
            logger.error(f"Error loading advanced multi-input LSTM model: {e}")
    logger.warning("Creating a new advanced multi-input LSTM model from scratch...")
    m = build_ensemble_model(
            window_5m=model_5m_window,   feature_5m=feature_dim,
            window_15m=model_15m_window, feature_15m=feature_dim,
            window_1h=model_1h_window,   feature_1h=feature_dim,
            santiment_dim=santiment_dim,
            ta_dim=ta_dim,      signal_dim=signal_dim,
            base_units=base_units,
            depth=depth,  # Pass the depth parameter
            memory_efficient=memory_efficient,
            mixed_precision=mixed_precision,
            gradient_accumulation=gradient_accumulation,
            gradient_accumulation_steps=gradient_accumulation_steps,
            massive_model=massive_model)  # Pass the massive_model parameter
    
    plot_model(m, to_file="models/advanced_lstm_model_architecture.png", show_shapes=True, show_layer_names=True)
    
    try:
        m.save(ADVANCED_MODEL_PATH)
    except Exception as e:
        logger.error(f"Could not save new advanced multi-input LSTM model: {e}")
    return m