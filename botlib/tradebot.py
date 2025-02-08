#!/usr/bin/env python3
"""
tradebot.py

TradingBot class that ties it all together:
- Initialization (Binance client, local model, advanced LSTM, RL agent)
- Data fetching (calls out to datafetchers.py)
- Technical indicators (calls out to indicators.py)
- Position management (paper or live) with 0.1% fee
- RL-based action decisions (Uses DQN)

Keeps all aggregator logic intact.
"""

import os
import csv
import datetime
import traceback
import json
import threading

import numpy as np

from binance.client import Client
from binance.enums import *

from .environment import (
    PAPER_TRADING,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    GPT_CACHE_FILE,
    get_logger
)
from .datafetchers import (
    get_klines as df_get_klines,
    fetch_price_at_hour as df_fetch_price_at_hour,
    fetch_order_book as df_fetch_order_book,
    fetch_news_data as df_fetch_news_data,
    fetch_google_trends as df_fetch_google_trends,
    fetch_santiment_data as df_fetch_santiment_data,
    fetch_reddit_sentiment as df_fetch_reddit_sentiment,
    analyze_news_sentiment
)
from .indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_stoch_rsi,
    compute_adx,
    compute_obv,
    compute_atr_from_klines,
    compute_sma,
    compute_ema,
    compute_cci,
    compute_mfi,
    compute_williams_r,
    compute_ichimoku,
    compute_parabolic_sar
)
from .rl import DQNAgent
from .models import (
    load_local_hf_model,
    get_local_model_assessment as m_get_local_model_assessment,
    load_advanced_lstm_model
)
from botlib.input_preprocessing import ModelScaler, prepare_for_model_inputs


class TradingBot:
    def __init__(self):
        # Setup logger
        self.logger = get_logger("AdvancedTradingBot")

        # Local / HF model
        self.local_tokenizer, self.local_model, self.local_model_type = load_local_hf_model()
        self.use_openai_fallback = (self.local_model is None)
        
        # GPT Cache
        self.gpt_cache = {}
        if os.path.exists(GPT_CACHE_FILE):
            with open(GPT_CACHE_FILE, "r", encoding="utf-8") as f:
                try:
                    self.gpt_cache = json.load(f)
                except:
                    self.logger.warning("Cannot parse gpt_cache.json => starting empty.")
                    self.gpt_cache = {}
        else:
            self.logger.info("No gpt_cache found => create new one.")

        # Additional API keys
        self.news_api_key = os.getenv("NEWS_API_KEY", None)
        self.binance_api_key = BINANCE_API_KEY
        self.binance_api_secret = BINANCE_API_SECRET

        # Binance
        self.binance_client = Client(self.binance_api_key, self.binance_api_secret)
        self.symbol = "BTCEUR"

        # Basic params
        self.max_trade_fraction = 0.1
        self.min_trade_eur_trade = 10.0
        self.timeframes = [
            Client.KLINE_INTERVAL_5MINUTE,
            Client.KLINE_INTERVAL_15MINUTE,
            Client.KLINE_INTERVAL_1HOUR,
            Client.KLINE_INTERVAL_4HOUR,
            Client.KLINE_INTERVAL_1DAY
        ]
        self.lookback_days = 7

        # CSV logging
        self.trade_log_file = "output/txs.csv"
        self._initialize_trade_log()
        
        # load the scalers that we fitted in training
        try:
            self.model_scaler = ModelScaler.load("models/scalers.pkl")
            self.logger.info("Loaded model scalers from models/scalers.pkl.")
        except FileNotFoundError:
            self.logger.warning("No scalers found at models/scalers.pkl. Using pass-through (unfitted) ModelScaler.")
            self.model_scaler = ModelScaler()  # pass-thru that does no scaling

        # Paper or live
        if PAPER_TRADING:
            self.simulated_balances = {"BTC": 0.0, "EUR": 10000.0}
            self.current_position = None
        self.last_log_data = {}

        # New DQN agent for advanced RL:
        # Example state_dim=4: [final_signal/100, atr%, BTC fraction, EUR fraction]
        self.rl_state_dim = 4
        self.rl_agent = DQNAgent(
            state_dim=self.rl_state_dim,
            gamma=0.99,
            lr=0.0005,
            batch_size=32,
            max_memory=10000
        )

        self.last_equity = None  # for PnL-based reward

        # Multi-timeframe LSTM model config
        self.model_5m_window = 60     # how many bars for 5m
        self.model_15m_window = 60    # how many bars for 15m
        self.model_1h_window = 60     # how many bars for 1h
        self.num_features_per_bar = 9  # e.g. [open, high, low, close, volume, quote_volume, trades, taker_base_volume, taker_quote_volume]

        self.santiment_dim = 12        # Santiment features
        self.context_ta_dim = 63       # TA features (21 per timeframe => 63 total)
        self.context_sig_dim = 11      # 11 signals for price, google, local_gpt, equity, etc.

        # Load advanced input LSTM model
        self.advanced_model = load_advanced_lstm_model(
            model_5m_window=self.model_5m_window,
            model_15m_window=self.model_15m_window,
            model_1h_window=self.model_1h_window,
            feature_dim=self.num_features_per_bar,
            santiment_dim=self.santiment_dim,
            ta_dim=self.context_ta_dim,
            signal_dim=self.context_sig_dim
        )

    def _initialize_trade_log(self):
        """
        Creates txs.csv with headers if it does not exist.
        """
        try:
            os.makedirs(os.path.dirname(self.trade_log_file), exist_ok=True)
            with open(self.trade_log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "action", "price", "size_EUR", "size_BTC",
                    "signal", "notes", "BTC_balance", "EUR_balance"
                ])
        except Exception as e:
            self.logger.error(f"Error initializing trade log: {e}")

    def save_gpt_cache(self):
        """
        Save updated GPT cache to disk each iteration.
        """
        try:
            with open(GPT_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.gpt_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Cannot save GPT cache: {e}")

    ###########################################################################
    # Overriden methods for backtesting or live usage
    ###########################################################################
    
    def fetch_klines(self, symbol="BTCEUR", interval=Client.KLINE_INTERVAL_5MINUTE, limit=60, end_dt:datetime.datetime=None):
        return df_get_klines(self.binance_client, symbol=symbol, interval=interval, limit=limit, end_dt=end_dt)
    
    def fetch_price_at_hour(self, symbol="BTCEUR", dt: datetime.datetime = datetime.datetime.now()):
        return df_fetch_price_at_hour(self.binance_client, symbol, dt)

    def fetch_order_book(self, symbol="BTCEUR", limit=20, dt: datetime.datetime = None):
        return df_fetch_order_book(self.binance_client, symbol, limit, dt)
    
    def fetch_news_data(self, days=1, end_dt=None):
        return df_fetch_news_data(days, end_dt)
        
    def fetch_google_trends(self, end_dt=None):
        return df_fetch_google_trends(end_dt)
        
    def fetch_santiment_data(self, end_dt=None, only_look_in_cache = False):
        return df_fetch_santiment_data(end_dt, only_look_in_cache)
        
    def fetch_reddit_sentiment(self, dt: datetime.datetime = None):
        return df_fetch_reddit_sentiment(dt)
        
    def get_local_model_assessment(
        self,
        gpt_cache,
        tokenizer,
        model,
        model_type,
        type,
        prompt,
        temperature,
        timestamp = None,
        use_real_gpt = True
    ):
        return m_get_local_model_assessment(
            gpt_cache,
            tokenizer,
            model,
            model_type,
            type,
            prompt,
            temperature,
            timestamp,
            use_real_gpt
        )

    ###########################################################################
    # LSTM Inference (5 inputs) + aggregator
    ###########################################################################
    def get_input_signal(self, arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx):
        """
        Inference using the advanced multi-input LSTM model:
          - arr_5m  -> shape (1, 60, 9)
          - arr_15m -> shape (1, 60, 9)
          - arr_1h  -> shape (1, 60, 9)
          - arr_google_trend -> shape (1,8,1)
          - arr_santiment -> shape (1,12)
          - arr_ta   -> shape (1,63)
          - arr_ctx  -> shape (1,11)

        Returns a float in [-100, 100].
        """
        s_5m, s_15m, s_1h, s_gt, s_sa, s_ta, s_ctx = prepare_for_model_inputs(
            arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta, arr_ctx, self.model_scaler
        )
        
        try:
            raw_pred = self.advanced_model.predict(
                [s_5m, s_15m, s_1h, s_gt, s_sa, s_ta, s_ctx],
                verbose=0
            )[0][0]
            signal = max(min(raw_pred * 100, 100), -100)
            return signal
        except Exception as e:
            self.logger.error(f"5-input LSTM predict error: {e}")
            return 0.0

    ###########################################################################
    # Aggregation
    ###########################################################################
    
    def get_aggregated_signal(
        self,
        current_utc_time,
        current_price,
        klines_5m,
        klines_15m,
        klines_1h,
        news_articles,
        ob_data,
        google_trend,
        balances,
        santiment_data,
        reddit_sent,
        use_real_gpt = True,
        use_model_pred = True
    ):
        """
        A 'mega' aggregator that:
        ) Analyzes news with GPT
        ) Builds a big textual prompt incl. multi-timeframe TAs (all 21 TAs each for 5m,15m,1h),
            order book ratio, google trends, santiment, reddit sentiment, etc.
        ) Generates GPT signals
        ) Computes TAs for each timeframe => 63 features
        ) Builds an 11-feature context vector
        ) Feeds all inputs (3 time-series arrays + 63 TA + 11 context + etc...) into LSTM => final signal
        ) Returns final_signal, local_gpt_signals, news_signal plus arrays for logging, plus the aggregator prompt
        """

        # ) News GPT
        newsapi_sent = analyze_news_sentiment(news_articles)
        
        # ) Start building the textual prompt
        prompt_lines = []
        prompt_lines.append("Focus on advanced short-term volatility and multi-timeframe TA.")
        prompt_lines.append("Predict BTC price action for the next hours. Provide a single integer between -100 and 100.")
        prompt_lines.append("")
        prompt_lines.append(f"Analyzing pair: {self.symbol}")
        # prompt_lines.append(f"Current BTC balance: {balances['BTC']}")
        # prompt_lines.append(f"Current EUR balance: {balances['EUR']}")
        prompt_lines.append(f"UTC Timestamp: {current_utc_time}")
        prompt_lines.append(f"Current Price: {current_price:.2f}")
        ratio = ob_data.get("bid_ask_volume_ratio", 0.5)
        prompt_lines.append(f"Order Book Ratio: {ratio:.2f}")
        if (google_trend[-1] > 0):
            prompt_lines.append(f"Google Trends score per Day (0 → 100 scale, oldest → newest): {google_trend}")
        prompt_lines.append(f"Bitcoin News VADER Sentiment (-100 → 100 scale): {newsapi_sent:.2f}")
        prompt_lines.append(f"Bitcoin Reddit VADER Sentiment (-100 → 100 scale): {reddit_sent:.2f}")
        
        metrics = [
            ("social_volume_total", "Social Volume Total"),
            ("social_dominance", "Social Dominance"),
            ("sentiment_positive", "Sentiment Positive"),
            ("sentiment_negative", "Sentiment Negative"),
            ("daily_active_addresses", "Daily Active Addresses"),
            ("transaction_volume", "Transaction Volume"),
            ("exchange_inflow", "Exchange Inflow"),
            ("exchange_outflow", "Exchange Outflow"),
            ("whale_transaction_count", "Whale Transaction Count"),
            ("mvrv_ratio", "MVRV Ratio"),
            ("nvt_ratio", "NVT Ratio"),
            ("dev_activity", "Dev Activity")
        ]
        for key, label in metrics:
            if key in santiment_data and santiment_data[key] > 0:
                prompt_lines.append(f"Santiment Metric ({label}): {santiment_data[key]:.2f}")
                
        prompt_lines.append("")

        # We'll store time-series arrays as zeros if we lack exactly 60 bars
        arr_5m  = np.zeros((1,self.model_5m_window,self.num_features_per_bar), dtype=np.float32)
        arr_15m = np.zeros((1,self.model_15m_window,self.num_features_per_bar), dtype=np.float32)
        arr_1h  = np.zeros((1,self.model_1h_window,self.num_features_per_bar), dtype=np.float32)
        arr_google_trend  = np.zeros((1,8,1), dtype=np.float32)
        arr_santiment = [0.0]*12
        ta_63_list = [0.0]*63
        arr_ta_63  = np.array(ta_63_list, dtype=np.float32).reshape((1,63))

        # Utility to build array from klines
        def klines_to_array(klines):
            arr = []
            for k in klines:
                o = float(k[0]); h = float(k[1]); l = float(k[2])
                c = float(k[3]); v = float(k[4]); qv = float(k[5])
                t = float(k[6]); tbav = float(k[7]); tqav = float(k[8])
                arr.append([o,h,l,c,v,qv,t,tbav,tqav])
            return np.array(arr, dtype=np.float32)
        
        # We'll define a helper to format the last N bars of klines:
        def format_klines(klines, label, max_bars=20):
            """
            klines is the usual Binance array of [openTime, open, high, low, close, volume, ...]
            We'll output the last `max_bars` lines into a string array to embed in prompt.
            label is a string like '5m' or '15m' or '1h'.
            """
            lines = []
            N = len(klines)
            start = max(0, N - max_bars)
            lines.append(f"--- Last {max_bars} {label} Klines (oldest → newest) ---")
            for i in range(start, N):
                bar = klines[i]
                open_   = float(bar[0])
                high_   = float(bar[1])
                low_    = float(bar[2])
                close_  = float(bar[3])
                volume_ = float(bar[4])
                quote_volume_ = float(bar[5])
                trades = float(bar[6])
                taker_base_volume = float(bar[7])
                taker_quote_volume = float(bar[8])
                lines.append(
                    f"{label} bar {i}: Open={open_:.2f}, High={high_:.2f}, Low={low_:.2f}, Close={close_:.2f}, Vol={volume_:.2f}, QuoteVol={quote_volume_:.2f}, Trades={trades:.2f}, TakerBase={taker_base_volume:.2f}, TakerQuote={taker_quote_volume:.2f}"
                )
            return lines

        arr = []
        for k in google_trend:
            arr.append([k])
        arr_google_trend = np.array(arr, dtype=np.float32).reshape((1,8,1))
        
        arr = []
        for k in santiment_data.values():
            arr.append(k)
        arr_santiment = np.array(arr, dtype=np.float32).reshape((1,12))
        
        # ) If we have 60 bars in each timeframe, do full TAs & arrays
        if len(klines_5m) == 60 and len(klines_15m) == 60 and len(klines_1h) == 60:
            arr_5m  = klines_to_array(klines_5m).reshape((1,self.model_5m_window,self.num_features_per_bar))
            arr_15m = klines_to_array(klines_15m).reshape((1,self.model_15m_window,self.num_features_per_bar))
            arr_1h  = klines_to_array(klines_1h).reshape((1,self.model_1h_window,self.num_features_per_bar))
        
            prompt_lines.extend(format_klines(klines_5m, '5m', max_bars=20))
            prompt_lines.append("")
            prompt_lines.extend(format_klines(klines_15m, '15m', max_bars=20))
            prompt_lines.append("")
            prompt_lines.extend(format_klines(klines_1h, '1h', max_bars=20))
            prompt_lines.append("")

            # TAs => 63
            ta_5m  = self.compute_timeframe_tas(klines_5m)
            ta_15m = self.compute_timeframe_tas(klines_15m)
            ta_1h  = self.compute_timeframe_tas(klines_1h)
            ta_63_list = self.build_ta_context_63(ta_5m, ta_15m, ta_1h)
            arr_ta_63  = np.array(ta_63_list, dtype=np.float32).reshape((1,63))

            # Add them all to prompt
            prompt_lines.append("--- 5m TAs ---")
            prompt_lines.append(
                f"rsi_5m={ta_5m['rsi']:.2f}, macd_5m={ta_5m['macd']:.2f}, macd_hist_5m={ta_5m['macd_hist']:.2f}, "
                f"bb_up_5m={ta_5m['bb_up']:.2f}, bb_mid_5m={ta_5m['bb_mid']:.2f}, bb_low_5m={ta_5m['bb_low']:.2f}, "
                f"stoch_k_5m={ta_5m['stoch_k']:.2f}, stoch_d_5m={ta_5m['stoch_d']:.2f}, adx_5m={ta_5m['adx']:.2f}, obv_5m={ta_5m['obv']:.2f}, "
                f"atr_5m={ta_5m['atr']:.2f}, sma_20_5m={ta_5m['sma20']:.2f}, sma_50_5m={ta_5m['sma50']:.2f}, "
                f"ema_20_5m={ta_5m['ema20']:.2f}, ema_50_5m={ta_5m['ema50']:.2f}, cci_5m={ta_5m['cci']:.2f}, mfi_5m={ta_5m['mfi']:.2f}, "
                f"williams_r_5m={ta_5m['williams_r']:.2f}, ichimoku_conversion_5m={ta_5m['ichimoku_conv']:.2f}, "
                f"ichimoku_base_5m={ta_5m['ichimoku_base']:.2f}, parabolic_sar_5m={ta_5m['parabolic_sar']:.2f}"
            )
            prompt_lines.append("")

            prompt_lines.append("--- 15m TAs ---")
            prompt_lines.append(
                f"rsi_15m={ta_15m['rsi']:.2f}, macd_15m={ta_15m['macd']:.2f}, macd_hist_15m={ta_15m['macd_hist']:.2f}, "
                f"bb_up_15m={ta_15m['bb_up']:.2f}, bb_mid_15m={ta_15m['bb_mid']:.2f}, bb_low_15m={ta_15m['bb_low']:.2f}, "
                f"stoch_k_15m={ta_15m['stoch_k']:.2f}, stoch_d_15m={ta_15m['stoch_d']:.2f}, adx_15m={ta_15m['adx']:.2f}, obv_15m={ta_15m['obv']:.2f}, "
                f"atr_15m={ta_15m['atr']:.2f}, sma_20_15m={ta_15m['sma20']:.2f}, sma_50_15m={ta_15m['sma50']:.2f}, "
                f"ema_20_15m={ta_15m['ema20']:.2f}, ema_50_15m={ta_15m['ema50']:.2f}, cci_15m={ta_15m['cci']:.2f}, mfi_15m={ta_15m['mfi']:.2f}, "
                f"williams_r_15m={ta_15m['williams_r']:.2f}, ichimoku_conversion_15m={ta_15m['ichimoku_conv']:.2f}, "
                f"ichimoku_base_15m={ta_15m['ichimoku_base']:.2f}, parabolic_sar_15m={ta_15m['parabolic_sar']:.2f}"
            )
            prompt_lines.append("")

            prompt_lines.append("--- 1h TAs ---")
            prompt_lines.append(
                f"rsi_1h={ta_1h['rsi']:.2f}, macd_1h={ta_1h['macd']:.2f}, macd_hist_1h={ta_1h['macd_hist']:.2f}, "
                f"bb_up_1h={ta_1h['bb_up']:.2f}, bb_mid_1h={ta_1h['bb_mid']:.2f}, bb_low_1h={ta_1h['bb_low']:.2f}, "
                f"stoch_k_1h={ta_1h['stoch_k']:.2f}, stoch_d_1h={ta_1h['stoch_d']:.2f}, adx_1h={ta_1h['adx']:.2f}, obv_1h={ta_1h['obv']:.2f}, "
                f"atr_1h={ta_1h['atr']:.2f}, sma_20_1h={ta_1h['sma20']:.2f}, sma_50_1h={ta_1h['sma50']:.2f}, "
                f"ema_20_1h={ta_1h['ema20']:.2f}, ema_50_1h={ta_1h['ema50']:.2f}, cci_1h={ta_1h['cci']:.2f}, mfi_1h={ta_1h['mfi']:.2f}, "
                f"williams_r_1h={ta_1h['williams_r']:.2f}, ichimoku_conversion_1h={ta_1h['ichimoku_conv']:.2f}, "
                f"ichimoku_base_1h={ta_1h['ichimoku_base']:.2f}, parabolic_sar_1h={ta_1h['parabolic_sar']:.2f}"
            )
            prompt_lines.append("")
        else:
            prompt_lines.append("[WARNING] Not enough bars for 5m/15m/1h => fallback zeros for TAs")
        
        prompt_lines.append("--- Latest 20 News Articles Headlines ---")
        for article in news_articles[:20]:
            prompt_lines.append(f"- {article['title']}: {article['description']} | By {article['author']} ({article['source']['name']}) | {article['publishedAt']}")

        aggregator_prompt = "\n".join(prompt_lines)
        
        # ) GPT signals from aggregator_prompt
        temperature_variants = [0.1, 0.5]
        local_signals = []
        for temp in temperature_variants:
            val = self.get_local_model_assessment(
                self.gpt_cache,
                self.local_tokenizer,
                self.local_model,
                self.local_model_type,
                str(temp),
                aggregator_prompt,
                temp,
                None,
                use_real_gpt
            )
            local_signals.append(val)
        self.logger.info(f"[Aggregator] GPT signals => {local_signals}")

        # ) Build the 11-dim context
        eq_btc_val = balances["BTC"] * current_price
        eq_eur_val = balances["EUR"]
        tot_equity = eq_btc_val + eq_eur_val
        if tot_equity > 0:
            btc_pct = eq_btc_val / tot_equity
            eur_pct = eq_eur_val / tot_equity
        else:
            btc_pct = 0.0
            eur_pct = 1.0

        sig0, sig1 = local_signals
        ctx_11_list = self.build_signal_context_11(
            current_price,
            float(google_trend[-1]),
            float(reddit_sent),
            float(ratio),
            sig0,
            sig1,
            0,              # Empty context value!
            newsapi_sent,
            0,              # Empty context value!
            btc_pct,
            eur_pct
        )
        arr_ctx_11 = np.array(ctx_11_list, dtype=np.float32).reshape((1, self.context_sig_dim))

        if use_model_pred:
            # ) Multi-input LSTM => final_pred
            final_pred = self.get_input_signal(
                arr_5m,    # shape=(1,60,9)
                arr_15m,   # shape=(1,60,9)
                arr_1h,    # shape=(1,60,9)
                arr_google_trend,    # shape=(1,8,1)
                arr_santiment,    # shape=(1,12)
                arr_ta_63, # shape=(1,63)
                arr_ctx_11 # shape=(1,11)
            )
            self.logger.info(f"[Aggregator] Model pred => {final_pred:.2f}")
        else:
            final_pred = 0 #

        # ) Combine GPT average with final_pred
        # gpt_avg = sum(local_signals)/len(local_signals)
        # final_signal = 0.5*final_pred + 0.5*gpt_avg
        # self.logger.info(f"[Aggregator] final combined => {final_signal:.2f}")

        # Return final signal plus everything needed by run_cycle
        return final_pred, local_signals, newsapi_sent, arr_5m, arr_15m, arr_1h, arr_google_trend, arr_santiment, arr_ta_63, arr_ctx_11, aggregator_prompt


    def compute_timeframe_tas(self, klines):
        """
        Compute the 21 TAs for a single timeframe. 
        Returns a dict of 21 TAs:
          rsi, macd, macd_hist, bb_up, bb_mid, bb_low, stoch_k, stoch_d, adx, obv,
          atr, sma20, sma50, ema20, ema50, cci, mfi, williams_r, ichimoku_conv, 
          ichimoku_base, parabolic_sar
        """
        highs  = [float(k[1]) for k in klines]
        lows   = [float(k[2]) for k in klines]
        closes = [float(k[3]) for k in klines]
        volumes= [float(k[4]) for k in klines]

        # RSI
        rsi_val = compute_rsi(closes, 14) or 50
        # MACD
        macd_line, macd_signal, macd_hist = compute_macd(closes, 12, 26, 9)
        if macd_line is None: 
            macd_line, macd_signal, macd_hist = 0,0,0
        # Bollinger
        bbs = compute_bollinger_bands(closes, 20, 2) or (0,0,0)
        bb_mid, bb_up, bb_low = bbs[0], bbs[1], bbs[2]
        # Stoch RSI
        s_k, s_d = compute_stoch_rsi(closes, 14, 3, 3)
        if s_k is None: s_k = 50
        if s_d is None: s_d = 50
        # ADX
        adx_val = compute_adx(klines, 14) or 20
        # OBV
        obv_val = compute_obv(closes, volumes) or 0
        # ATR
        atr_val = compute_atr_from_klines(klines, 14) or 0

        sma20_val = compute_sma(closes, 20) or 0
        sma50_val = compute_sma(closes, 50) or 0
        ema20_arr = compute_ema(closes, 20) or [0]
        ema20_val = ema20_arr[-1] if len(ema20_arr)>0 else 0
        ema50_arr = compute_ema(closes, 50) or [0]
        ema50_val = ema50_arr[-1] if len(ema50_arr)>0 else 0
        cci_val   = compute_cci(highs, lows, closes, 20) or 0
        mfi_val   = compute_mfi(highs, lows, closes, volumes, 14) or 0
        w_r       = compute_williams_r(highs, lows, closes, 14) or 0
        ichimoku_conv, ichimoku_base = compute_ichimoku(highs, lows, closes)
        if ichimoku_conv is None: ichimoku_conv=0
        if ichimoku_base is None: ichimoku_base=0
        p_sar_val = compute_parabolic_sar(highs, lows, closes) or 0

        return {
            "rsi": rsi_val,
            "macd": macd_line,
            "macd_hist": macd_hist,
            "bb_up": bb_up,
            "bb_mid": bb_mid,
            "bb_low": bb_low,
            "stoch_k": s_k,
            "stoch_d": s_d,
            "adx": adx_val,
            "obv": obv_val,
            "atr": atr_val,
            "sma20": sma20_val,
            "sma50": sma50_val,
            "ema20": ema20_val,
            "ema50": ema50_val,
            "cci": cci_val,
            "mfi": mfi_val,
            "williams_r": w_r,
            "ichimoku_conv": ichimoku_conv,
            "ichimoku_base": ichimoku_base,
            "parabolic_sar": p_sar_val
        }

    def build_ta_context_63(self, ta_5m, ta_15m, ta_1h):
        """
        Merges the 21 TAs from each timeframe => 63 total
        """
        seq_5m = [
            ta_5m["rsi"], ta_5m["macd"], ta_5m["macd_hist"],
            ta_5m["bb_up"], ta_5m["bb_mid"], ta_5m["bb_low"],
            ta_5m["stoch_k"], ta_5m["stoch_d"], ta_5m["adx"], ta_5m["obv"],
            ta_5m["atr"], ta_5m["sma20"], ta_5m["sma50"], ta_5m["ema20"], ta_5m["ema50"],
            ta_5m["cci"], ta_5m["mfi"], ta_5m["williams_r"], ta_5m["ichimoku_conv"],
            ta_5m["ichimoku_base"], ta_5m["parabolic_sar"]
        ]
        seq_15m = [
            ta_15m["rsi"], ta_15m["macd"], ta_15m["macd_hist"],
            ta_15m["bb_up"], ta_15m["bb_mid"], ta_15m["bb_low"],
            ta_15m["stoch_k"], ta_15m["stoch_d"], ta_15m["adx"], ta_15m["obv"],
            ta_15m["atr"], ta_15m["sma20"], ta_15m["sma50"], ta_15m["ema20"], ta_15m["ema50"],
            ta_15m["cci"], ta_15m["mfi"], ta_15m["williams_r"], ta_15m["ichimoku_conv"],
            ta_15m["ichimoku_base"], ta_15m["parabolic_sar"]
        ]
        seq_1h = [
            ta_1h["rsi"], ta_1h["macd"], ta_1h["macd_hist"],
            ta_1h["bb_up"], ta_1h["bb_mid"], ta_1h["bb_low"],
            ta_1h["stoch_k"], ta_1h["stoch_d"], ta_1h["adx"], ta_1h["obv"],
            ta_1h["atr"], ta_1h["sma20"], ta_1h["sma50"], ta_1h["ema20"], ta_1h["ema50"],
            ta_1h["cci"], ta_1h["mfi"], ta_1h["williams_r"], ta_1h["ichimoku_conv"],
            ta_1h["ichimoku_base"], ta_1h["parabolic_sar"]
        ]
        return seq_5m + seq_15m + seq_1h  # length=63

    def build_signal_context_11(self, current_price, google_trend, reddit_sent, ob_ratio,
                                signal_0, signal_1, signal_2, news_signal, santiment,
                                btc_equity_pct, eur_equity_pct):
        """
        The 11 context features we decided:
          1) current_price
          2) google_trend
          3) reddit_sent
          4) ob_ratio
          5) signal_0
          6) signal_1
          7) 
          8) news_signal
          9) 
          10) btc_equity (balance in %)
          11) eur_equity (balance in %)
        """
        return [
            current_price,
            google_trend,
            reddit_sent,
            ob_ratio,
            signal_0,
            signal_1,
            signal_2,
            news_signal,
            santiment,
            btc_equity_pct,
            eur_equity_pct
        ]

    ###########################################################################
    # Balances & PnL
    ###########################################################################
    def get_account_balances(self):
        if PAPER_TRADING:
            return self.simulated_balances
        else:
            b = {"BTC": 0.0, "EUR": 0.0}
            try:
                info = self.binance_client.get_account()
                for asset in info.get("balances", []):
                    if asset["asset"] == "BTC":
                        b["BTC"] = float(asset["free"])
                    if asset["asset"] == "EUR":
                        b["EUR"] = float(asset["free"])
            except Exception as e:
                self.logger.error(f"Live balances error: {e}")
            return b

    def get_total_equity(self, price, balances):
        return balances.get("EUR", 0) + balances.get("BTC", 0) * price

    def compute_reward(self, current_equity):
        if self.last_equity is None:
            self.last_equity = current_equity
            return 0.0
        reward = current_equity - self.last_equity
        self.last_equity = current_equity
        return reward

    ###########################################################################
    # Paper Trading with 0.1% Fee
    ###########################################################################
    def open_position_paper(self, pos_type, price, signal_val, atr_percent):
        fee_rate = 0.001  # 0.1%
        if pos_type == "LONG":
            eur_bal = self.simulated_balances["EUR"]
            if eur_bal < self.min_trade_eur_trade:
                self.logger.info("Not enough EUR for LONG.")
                return

            fraction_of_balance = min(1.0, signal_val / 100) * self.max_trade_fraction
            eur_to_spend = eur_bal * fraction_of_balance
            if eur_to_spend < self.min_trade_eur_trade:
                self.logger.info("LONG trade size too low.")
                return

            # fee in EUR
            fee_eur = eur_to_spend * fee_rate
            total_eur_spent = eur_to_spend + fee_eur
            if total_eur_spent > eur_bal:
                self.logger.info("Not enough EUR for LONG + fee.")
                return

            size_btc = eur_to_spend / price
            self.simulated_balances["EUR"] -= total_eur_spent
            self.simulated_balances["BTC"] += size_btc

            trailing_stop = price * (1 - max(0.02, (atr_percent or 0.01) * 2))
            self.current_position = {
                "type": "LONG",
                "entry_price": price,
                "max_price": price,
                "trailing_stop": trailing_stop,
                "size_btc": size_btc,
                "eur_spent": total_eur_spent,
                "signal": signal_val
            }
            self.log_trade("BUY", price, total_eur_spent, size_btc, signal_val, "Paper LONG open")

        else:  # SHORT
            btc_bal = self.simulated_balances["BTC"]
            btc_val_eur = btc_bal * price
            if btc_val_eur < self.min_trade_eur_trade:
                self.logger.info("Not enough BTC for SHORT.")
                return
            fraction_of_balance = min(1.0, abs(signal_val) / 100) * self.max_trade_fraction
            eur_to_sell = fraction_of_balance * btc_val_eur
            if eur_to_sell < self.min_trade_eur_trade:
                self.logger.info("SHORT trade size too low.")
                return

            fee_eur = eur_to_sell * fee_rate
            total_eur_received = eur_to_sell - fee_eur
            if total_eur_received < 0:
                return

            btc_to_sell = eur_to_sell / price
            if btc_to_sell > btc_bal:
                self.logger.info("Not enough BTC to short that fraction.")
                return

            self.simulated_balances["BTC"] -= btc_to_sell
            self.simulated_balances["EUR"] += total_eur_received

            trailing_stop = price * (1 + max(0.02, (atr_percent or 0.01) * 2))
            self.current_position = {
                "type": "SHORT",
                "entry_price": price,
                "min_price": price,
                "trailing_stop": trailing_stop,
                "size_btc": btc_to_sell,
                "eur_received": total_eur_received,
                "signal": signal_val
            }
            self.log_trade("SELL", price, eur_to_sell, btc_to_sell, signal_val, "Paper SHORT open")

    def close_position_paper(self, price):
        fee_rate = 0.001  # 0.1%
        if not self.current_position:
            return
        pos = self.current_position
        if pos["type"] == "LONG":
            btc_to_sell = pos["size_btc"]
            gross_eur = btc_to_sell * price
            fee_eur = gross_eur * fee_rate
            net_eur = gross_eur - fee_eur
            self.simulated_balances["BTC"] -= btc_to_sell
            self.simulated_balances["EUR"] += net_eur
            self.log_trade("SELL", price, gross_eur, btc_to_sell, pos["signal"], "Paper LONG close")
        else:
            btc_to_buy = pos["size_btc"]
            gross_eur_spent = btc_to_buy * price
            fee_eur = gross_eur_spent * fee_rate
            total_eur_spent = gross_eur_spent + fee_eur
            self.simulated_balances["BTC"] += btc_to_buy
            self.simulated_balances["EUR"] -= total_eur_spent
            self.log_trade("BUY", price, total_eur_spent, btc_to_buy, pos["signal"], "Paper SHORT close")

        self.current_position = None

    def manage_open_position_paper(self, current_price):
        if not self.current_position:
            return
        pos = self.current_position
        if pos["type"] == "LONG":
            if current_price > pos["max_price"]:
                pos["max_price"] = current_price
                trailing_pct = max(0.02, (current_price - pos["entry_price"]) / pos["entry_price"])
                pos["trailing_stop"] = current_price * (1 - trailing_pct)
            if current_price < pos["trailing_stop"]:
                self.logger.info(f"LONG stop triggered at {current_price}")
                self.close_position_paper(current_price)
        else:
            if current_price < pos.get("min_price", 999999):
                pos["min_price"] = current_price
                trailing_pct = max(0.02, (pos["entry_price"] - current_price) / pos["entry_price"])
                pos["trailing_stop"] = current_price * (1 + trailing_pct)
            if current_price > pos["trailing_stop"]:
                self.logger.info(f"SHORT stop triggered at {current_price}")
                self.close_position_paper(current_price)

    ###########################################################################
    # Live Trading (0.1% fee is generally handled by Binance on fill)
    ###########################################################################
    def execute_live_order(self, side, quantity):
        try:
            order = self.binance_client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            self.logger.info(f"[Live] {side} => {order}")
        except Exception as e:
            self.logger.error(f"[Live] create_order error: {e}")

    ###########################################################################
    # Action & Logging
    ###########################################################################
    def process_trade_signal(self, action, signal_val, current_price, atr_percent):
        self.logger.info(f"[TradeSignal] action={action}, sig={signal_val:.2f}")
        if PAPER_TRADING:
            if self.current_position:
                # Reversal
                if self.current_position["type"] == "LONG" and action == "SHORT":
                    self.logger.info("Reversal => close LONG => open SHORT")
                    self.close_position_paper(current_price)
                    self.open_position_paper("SHORT", current_price, signal_val, atr_percent)
                elif self.current_position["type"] == "SHORT" and action == "LONG":
                    self.logger.info("Reversal => close SHORT => open LONG")
                    self.close_position_paper(current_price)
                    self.open_position_paper("LONG", current_price, signal_val, atr_percent)
                # else no new action
            else:
                if action == "LONG":
                    self.open_position_paper("LONG", current_price, signal_val, atr_percent)
                elif action == "SHORT":
                    self.open_position_paper("SHORT", current_price, signal_val, atr_percent)
                else:
                    self.logger.info("HOLD => no new position (paper).")
        else:
            # Live trading
            balances = self.get_account_balances()
            eur_bal = balances.get("EUR", 0)
            btc_bal = balances.get("BTC", 0)
            if action == "LONG":
                frac = min(1.0, signal_val / 100) * self.max_trade_fraction
                eur_spend = eur_bal * frac
                if eur_spend < self.min_trade_eur_trade:
                    self.logger.info("[Live] Not enough EUR to buy.")
                    return
                px = self.fetch_price_at_hour(self.symbol)
                if px:
                    size_btc = eur_spend / px
                    self.logger.info(f"[Live] Market BUY {size_btc:.6f} BTC")
                    self.execute_live_order(SIDE_BUY, round(size_btc, 6))
            elif action == "SHORT":
                frac = min(1.0, abs(signal_val) / 100) * self.max_trade_fraction
                btc_to_sell = btc_bal * frac
                if btc_to_sell * current_price < self.min_trade_eur_trade:
                    self.logger.info("[Live] Not enough BTC to sell.")
                    return
                self.logger.info(f"[Live] Market SELL {btc_to_sell:.6f} BTC")
                self.execute_live_order(SIDE_SELL, round(btc_to_sell, 6))
            else:
                self.logger.info("[Live] HOLD => no trade.")

    def log_trade(self, action, price, size_eur, size_btc, signal, notes=""):
        ts = datetime.datetime.utcnow().isoformat()
        if PAPER_TRADING:
            btc_bal = self.simulated_balances["BTC"]
            eur_bal = self.simulated_balances["EUR"]
        else:
            b = self.get_account_balances()
            btc_bal = b["BTC"]
            eur_bal = b["EUR"]

        try:
            with open(self.trade_log_file, mode='a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    ts, action, price, size_eur, size_btc, signal, notes, btc_bal, eur_bal
                ])
        except Exception as e:
            self.logger.error(f"Trade log error: {e}")

    ###########################################################################
    # Build RL State Vector (for the new DQNAgent, if used)
    ###########################################################################
    def build_rl_state(self, final_signal, atr_percent, balances, price):
        """
        Example RL state of shape (4,):
          [final_signal/100, atr_percent, BTC fraction, EUR fraction]
        """
        fs_norm = np.clip(final_signal/100.0, -1.0, 1.0)
        ap = 0.0 if (atr_percent is None) else atr_percent
        eq_btc_val = balances["BTC"] * price
        eq_eur_val = balances["EUR"]
        tot_eq = eq_btc_val + eq_eur_val
        if tot_eq <= 0:
            return np.array([fs_norm, ap, 0.0, 0.0], dtype=np.float32)
        btc_frac = eq_btc_val / tot_eq
        eur_frac = eq_eur_val / tot_eq
        return np.array([fs_norm, ap, btc_frac, eur_frac], dtype=np.float32)

    ###########################################################################
    # Main Cycle
    ###########################################################################
    def run_cycle(self):
        """
        Gathers data, calls aggregator => final_signal, picks an RL action,
        trades, logs result. Integrates DQN with transitions. 
        """
        self.logger.info("=== Starting Trading Cycle ===")
        try:
            # 1) Current market price
            current_price = self.fetch_price_at_hour(self.symbol)
            if current_price is None:
                self.logger.error("No current price => abort.")
                return

            # manage trailing stops if paper
            if PAPER_TRADING:
                self.manage_open_position_paper(current_price)

            # 2) fetch 60-bar klines for 5m,15m,1h + other data
            klines_5m = self.fetch_klines(interval=Client.KLINE_INTERVAL_5MINUTE, limit=self.model_5m_window)
            klines_15m= self.fetch_klines(interval=Client.KLINE_INTERVAL_15MINUTE, limit=self.model_15m_window)
            klines_1h = self.fetch_klines(interval=Client.KLINE_INTERVAL_1HOUR, limit=self.model_1h_window)

            news_data_res = self.fetch_news_data()
            ob_data = self.fetch_order_book(self.symbol, limit=20)
            google_trend = self.fetch_google_trends()
            santiment_data= self.fetch_santiment_data()
            reddit_sent   = self.fetch_reddit_sentiment()
            balances      = self.get_account_balances()
            total_equity  = self.get_total_equity(current_price, balances)
            reward        = self.compute_reward(total_equity)

            # 3) aggregator => final_signal
            (
                final_signal,
                local_signals,
                news_signal,
                arr_5m,
                arr_15m_arr,
                arr_1h,
                arr_google_trend,
                arr_santiment,
                arr_ta_63,
                arr_ctx_11,
                aggregator_prompt
            ) = self.get_aggregated_signal(
                current_utc_time=datetime.datetime.now(datetime.timezone.utc),
                current_price=current_price,
                klines_5m=klines_5m,
                klines_15m=klines_15m,
                klines_1h=klines_1h,
                news_articles=news_data_res,
                ob_data=ob_data,
                google_trend=google_trend,
                balances=balances,
                santiment_data=santiment_data,
                reddit_sent=reddit_sent,
                use_real_gpt = True,
                use_model_pred = True
            )

            # 4) RL-based decision
            atr_val = 0.0
            if klines_15m and len(klines_15m) == 60:
                atr_v = compute_atr_from_klines(klines_15m, 14)
                atr_val = atr_v/current_price if atr_v else 0.0

            state_vec = self.build_rl_state(final_signal, atr_val, balances, current_price)
            action = self.rl_agent.select_action(state_vec)  # {LONG, SHORT, HOLD}

            # 5) Execute
            self.process_trade_signal(action, final_signal, current_price, atr_val)
            new_balances = self.get_account_balances()
            new_equity   = self.get_total_equity(current_price, new_balances)

            # 6) Store RL transition + train
            next_state_vec = self.build_rl_state(final_signal, atr_val, new_balances, current_price)
            done = False  # for now, always false in live/paper
            self.rl_agent.store_transition(state_vec, action, reward, next_state_vec, done)
            self.rl_agent.train_step()
            if threading.current_thread() is threading.main_thread():
                self.rl_agent.save()
            
            self.save_gpt_cache()

            # 7) Log everything
            self.last_log_data = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "current_price": float(current_price),
                "arr_5m": arr_5m.tolist(),
                "arr_15m": arr_15m_arr.tolist(),
                "arr_1h": arr_1h.tolist(),
                "arr_google_trend": arr_google_trend.tolist(),
                "arr_santiment": arr_santiment.tolist(),
                "arr_ta_63": arr_ta_63.tolist(),
                "arr_ctx_11": arr_ctx_11.tolist(),
                "reddit_sent": reddit_sent,
                "santiment": santiment_data,
                "local_gpt_signals": local_signals,
                "news_signal": news_signal,
                "final_signal": float(final_signal),
                "action": action,
                "balances": new_balances,
                "total_equity": float(new_equity),
                "atr_percent": atr_val,
                "aggregator_prompt": aggregator_prompt,
                "reward": reward
            }

        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
            self.logger.error(traceback.format_exc())
            
        self.logger.info("=== Trading Cycle Complete ===\n")

