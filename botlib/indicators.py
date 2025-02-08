#!/usr/bin/env python3
"""
Technical indicator functions.
"""

import statistics
import numpy as np


def compute_rsi(prices, period=14):
    """
    Computes the Relative Strength Index (RSI).
    Returns an RSI value in [0,100], or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        return 100.0
    rs = up / down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 2)


def compute_ema(data, period):
    """
    Exponential Moving Average (EMA).
    Returns a list of EMA values, one for each data point after the initial seed.
    If data length < period, returns an empty list.
    """
    if len(data) < period:
        return []
    alpha = 2 / (period + 1)
    ema_vals = [sum(data[:period]) / period]  # initial seed = SMA of first 'period' points
    for price in data[period:]:
        ema_vals.append(alpha * price + (1 - alpha) * ema_vals[-1])
    return ema_vals


def compute_sma(data, period):
    """
    Simple Moving Average (SMA).
    Returns a list of SMA values, one for each data point after the initial seed.
    If data length < period, returns None or empty.
    For convenience, we return just the last SMA value if we want a single float.
    """
    if len(data) < period:
        return None
    result = []
    for i in range(period, len(data) + 1):
        window = data[i - period : i]
        avg = sum(window) / period
        result.append(avg)
    return result[-1] if result else None


def compute_macd(prices, short=12, long=26, signal=9):
    """
    MACD: difference between short EMA and long EMA. 
    Returns (macd_line, macd_signal, macd_hist).
    If insufficient data, returns (None, None, None).
    """
    if len(prices) < long:
        return None, None, None
    ema_short = compute_ema(prices, short)
    ema_long = compute_ema(prices, long)
    n = min(len(ema_short), len(ema_long))
    macd_series = [s - l for s, l in zip(ema_short[-n:], ema_long[-n:])]
    if len(macd_series) < signal:
        return None, None, None
    ema_sig = compute_ema(macd_series, signal)
    return macd_series[-1], ema_sig[-1], macd_series[-1] - ema_sig[-1]


def compute_bollinger_bands(prices, period=20, num_std=2):
    """
    Bollinger Bands: (middle_band, upper_band, lower_band).
    middle_band = SMA(period), upper/lower = +/- num_std * stdev from middle.
    Returns (sma, upper, lower) or None if insufficient data.
    """
    if len(prices) < period:
        return None
    recent = prices[-period:]
    sma = sum(recent) / period
    std_dev = statistics.pstdev(recent)
    return (sma, sma + num_std * std_dev, sma - num_std * std_dev)


def compute_atr_from_klines(klines, period=14):
    """
    Average True Range (ATR) based on kline data.
    period = typical 14.
    Returns a float ATR or None if insufficient data.
    """
    if len(klines) < period + 1:
        return None
    highs = [float(k[1]) for k in klines]
    lows = [float(k[2]) for k in klines]
    closes = [float(k[3]) for k in klines]
    trs = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def compute_stoch_rsi(prices, period=14, k=3, d=3):
    """
    Stochastic RSI. Returns (k_val, d_val) or (None, None) if insufficient data.
    """
    if len(prices) < period:
        return None, None
    # standard RSI first
    rsi_vals = []
    for i in range(period, len(prices)):
        sub = prices[i - period : i + 1]
        rsi_ = compute_rsi(sub, period)
        if rsi_ is not None:
            rsi_vals.append(rsi_)
    if not rsi_vals:
        return None, None

    min_rsi = min(rsi_vals)
    max_rsi = max(rsi_vals)
    if max_rsi - min_rsi == 0:
        return 50.0, 50.0
    stoch = (rsi_vals[-1] - min_rsi) / (max_rsi - min_rsi) * 100
    if len(rsi_vals) < 3:
        return (round(stoch, 2), None)

    # k, d
    stoch_vals = []
    for i in range(2, len(rsi_vals)):
        window = rsi_vals[i - 2 : i + 1]
        w_min = min(window)
        w_max = max(window)
        if w_max - w_min == 0:
            stoch_vals.append(50.0)
        else:
            stoch_vals.append((rsi_vals[i] - w_min) / (w_max - w_min) * 100)
    k_val = stoch_vals[-1] if stoch_vals else stoch
    d_window = stoch_vals[-d:] if len(stoch_vals) >= d else stoch_vals
    d_val = sum(d_window) / len(d_window) if d_window else k_val
    return (round(k_val, 2), round(d_val, 2))


def compute_adx(klines, period=14):
    """
    Average Directional Index (ADX).
    Returns a float or None if insufficient data.
    """
    if len(klines) < period + 1:
        return None
    highs = [float(k[1]) for k in klines]
    lows = [float(k[2]) for k in klines]
    closes = [float(k[3]) for k in klines]

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    for i in range(1, len(klines)):
        curr_high = highs[i]
        curr_low = lows[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]
        tr = max(curr_high - curr_low, abs(curr_high - prev_close), abs(curr_low - prev_close))
        plus_dm = curr_high - prev_high if (curr_high - prev_high) > (prev_low - curr_low) else 0
        minus_dm = prev_low - curr_low if (prev_low - curr_low) > (curr_high - prev_high) else 0
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    def wilder_smooth(data, n=14):
        if len(data) < n:
            return []
        smoothed = [sum(data[:n])]
        for i in range(n, len(data)):
            prev = smoothed[-1]
            smoothed.append(prev - (prev / n) + data[i])
        return smoothed

    tr_sm = wilder_smooth(tr_list, period)
    plus_sm = wilder_smooth(plus_dm_list, period)
    minus_sm = wilder_smooth(minus_dm_list, period)

    if not (tr_sm and plus_sm and minus_sm):
        return None
    min_len = min(len(tr_sm), len(plus_sm), len(minus_sm))
    tr_sm = tr_sm[-min_len:]
    plus_sm = plus_sm[-min_len:]
    minus_sm = minus_sm[-min_len:]

    plus_di = [(plus_sm[i] / tr_sm[i]) * 100 if tr_sm[i] != 0 else 0 for i in range(min_len)]
    minus_di = [(minus_sm[i] / tr_sm[i]) * 100 if tr_sm[i] != 0 else 0 for i in range(min_len)]
    dx = []
    for i in range(min_len):
        top = abs(plus_di[i] - minus_di[i])
        bot = plus_di[i] + minus_di[i] if (plus_di[i] + minus_di[i]) != 0 else 1
        dx.append((top / bot) * 100)

    if len(dx) < period:
        return None
    adx = sum(dx[-period:]) / period
    return round(adx, 2)


def compute_obv(closes, volumes):
    """
    On-Balance Volume (OBV).
    Returns a float or None if input lengths differ or empty.
    """
    if not closes or not volumes or len(closes) != len(volumes):
        return None
    obv_val = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv_val += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_val -= volumes[i]
    return obv_val


def compute_cci(highs, lows, closes, period=20):
    """
    Commodity Channel Index (CCI).
    Returns the last CCI value or None if insufficient data.
    Typical formula:
      CCI = (TP - SMA(TP, period)) / (0.015 * MeanDev)
      where TP = (high + low + close)/3
    """
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None

    typical_prices = []
    for h, l, c in zip(highs, lows, closes):
        typical_prices.append((h + l + c) / 3.0)

    # We need SMA of typical_prices, then mean deviation
    cci_vals = []
    for i in range(period, len(typical_prices) + 1):
        window = typical_prices[i - period : i]
        sma_tp = sum(window) / period
        mean_dev = np.mean([abs(x - sma_tp) for x in window])
        if mean_dev == 0:
            cci_vals.append(0.0)
        else:
            cci_ = (window[-1] - sma_tp) / (0.015 * mean_dev)
            cci_vals.append(cci_)
    if not cci_vals:
        return None
    return float(cci_vals[-1])


def compute_mfi(highs, lows, closes, volumes, period=14):
    """
    Money Flow Index (MFI).
    Returns the last MFI value or None if insufficient data.
    MFI = 100 - [100 / (1 + (PositiveMF / NegativeMF))]
    """
    if len(highs) < period or len(lows) < period or len(closes) < period or len(volumes) < period:
        return None

    typical_prices = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    raw_money_flows = []
    for i in range(len(typical_prices)):
        raw_money_flows.append(typical_prices[i] * volumes[i])

    pos_flow = []
    neg_flow = []
    for i in range(1, len(typical_prices)):
        if typical_prices[i] > typical_prices[i - 1]:
            pos_flow.append(raw_money_flows[i])
            neg_flow.append(0)
        else:
            neg_flow.append(raw_money_flows[i])
            pos_flow.append(0)

    # We'll compute MFI for last 'period' steps
    if len(pos_flow) < period:
        return None
    pos_mf = sum(pos_flow[-period:])
    neg_mf = sum(neg_flow[-period:])
    if neg_mf == 0:
        return 100.0
    mr = pos_mf / neg_mf
    mfi = 100.0 - (100.0 / (1.0 + mr))
    return round(mfi, 2)


def compute_williams_r(highs, lows, closes, period=14):
    """
    Williams %R:
      %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100
    Typically in range [-100, 0].
    """
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return None

    williams_vals = []
    for i in range(period, len(closes) + 1):
        window_h = highs[i - period : i]
        window_l = lows[i - period : i]
        c = closes[i - 1]
        hh = max(window_h)
        ll = min(window_l)
        if hh == ll:
            wpr = 0
        else:
            wpr = (hh - c) / (hh - ll) * -100
        williams_vals.append(wpr)

    if not williams_vals:
        return None
    return round(williams_vals[-1], 2)


def compute_ichimoku(highs, lows, closes):
    """
    Ichimoku Conversion and Base lines (simplified).
    Many full Ichimoku implementations also compute Leading Span A/B, Chikou Span, etc.
    We just do:
      Conversion = (9-period high + 9-period low)/2
      Base = (26-period high + 26-period low)/2
    Returns (conversion_line, base_line) or (None, None) if not enough data.
    """
    if len(highs) < 26 or len(lows) < 26:
        return None, None

    # last 9 bars for conversion
    recent_9h = highs[-9:]
    recent_9l = lows[-9:]
    conv_line = (max(recent_9h) + min(recent_9l)) / 2.0

    # last 26 bars for base
    recent_26h = highs[-26:]
    recent_26l = lows[-26:]
    base_line = (max(recent_26h) + min(recent_26l)) / 2.0

    return (conv_line, base_line)


def compute_parabolic_sar(highs, lows, closes, af_step=0.02, af_max=0.2):
    """
    Parabolic SAR (simplified):
    We'll do a single pass approach: if we assume an uptrend start, etc.
    In real usage, we might track state (trend up or down).
    Returns the last parabolic SAR value or None if insufficient data.
    """
    if len(highs) < 2 or len(lows) < 2:
        return None

    # We do a naive approach:
    # 1) Start with an uptrend if close[1] > close[0]
    # 2) Keep track of extreme points, acceleration factor
    psar = [None] * len(closes)
    trend_up = closes[1] > closes[0]
    af = af_step
    ep = highs[0] if trend_up else lows[0]
    psar[0] = lows[0] - (highs[0] - lows[0])  # or just None

    for i in range(1, len(closes)):
        prior_sar = psar[i - 1] if psar[i - 1] is not None else (lows[i - 1] if trend_up else highs[i - 1])
        if trend_up:
            # current SAR
            curr_sar = prior_sar + af * (ep - prior_sar)
            # safety check
            if curr_sar > lows[i]:
                # switch to downtrend
                trend_up = False
                curr_sar = ep
                af = af_step
                ep = lows[i]
            else:
                # continue uptrend
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            # downtrend
            curr_sar = prior_sar - af * (ep - prior_sar)
            if curr_sar < highs[i]:
                trend_up = True
                curr_sar = ep
                af = af_step
                ep = highs[i]
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)

        psar[i] = curr_sar

    return psar[-1] if psar[-1] is not None else None
