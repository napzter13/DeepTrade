#!/usr/bin/env python3
"""
Data fetchers and caching logic for:
- Binance price + klines + orderbook
- News API
- Google Trends (pytrends)
- Santiment
- Reddit sentiment

Now:
 - No 'live' key is used. If request has no date, we floor the current time to hour and store as that key.
 - For fetch_order_book and fetch_reddit_sentiment (no true historical data):
    * If dt is given but not in cache, we find the nearest cached hour
      or if none, we do a new fetch as of now() floored hour, store it, and return.
"""

import os
import json
import datetime
import time
import requests
import praw
import hashlib
import portalocker
import numpy as np

from binance.client import Client
from binance.enums import *
from pytrends.request import TrendReq
from nltk.sentiment import SentimentIntensityAnalyzer

from .environment import (
    ORDERBOOK_CACHE_FILE,
    REDDIT_SENTIMENT_CACHE_FILE,
    GOOGLE_TRENDS_CACHE_FILE,
    NEWS_ANALYZE_CACHE_FILE,
    SANTIMENT_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    NEWS_API_KEY,
    get_logger
)

logger = get_logger("DataFetchers")

pytrends = TrendReq(hl='en-US', tz=360)

sentiment_analyzer = SentimentIntensityAnalyzer()

###############################################################################
# Utility method for sentiment analysis
###############################################################################
def compute_average_sentiment(texts):
    # sent = {
    #     'neg': 0,
    #     'neu': 0,
    #     'pos': 0,
    #     'compound': 0,
    #     'compound_rescaled': 0,
    #     'diff_pos_neg': 0,
    #     'sum_pos_neg': 0,
    #     'is_strongly_positive': 0,
    #     'is_strongly_negative': 0,
    #     'ratio_pos_neg': 0
    # }
    
    compunds = []
    
    for text in texts:
        scores = sentiment_analyzer.polarity_scores(text)
        neg = scores['neg']
        neu = scores['neu']
        pos = scores['pos']
        compound = scores['compound']
        
        compunds.append(compound)
        
        # compound_rescaled = (compound + 1) / 2
        # diff_pos_neg = pos - neg
        # sum_pos_neg = pos + neg
        # is_strongly_positive = 1 if compound >= 0.5 else 0
        # is_strongly_negative = 1 if compound <= -0.5 else 0
        # ratio_pos_neg = pos / (neg + 1e-6)
        
        # sent['neg'] += neg
        # sent['neu'] += neu
        # sent['pos'] += pos
        # sent['compound'] += compound
        # sent['compound_rescaled'] += compound_rescaled
        # sent['diff_pos_neg'] += diff_pos_neg
        # sent['sum_pos_neg'] += sum_pos_neg
        # sent['is_strongly_positive'] += is_strongly_positive
        # sent['is_strongly_negative'] += is_strongly_negative
        # sent['ratio_pos_neg'] += ratio_pos_neg
        
    return (sum(compunds)/len(compunds))*100

# Utility: load/save JSON
def load_json_cache(filepath):
    """
    Loads the JSON cache file in a thread-safe way:
      - Acquires a shared lock (LOCK_SH) so multiple readers can read concurrently.
      - Returns {} if the file doesn't exist or can't be parsed.
    """
    lock_path = filepath + ".lock"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(lock_path, "a") as lock_file:
        portalocker.lock(lock_file, portalocker.LOCK_SH)  # Shared lock for reading
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse {filepath}: {e}")
                return {}
        else:
            return {}

def save_json_cache(filepath, new_data):
    """
    Saves new_data into the JSON cache file in a thread-safe, atomic manner:
      1. Acquires an exclusive lock (LOCK_EX) so only one writer at a time.
      2. Reads the existing file from disk again (while locked) to avoid losing updates from other threads.
      3. Inserts (merges) new_data into the old data—keys from new_data overwrite or add to what's there.
         * If we only pass a single key in new_data (like {"myKey": ...}), this is a minimal overhead insertion.
      4. Writes out the merged JSON to a temp file and uses os.replace to ensure an atomic write.
    """
    lock_path = filepath + ".lock"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(lock_path, "a") as lock_file:
        portalocker.lock(lock_file, portalocker.LOCK_EX)  # Exclusive lock for writing

        # 1) Read current on-disk data again while we hold the lock.
        old_data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse {filepath} before saving: {e}")

        # 2) Insert new_data into old_data (avoiding large merges if new_data is only 1 key).
        #    If new_data = {"newKey": ...}, then only that one key is updated.
        old_data.update(new_data)

        # 3) Write to a temp file, then atomically replace the original.
        tmp_filepath = filepath + ".tmp"
        try:
            with open(tmp_filepath, "w", encoding="utf-8") as f:
                json.dump(old_data, f, indent=2)
            os.replace(tmp_filepath, filepath)
        except Exception as e:
            logger.error(f"Could not write cache to {filepath}: {e}")
    # Lock is released on exit


###############################################################################
# Utility: For fetch_order_book & fetch_reddit_sentiment "nearest hour" approach
###############################################################################
def find_nearest_hour_key(cache_dict, dt_floor):
    """Given a dictionary of hour-based keys (YYYY-MM-DD HH:00:00),
       find the stored key with minimal absolute difference from dt_floor."""
    if not cache_dict:
        return None
    all_keys = list(cache_dict.keys())
    dt_list = []
    for k in all_keys:
        try:
            dtp = datetime.datetime.strptime(k, "%Y-%m-%d %H:%M:%S")
            dt_list.append(dtp)
        except:
            pass
    if not dt_list:
        return None

    def abs_diff(a, b):
        return abs((a - b).total_seconds())

    best_match = min(dt_list, key=lambda x: abs_diff(x, dt_floor))
    return best_match.strftime("%Y-%m-%d %H:%M:%S")

###############################################################################
# get_klines
###############################################################################
INTERVAL_TO_MINUTES = {
    Client.KLINE_INTERVAL_1MINUTE:   1,
    Client.KLINE_INTERVAL_3MINUTE:   3,
    Client.KLINE_INTERVAL_5MINUTE:   5,
    Client.KLINE_INTERVAL_15MINUTE:  15,
    Client.KLINE_INTERVAL_30MINUTE:  30,
    Client.KLINE_INTERVAL_1HOUR:     60,
    Client.KLINE_INTERVAL_2HOUR:     120,
    Client.KLINE_INTERVAL_4HOUR:     240,
    Client.KLINE_INTERVAL_6HOUR:     360,
    Client.KLINE_INTERVAL_8HOUR:     480,
    Client.KLINE_INTERVAL_12HOUR:    720,
    Client.KLINE_INTERVAL_1DAY:      1440,
    Client.KLINE_INTERVAL_3DAY:      4320,
    Client.KLINE_INTERVAL_1WEEK:     10080,
    Client.KLINE_INTERVAL_1MONTH:    43200,
}

def get_klines(
    binance_client: Client,
    symbol="BTCEUR",
    interval=Client.KLINE_INTERVAL_5MINUTE,
    limit=60,
    end_dt: datetime.datetime = None
):
    """
    Fetch `limit` bars of klines for `symbol` at `interval`.
    If end_dt is None => use current time floored to hour as key, fetch if missing.
    Otherwise => also floor end_dt => key.

    Then we do the logic: if not found => fetch from binance, store, return.
    Cache file => input_cache/klines_cache.json
    """
    cache_file = "input_cache/klines_cache.json"
    cache_dict = load_json_cache(cache_file)

    if end_dt is None:
        end_dt = datetime.datetime.utcnow()
    dt_floor = end_dt.replace(minute=0, second=0, microsecond=0)

    key = f"{symbol}-{interval}-{limit}-{dt_floor.strftime('%Y-%m-%d %H:%M:%S')}"

    if key in cache_dict:
        logger.info(f"[get_klines] Using cache => {key}")
        return cache_dict[key]

    logger.info(f"[get_klines] Not in cache => fetch => {key}")

    # Now do the real fetch
    # Convert dt_floor to ms
    end_ms = int(dt_floor.timestamp() * 1000)
    if interval not in INTERVAL_TO_MINUTES:
        raise ValueError(f"Unsupported interval: {interval}")

    minutes_per_bar = INTERVAL_TO_MINUTES[interval]
    total_minutes = minutes_per_bar * limit
    start_dt = dt_floor - datetime.timedelta(minutes=total_minutes)
    start_ms = int(start_dt.timestamp() * 1000)

    data = binance_client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
        limit=limit
    )
    
    # Clean data
    cleaned_data = []

    # Define the indexes to keep:
    # 1: Open Price, 2: High Price, 3: Low Price, 4: Close Price,
    # 5: Volume, 7: Quote Asset Volume, 8: Number of Trades,
    # 9: Taker Buy Base Asset Volume, 10: Taker Buy Quote Asset Volume
    indexes_to_keep = [1, 2, 3, 4, 5, 7, 8, 9, 10]      # 9 features

    for kline in data:
        cleaned_kline = [float(kline[i]) for i in indexes_to_keep]
        cleaned_data.append(cleaned_kline)

    save_json_cache(cache_file, {key: cleaned_data})
    return cleaned_data


###############################################################################
# fetch_price_at_hour
###############################################################################
def fetch_price_at_hour(binance_client: Client, symbol="BTCEUR", dt: datetime.datetime = None):
    """
    Return a BTC/EUR price. If dt is None => use now floored to hour as key.
    If dt < 1 hour ago => fetch 1-hour kline from dt..(dt+1h). If none => None.
    Store in input_cache/price_at_hour_cache.json
    """
    cache_file = "input_cache/price_at_hour_cache.json"
    cache_dict = load_json_cache(cache_file)

    now_utc = datetime.datetime.utcnow()
    if dt is None:
        dt = now_utc
    dt_floor = dt.replace(minute=0, second=0, microsecond=0)

    key = f"{symbol}-{dt_floor.strftime('%Y-%m-%d %H:%M:%S')}"
    if key in cache_dict:
        logger.info(f"[fetch_price_at_hour] Using cache => {key}")
        return cache_dict[key]

    logger.info(f"[fetch_price_at_hour] Not in cache => fetch => {key}")

    # Logic from before
    time_diff_sec = (now_utc - dt).total_seconds()
    if time_diff_sec < 3600:
        # fetch live
        try:
            ticker = binance_client.get_symbol_ticker(symbol=symbol)
            val = float(ticker["price"])
        except Exception as e:
            logger.error(f"Error fetching live ticker: {e}")
            return np.nan
    else:
        # dt more than 1h ago => fetch 1h kline
        try:
            start_ts = int(dt.timestamp() * 1000)
            end_ts   = start_ts + 3600_000
            klines = binance_client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1HOUR,
                startTime=start_ts,
                endTime=end_ts,
                limit=1
            )
            if klines:
                val = float(klines[0][4])
            else:
                return np.nan
        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}")
            return np.nan

    save_json_cache(cache_file, {key: val})
    return val


###############################################################################
# fetch_order_book
###############################################################################
def fetch_order_book(binance_client: Client,
                     symbol="BTCEUR",
                     limit=20,
                     dt: datetime.datetime = None):
    """
    We cannot fetch real historical order books from binance. If dt is None => fetch live from now floored hour.
    If dt is provided => we search for that floored hour in the cache, if not found => we do nearest hour,
    if that fails => fetch new as of now, store under that now floored hour.
    """
    cache_file = ORDERBOOK_CACHE_FILE
    cache_data = load_json_cache(cache_file)

    if dt is None:
        dt_now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        key = f"{symbol}-{dt_now.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        dt_floor = dt.replace(minute=0, second=0, microsecond=0)
        key = dt_floor.strftime("%Y-%m-%d %H:%M:%S")
        key = f"{symbol}-{key}"

    if key in cache_data:
        logger.info(f"[fetch_order_book] Using cached => {key}")
        return cache_data[key]

    # Not found => if dt is not None, find nearest
    if dt is not None:
        logger.info(f"[fetch_order_book] Not found => search nearest hour => {key}")
        # key is e.g. 'BTCEUR-2023-09-15 15:00:00'
        # We want to do: find nearest. Let's parse out the actual dt from the key
        splitted = key.split("-",1)
        # splitted[0] => symbol? splitted[1] => '2023-09-15 15:00:00'
        if len(splitted)==2:
            dt_part = splitted[1]
        else:
            dt_part = "2023-01-01 00:00:00"
        try:
            dt_req = datetime.datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
        except:
            dt_req = datetime.datetime.min

        # find nearest
        nearest_key = find_nearest_hour_key(cache_data, dt_req)
        if nearest_key:
            real_key = f"{symbol}-{nearest_key}"
            logger.info(f"[fetch_order_book] Found nearest => {real_key}")
            return cache_data[real_key]
        else:
            logger.info("[fetch_order_book] No parseable entries => fallback fetch now")

    # If we get here => we fetch "live" as of now
    dt_now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    newkey = f"{symbol}-{dt_now.strftime('%Y-%m-%d %H:%M:%S')}"
    logger.info(f"[fetch_order_book] fallback fetch => store under => {newkey}")

    try:
        ob = binance_client.get_order_book(symbol=symbol, limit=limit)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bid_p = [float(x[0]) for x in bids]
        ask_p = [float(x[0]) for x in asks]
        bid_v = [float(x[1]) for x in bids]
        ask_v = [float(x[1]) for x in asks]

        result = {}
        if bid_p and ask_p:
            avg_bid = sum(bid_p) / len(bid_p)
            avg_ask = sum(ask_p) / len(ask_p)
            spread = avg_ask - avg_bid
            total_bid_vol = sum(bid_v)
            total_ask_vol = sum(ask_v)
            ratio = total_bid_vol / total_ask_vol if total_ask_vol else 1
            result = {
                "avg_bid": avg_bid,
                "avg_ask": avg_ask,
                "spread": spread,
                "bid_ask_volume_ratio": ratio,
                "bids": bids,
                "asks": asks
            }
        
        save_json_cache(cache_file, {newkey: result})
        return result
    except Exception as e:
        logger.error(f"Order book error: {e}")
        return {}


###############################################################################
# fetch_news_data
###############################################################################
def fetch_news_data(days=1, end_dt=None):
    """
    If end_dt is None => we do now floored hour, if end_dt => floor => key
    from=(end_dt-days)..end_dt
    Cache => input_cache/news_data_cache.json
    """
    cache_file = "input_cache/news_data_cache.json"
    cache_dict = load_json_cache(cache_file)

    if end_dt is None:
        dt_now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        key = f"btc-news-{days}-{dt_now.strftime('%Y-%m-%d %H:%M:%S')}"
        dt_end = dt_now
    else:
        dt_floor = end_dt.replace(minute=0, second=0, microsecond=0)
        key = f"btc-news-{days}-{dt_floor.strftime('%Y-%m-%d %H:%M:%S')}"
        dt_end = dt_floor

    if key in cache_dict:
        logger.info(f"[fetch_news_data] Using cache => {key}")
        return cache_dict[key]
    
    if end_dt != None and end_dt < (datetime.datetime.now() - datetime.timedelta(days=30)):
        return []

    logger.info(f"[fetch_news_data] Not in cache => fetch => {key}")

    if not NEWS_API_KEY:
        logger.info("No NEWS_API_KEY => returning empty list.")
        articles = []
    else:
        start_dt = dt_end - datetime.timedelta(days=days)
        from_str = start_dt.strftime("%Y-%m-%d")
        to_str   = dt_end.strftime("%Y-%m-%d")
        url = (
            f"https://newsapi.org/v2/everything?q=Bitcoin"
            f"&from={from_str}&to={to_str}"
            f"&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            articles = []

    save_json_cache(cache_file, {key: articles})
    return articles


###############################################################################
# fetch_google_trends
###############################################################################
def fetch_google_trends(end_dt=None, max_retries=1, delay=5, days=7):
    """
    Fetch daily Google Trends data for "Bitcoin" over a specified window.
    
    - If end_dt is provided:  
         * Floor it to day (YYYY-MM-DD 00:00:00) and use that as the cache key.
         * Build a timeframe from (end_dt - days) to end_dt (both in YYYY-MM-DD format).
         * Cache the resulting array (list) of daily values.
    - If end_dt is None:  
         * Use the current time (floored to day) only for constructing the timeframe.
         * Do not cache the result (since the day is ongoing and values may still change).
    
    Returns:
         A list of floats representing daily Google Trends values for "Bitcoin".
    """
    cache_file = GOOGLE_TRENDS_CACHE_FILE
    cache_dict = load_json_cache(cache_file)
    
    if end_dt is not None:
        # Floor end_dt to the day (set hour, minute, second, microsecond to zero)
        key_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Use a cache key based on the day
        key = f"google-trends-{key_dt.strftime('%Y-%m-%d')}-days={days}"
    else:
        # If no end_dt is provided, we do not cache because the day is still in progress.
        key = None

    # If we have a key and it's in the cache, return the cached array.
    if key is not None and key in cache_dict:
        logger.info(f"[fetch_google_trends] Using cache for key={key}")
        # Expect cached value is already a list of numbers.
        return cache_dict[key]

    # Build the timeframe string to force daily resolution.
    # Use end_dt (if provided) or the current time, floored to day.
    if end_dt is None:
        dt_now = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt_use = dt_now
    else:
        end_dt_use = key_dt  # already floored to day

    start_dt = end_dt_use - datetime.timedelta(days=days)
    # Build timeframe string as "YYYY-MM-DD YYYY-MM-DD" (Google Trends expects two dates separated by space)
    tf_str = f"{start_dt.strftime('%Y-%m-%d')} {end_dt_use.strftime('%Y-%m-%d')}"
    
    logger.info(f"[fetch_google_trends] Fetching data for timeframe: {tf_str}")

    retries = 0
    while retries < max_retries:
        try:
            pytrends.build_payload(["Bitcoin"], cat=0, timeframe=tf_str, geo="", gprop="")
            df = pytrends.interest_over_time()
            if not df.empty:
                # Ensure daily data by converting the index to date strings and getting the corresponding values.
                # Typically, for a multi-day timeframe, Google Trends returns daily data.
                daily_values = []
                for dt in df.index:
                    daily_values.append(float(df.loc[dt, "Bitcoin"]))
                # If caching is enabled (end_dt provided), store the array.
                if key is not None:
                    save_json_cache(cache_file, {key: daily_values})
                    logger.info(f"[fetch_google_trends] Stored daily values for key={key} with {len(daily_values)} points.")
                return daily_values
            else:
                return [np.nan] * (days + 1)
        except Exception as e:
            # Check for rate-limiting (HTTP 429)
            rate_limited = False
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    rate_limited = True
            elif "429" in str(e):
                rate_limited = True

            logger.error(f"[fetch_google_trends] Error: {e}")
            if rate_limited:
                retries += 1
                if retries < max_retries:
                    logger.info(f"[fetch_google_trends] Rate-limited, retrying in {delay}s (attempt {retries}/{max_retries}).")
                    time.sleep(delay)
            else:
                # For other errors, return fallback array
                return [np.nan] * (days + 1)
    
    logger.warning("[fetch_google_trends] Exhausted retries, returning fallback nan array.")
    return [np.nan] * (days + 1)


###############################################################################
# fetch_santiment_data
###############################################################################
def fetch_santiment_data(end_dt=None, only_look_in_cache = False):
    """
    We fetch last 3 days from end_dt floored hour or now floored.
    store => input_cache/santiment_data_cache.json
    """
    cache_file = "input_cache/santiment_data_cache.json"
    cache_dict = load_json_cache(cache_file)

    if end_dt is None:
        dt_now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        key_dt = dt_now
    else:
        key_dt = end_dt.replace(minute=0, second=0, microsecond=0)

    key = f"santiment-{key_dt.strftime('%Y-%m-%d %H:%M:%S')}"

    if key in cache_dict:
        logger.info(f"[fetch_santiment_data] Using cache => {key}")
        for metric in cache_dict[key]:
            if cache_dict[key][metric] == "Null":
                cache_dict[key][metric] = np.nan
                
        return cache_dict[key]

    logger.info(f"[fetch_santiment_data] Not in cache => fetch => {key}")

    start_dt = key_dt - datetime.timedelta(days=1)
    from_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_iso   = key_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = "https://api.santiment.net/graphql"
    
    metrics = [
        "social_volume_total",
        "social_dominance",
        "sentiment_positive",
        "sentiment_negative",
        "daily_active_addresses",
        "transaction_volume",
        "exchange_inflow",
        "exchange_outflow",
        "whale_transaction_count",
        "mvrv_ratio",
        "nvt_ratio",
        "dev_activity"
    ]
    
    query_fields = []
    # BTC
    for metric in metrics:
        field = f'''
        {metric}: getMetric(metric: "{metric}") {{
            timeseriesData(slug: "bitcoin", from: "{from_iso}", to: "{to_iso}", interval: "1d") {{
                datetime
                value
            }}
        }}'''
        query_fields.append(field)
    query_fields_str = "\n".join(query_fields)
    query = f"""{{\n{query_fields_str}\n}}"""
    headers = {"Authorization": f"Apikey {SANTIMENT_API_KEY}"}
    
    results = {}
    for metric in metrics:
        results[metric] = np.nan
    
    if only_look_in_cache:
        return results
    
    max_retries = 1
    delay = 5
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, json={"query": query}, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for metric in metrics:
                metric_data = data["data"].get(metric)
                if metric_data and "timeseriesData" in metric_data:
                    timeseries = metric_data["timeseriesData"]
                    if timeseries:
                        avg_value = sum(item["value"] for item in timeseries) / len(timeseries)
                        results[metric] = avg_value
                        
            cached_results = results.copy()
            for metric in metrics:
                if np.isnan(cached_results[metric]):
                    cached_results[metric] = "Null"
                    
            save_json_cache(cache_file, {key: cached_results})
            
        except Exception as e:
            # Check for rate-limiting (HTTP 429)
            rate_limited = False
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 429:
                    rate_limited = True
            elif "429" in str(e):
                rate_limited = True

            logger.error(f"[Santiment] Error: {e}")
            if rate_limited:
                retries += 1
                if retries < max_retries:
                    logger.info(f"[Santiment] Rate-limited, retrying in {delay}s (attempt {retries}/{max_retries}).")
                    time.sleep(delay)
            else:
                # For other errors, return fallback array
                return results

    return results


###############################################################################
# fetch_reddit_sentiment
###############################################################################
def fetch_reddit_sentiment(dt: datetime.datetime = None):
    """
    If dt= None => current time => floor => key => fetch live.
    If dt => floor => if found => return. If not => find nearest, if not => fetch new => store as now floored.
    """
    cache_file = REDDIT_SENTIMENT_CACHE_FILE
    cache_data = load_json_cache(cache_file)

    if dt is None:
        dt_now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        key = dt_now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        dt_floor = dt.replace(minute=0, second=0, microsecond=0)
        key = dt_floor.strftime("%Y-%m-%d %H:%M:%S")

    if f"reddit-{key}" in cache_data:
        logger.info(f"[fetch_reddit_sentiment] Using cache => reddit-{key}")
        return float(cache_data[f"reddit-{key}"])

    if dt is not None:
        logger.info(f"[fetch_reddit_sentiment] Not found => check nearest => reddit-{key}")
        nearest = find_nearest_hour_key(cache_data, dt_floor)
        if nearest:
            best_key = f"reddit-{nearest}"
            logger.info(f"[fetch_reddit_sentiment] nearest => {best_key}")
            return float(cache_data[best_key])
        else:
            logger.info("[fetch_reddit_sentiment] no parseable => fallback => fetch now")

    # fallback => fetch live => store under now floored
    dt_now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    newkey = f"reddit-{dt_now.strftime('%Y-%m-%d %H:%M:%S')}"
    logger.info(f"[fetch_reddit_sentiment] fallback fetch => store => {newkey}")

    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        logger.info("No Reddit API credentials => 0.0 sentiment.")
        avg_sent = np.nan
    else:
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            subreddit = reddit.subreddit("Bitcoin")
            posts = list(subreddit.new(limit=20))
            posts_text = [post.title + " " + post.selftext for post in posts]
            avg_sent = compute_average_sentiment(posts_text)
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            avg_sent = 0.0

    save_json_cache(cache_file, {newkey: avg_sent})
    return avg_sent

def analyze_news_sentiment(news_articles):
    """
    Collect the text from each news article, compute average sentiment using
    compute_average_sentiment(), and store/retrieve the result from cache
    keyed by a hash of the articles.
    """
    if not news_articles:
        return np.nan

    # 1) Load the cache
    cache_data = load_json_cache(NEWS_ANALYZE_CACHE_FILE)

    # 2) Build a stable string from the articles, then hash it
    articles_str = json.dumps(news_articles, sort_keys=True)
    articles_hash = hashlib.md5(articles_str.encode("utf-8")).hexdigest()

    if articles_hash in cache_data:
        logger.info(f"[analyze_news_sentiment] Using cached news sentiment => {articles_hash}")
        return float(cache_data[articles_hash])

    # 3) Gather text from each article
    text_list = []
    for art in news_articles:
        title       = art.get("title", "") or ""
        description = art.get("description", "") or ""
        content     = art.get("content", "") or ""
        # Clean newlines, etc.
        description = description.replace("\n", " - ")
        content     = content.replace("\n", " - ")
        text_list.append(f"{title} | {description} | {content}")

    # 4) Compute the average sentiment
    avg_sentiment = compute_average_sentiment(text_list)

    # 5) Store in cache
    save_json_cache(NEWS_ANALYZE_CACHE_FILE, {articles_hash: avg_sentiment})

    return avg_sentiment