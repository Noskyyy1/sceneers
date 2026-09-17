```python

import io
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IDX Stock Screener",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 IDX Stock Screener — Technical + Fundamental"
)

st.caption(
    "Screener saham IDX untuk research: trend, momentum, "
    "volume, fundamental, relative strength, breakout, "
    "risk/reward dan AI analysis prompt."
)


# ============================================================
# DEFAULT WATCHLIST
# ============================================================

DEFAULT_TICKERS = [
    "BBCA",
    "BBRI",
    "BMRI",
    "BBNI",
    "TLKM",
    "ASII",
    "UNVR",
    "ICBP",
    "AMRT",
    "ADRO",
    "PTBA",
    "ITMG",
    "PGAS",
    "ANTM",
    "INCO",
    "CPIN",
    "BRIS",
    "MEDC",
    "MDKA",
    "AKRA",
    "AMMN",
    "BREN",
    "TPIA",
    "GOTO",
]


# ============================================================
# IDX SOURCES
# ============================================================

IDX_STOCK_PRICE_URL = (
    "https://www.idx.co.id/id/data-pasar/"
    "laporan-statistik/digital-statistic/monthly/"
    "trading-summary/table-of-stock-price/"
)

IDX_LENDABLE_URL = (
    "https://www.idx.co.id/id/market-data/"
    "securities-borrowing-and-lending/sections/"
    "lendable-stock"
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state["results"] = []

if "errors" not in st.session_state:
    st.session_state["errors"] = []

if "market" not in st.session_state:
    st.session_state["market"] = {
        "regime": "Unknown",
        "score": 50.0,
        "df": pd.DataFrame(),
    }

if "universe" not in st.session_state:
    st.session_state["universe"] = []


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_float(value, default=np.nan):
    """Convert value into float safely."""

    try:
        if value is None:
            return default

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except (TypeError, ValueError):
        return default


def format_rupiah(value):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"Rp {value:,.0f}".replace(",", ".")


def clean_ticker(value):
    """
    Normalize ticker.

    Examples:
    BBCA
    BBCA.JK
    $BBCA
    -> BBCA
    """

    if value is None:
        return ""

    ticker = str(value).upper().strip()

    ticker = ticker.replace("$", "")
    ticker = ticker.replace(".JK", "")

    ticker = re.sub(
        r"[^A-Z0-9]",
        "",
        ticker,
    )

    return ticker


# ============================================================
# LOAD ALL IDX TICKERS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_idx_tickers():
    """
    Try to retrieve the broad IDX stock universe.

    Primary source:
        IDX Table of Stock Price

    Fallback:
        IDX Lendable Stock

    Important:
    IDX website is dynamic, so parsing may occasionally change.
    The application keeps manual input available as fallback.
    """

    tickers = set()

    sources = [
        IDX_STOCK_PRICE_URL,
        IDX_LENDABLE_URL,
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    }

    # --------------------------------------------------------
    # SOURCE 1 / SOURCE 2
    # --------------------------------------------------------

    for url in sources:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=20,
            )

            if response.status_code != 200:
                continue

            html = response.text

            # ------------------------------------------------
            # Parse HTML tables
            # ------------------------------------------------

            try:

                tables = pd.read_html(
                    io.StringIO(html)
                )

            except ValueError:
                tables = []

            for table in tables:

                if table.empty:
                    continue

                # Normalize columns
                table.columns = [
                    str(col).strip()
                    for col in table.columns
                ]

                possible_code_columns = [
                    col
                    for col in table.columns
                    if (
                        "code" in col.lower()
                        or "kode" in col.lower()
                    )
                ]

                for col in possible_code_columns:

                    for value in table[col].astype(str):

                        ticker = clean_ticker(
                            value
                        )

                        # Valid IDX ticker:
                        # usually 4 letters/numbers
                        if (
                            3 <= len(ticker) <= 5
                            and ticker.isalnum()
                        ):
                            tickers.add(
                                ticker
                            )

            # ------------------------------------------------
            # Regex fallback
            # ------------------------------------------------

            if not tickers:

                patterns = re.findall(
                    r"\b[A-Z]{4}\b",
                    html,
                )

                for ticker in patterns:

                    ticker = clean_ticker(
                        ticker
                    )

                    if ticker:
                        tickers.add(
                            ticker
                        )

            # If primary source has enough tickers
            if len(tickers) >= 100:
                break

        except Exception:
            continue

    # --------------------------------------------------------
    # Remove obvious non-stock values
    # --------------------------------------------------------

    blacklist = {
        "CODE",
        "KODE",
        "NAME",
        "SECTOR",
        "STOCK",
        "INDEX",
        "DATA",
        "PRICE",
        "DATE",
        "VOLUME",
        "VALUE",
        "TOTAL",
        "TABLE",
        "PREV",
    }

    tickers = {
        ticker
        for ticker in tickers
        if ticker not in blacklist
    }

    result = sorted(
        tickers
    )

    return result


# ============================================================
# PRICE DATA
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def get_price_data(
    symbol,
    period_days=550,
):
    """
    Download OHLCV from Yahoo Finance.
    """

    end_date = datetime.now()

    start_date = (
        end_date
        - timedelta(
            days=period_days
        )
    )

    try:

        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=False,
        )

    except Exception:

        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Handle MultiIndex
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    df = df[
        required_columns
    ].copy()

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df.index = pd.to_datetime(
        df.index
    )

    return df


# ============================================================
# FUNDAMENTALS
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def get_fundamental_data(
    ticker
):
    symbol = (
        f"{ticker}.JK"
    )

    try:

        info = (
            yf.Ticker(
                symbol
            ).info
            or {}
        )

    except Exception:

        info = {}

    return {

        "trailingPE":
            safe_float(
                info.get(
                    "trailingPE"
                )
            ),

        "forwardPE":
            safe_float(
                info.get(
                    "forwardPE"
                )
            ),

        "priceToBook":
            safe_float(
                info.get(
                    "priceToBook"
                )
            ),

        "returnOnEquity":
            safe_float(
                info.get(
                    "returnOnEquity"
                )
            ),

        "dividendYield":
            safe_float(
                info.get(
                    "dividendYield"
                ),
                0.0,
            ),

        "marketCap":
            safe_float(
                info.get(
                    "marketCap"
                ),
                0.0,
            ),

        "sector":
            info.get(
                "sector"
            )
            or "Unknown",

        "industry":
            info.get(
                "industry"
            )
            or "Unknown",

        "longName":
            info.get(
                "longName"
            )
            or ticker,
    }


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_ema(
    series,
    span,
):
    return series.ewm(
        span=span,
        adjust=False,
    ).mean()


def calculate_rsi(
    series,
    length=14,
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return rsi.fillna(50)


def calculate_atr(
    df,
    length=14,
):

    previous_close = (
        df["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            (
                df["High"]
                - df["Low"]
            ),

            (
                df["High"]
                - previous_close
            ).abs(),

            (
                df["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        length
    ).mean()


def add_indicators(
    df,
    breakout_window=20,
):

    output = df.copy()

    # --------------------------------------------------------
    # Moving Average
    # --------------------------------------------------------

    output["SMA20"] = (
        output["Close"]
        .rolling(20)
        .mean()
    )

    output["SMA50"] = (
        output["Close"]
        .rolling(50)
        .mean()
    )

    output["SMA200"] = (
        output["Close"]
        .rolling(200)
        .mean()
    )

    output["EMA20"] = (
        calculate_ema(
            output["Close"],
            20,
        )
    )

    output["EMA50"] = (
        calculate_ema(
            output["Close"],
            50,
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = calculate_ema(
        output["Close"],
        12,
    )

    ema26 = calculate_ema(
        output["Close"],
        26,
    )

    output["MACD"] = (
        ema12
        - ema26
    )

    output["MACDSignal"] = (
        calculate_ema(
            output["MACD"],
            9,
        )
    )

    output["MACDHist"] = (
        output["MACD"]
        - output["MACDSignal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    output["RSI"] = calculate_rsi(
        output["Close"],
        14,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    output["ATR"] = calculate_atr(
        output,
        14,
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    output["VolumeSMA20"] = (
        output["Volume"]
        .rolling(20)
        .mean()
    )

    output["VolumeRatio"] = (
        output["Volume"]
        /
        output["VolumeSMA20"]
        .replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------

    direction = (
        np.sign(
            output["Close"].diff()
        )
        .fillna(0)
    )

    output["OBV"] = (
        direction
        * output["Volume"]
    ).cumsum()

    output["OBVMA20"] = (
        output["OBV"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # CMF
    # --------------------------------------------------------

    price_range = (
        output["High"]
        - output["Low"]
    ).replace(
        0,
        np.nan,
    )

    mfm = (
        (
            output["Close"]
            - output["Low"]
        )
        -
        (
            output["High"]
            - output["Close"]
        )
    ) / price_range

    mfv = (
        mfm
        .fillna(0)
        * output["Volume"]
    )

    output["CMF20"] = (
        mfv.rolling(20).sum()
        /
        output["Volume"]
        .rolling(20)
        .sum()
    )

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    output["BreakoutHigh"] = (
        output["High"]
        .shift(1)
        .rolling(
            breakout_window
        )
        .max()
    )

    output["BreakoutLow"] = (
        output["Low"]
        .shift(1)
        .rolling(
            breakout_window
        )
        .min()
    )

    return output


# ============================================================
# MARKET REGIME
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def get_market_regime():

    ihsg = get_price_data(
        "^JKSE",
        550,
    )

    if ihsg.empty:

        return {
            "regime": "Unknown",
            "score": 50.0,
            "df": pd.DataFrame(),
        }

    ihsg = add_indicators(
        ihsg,
        20,
    )

    latest = ihsg.iloc[-1]

    score = 50.0

    if (
        latest["Close"]
        >
        latest["SMA50"]
    ):
        score += 15
    else:
        score -= 15

    if (
        latest["Close"]
        >
        latest["SMA200"]
    ):
        score += 20
    else:
        score -= 20

    if (
        latest["MACD"]
        >
        latest["MACDSignal"]
    ):
        score += 10
    else:
        score -= 10

    if (
        latest["RSI"]
        >= 50
    ):
        score += 5
    else:
        score -= 5

    score = float(
        np.clip(
            score,
            0,
            100,
        )
    )

    if score >= 65:

        regime = "Bullish"

    elif score <= 35:

        regime = "Bearish"

    else:

        regime = "Neutral"

    return {
        "regime": regime,
        "score": score,
        "df": ihsg,
    }


# ============================================================
# RSI SCORE
# ============================================================

def score_rsi(
    value
):

    if np.isnan(value):
        return 0

    if 45 <= value <= 60:
        return 10

    if (
        40 <= value < 45
        or
        60 < value <= 65
    ):
        return 7

    if (
        35 <= value < 40
        or
        65 < value <= 70
    ):
        return 4

    return 0


# ============================================================
# FUNDAMENTAL SCORE
# ============================================================

def score_fundamental(
    per,
    pbv,
    roe_percent,
):

    score = 0.0

    signals = []

    # ROE
    if not np.isnan(
        roe_percent
    ):

        if roe_percent >= 20:

            score += 15

            signals.append(
                f"ROE sangat kuat ({roe_percent:.1f}%)"
            )

        elif roe_percent >= 15:

            score += 12

            signals.append(
                f"ROE kuat ({roe_percent:.1f}%)"
            )

        elif roe_percent >= 10:

            score += 8

            signals.append(
                f"ROE sehat ({roe_percent:.1f}%)"
            )

        elif roe_percent > 0:

            score += 3

    # PER
    if (
        not np.isnan(per)
        and per > 0
    ):

        if per <= 10:

            score += 12.5

            signals.append(
                f"PER rendah ({per:.1f}x)"
            )

        elif per <= 15:

            score += 10

            signals.append(
                f"PER relatif menarik ({per:.1f}x)"
            )

        elif per <= 20:

            score += 7

        elif per <= 30:

            score += 3

    # PBV
    if (
        not np.isnan(pbv)
        and pbv > 0
    ):

        if pbv <= 1.5:

            score += 12.5

            signals.append(
                f"PBV rendah ({pbv:.2f}x)"
            )

        elif pbv <= 2.5:

            score += 10

            signals.append(
                f"PBV cukup terjangkau ({pbv:.2f}x)"
            )

        elif pbv <= 3:

            score += 7

        elif pbv <= 5:

            score += 3

    return (
        min(score, 40),
        signals,
    )


# ============================================================
# ANALYZE STOCK
# ============================================================

def analyze_stock(
    ticker,
    market_df,
    market_regime,
    support_window,
    breakout_window,
    atr_multiplier,
):

    ticker = clean_ticker(
        ticker
    )

    if not ticker:
        return None

    symbol = (
        f"{ticker}.JK"
    )

    raw_df = get_price_data(
        symbol
    )

    if (
        raw_df.empty
        or len(raw_df) < 220
    ):
        return None

    df = add_indicators(
        raw_df,
        breakout_window,
    ).copy()

    fundamentals = (
        get_fundamental_data(
            ticker
        )
    )

    latest = df.iloc[-1]

    price = safe_float(
        latest["Close"]
    )

    if np.isnan(price):
        return None

    # --------------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------------

    per = safe_float(
        fundamentals[
            "trailingPE"
        ]
    )

    pbv = safe_float(
        fundamentals[
            "priceToBook"
        ]
    )

    raw_roe = safe_float(
        fundamentals[
            "returnOnEquity"
        ]
    )

    if np.isnan(raw_roe):
        roe_percent = np.nan
    else:
        roe_percent = (
            raw_roe * 100
        )

    dividend_yield = (
        safe_float(
            fundamentals[
                "dividendYield"
            ],
            0.0,
        )
        * 100
    )

    # --------------------------------------------------------
    # TECHNICAL SCORE
    # --------------------------------------------------------

    tech_score = 0.0

    signals = []

    if (
        latest["Close"]
        >
        latest["SMA50"]
    ):

        tech_score += 12

        signals.append(
            "Harga di atas SMA 50"
        )

    if (
        latest["Close"]
        >
        latest["SMA200"]
    ):

        tech_score += 12

        signals.append(
            "Harga di atas SMA 200"
        )

    if (
        latest["SMA50"]
        >
        latest["SMA200"]
    ):

        tech_score += 6

        signals.append(
            "SMA50 > SMA200"
        )

    if (
        latest["MACD"]
        >
        latest["MACDSignal"]
    ):

        tech_score += 10

        signals.append(
            "MACD bullish"
        )

    rsi_points = score_rsi(
        latest["RSI"]
    )

    tech_score += rsi_points

    if rsi_points >= 7:

        signals.append(
            f"RSI mendukung ({latest['RSI']:.1f})"
        )

    volume_ratio = safe_float(
        latest["VolumeRatio"]
    )

    if not np.isnan(
        volume_ratio
    ):

        if volume_ratio >= 1.5:

            tech_score += 10

            signals.append(
                f"Volume spike ({volume_ratio:.2f}x)"
            )

        elif volume_ratio >= 1.2:

            tech_score += 6

            signals.append(
                f"Volume meningkat ({volume_ratio:.2f}x)"
            )

        elif volume_ratio >= 1:

            tech_score += 3

    # --------------------------------------------------------
    # RELATIVE STRENGTH
    # --------------------------------------------------------

    stock_return_60 = np.nan

    ihsg_return_60 = np.nan

    relative_strength = np.nan

    if len(df) > 60:

        stock_old = safe_float(
            df["Close"].iloc[-61]
        )

        if stock_old > 0:

            stock_return_60 = (
                (
                    price
                    / stock_old
                )
                - 1
            ) * 100

    if (
        not market_df.empty
        and
        len(market_df) > 60
    ):

        market_now = safe_float(
            market_df[
                "Close"
            ].iloc[-1]
        )

        market_old = safe_float(
            market_df[
                "Close"
            ].iloc[-61]
        )

        if (
            market_now > 0
            and
            market_old > 0
        ):

            ihsg_return_60 = (
                (
                    market_now
                    / market_old
                )
                - 1
            ) * 100

            if not np.isnan(
                stock_return_60
            ):

                relative_strength = (
                    stock_return_60
                    -
                    ihsg_return_60
                )

    # --------------------------------------------------------
    # MOMENTUM SCORE
    # --------------------------------------------------------

    momentum_score = 0.0

    if not np.isnan(
        stock_return_60
    ):

        if stock_return_60 >= 15:

            momentum_score += 12

        elif stock_return_60 >= 8:

            momentum_score += 9

        elif stock_return_60 >= 3:

            momentum_score += 6

        elif stock_return_60 >= 0:

            momentum_score += 3

    if not np.isnan(
        relative_strength
    ):

        if relative_strength >= 10:

            momentum_score += 8

            signals.append(
                "Outperforming IHSG kuat"
            )

        elif relative_strength >= 5:

            momentum_score += 6

            signals.append(
                "Outperforming IHSG"
            )

        elif relative_strength >= 0:

            momentum_score += 3

    momentum_score = min(
        momentum_score,
        20,
    )

    # --------------------------------------------------------
    # BREAKOUT / STRUCTURE
    # --------------------------------------------------------

    structure_score = 0.0

    breakout = False

    breakout_high = safe_float(
        latest[
            "BreakoutHigh"
        ]
    )

    if (
        not np.isnan(
            breakout_high
        )
        and
        price > breakout_high
    ):

        breakout = True

        structure_score += 10

        signals.append(
            f"Breakout {breakout_window}D"
        )

    elif (
        not np.isnan(
            breakout_high
        )
        and
        price >=
        breakout_high * 0.98
    ):

        structure_score += 6

        signals.append(
            f"Near breakout {breakout_window}D"
        )

    # OBV
    obvma = safe_float(
        latest["OBVMA20"]
    )

    if (
        not np.isnan(obvma)
        and
        latest["OBV"]
        > obvma
    ):

        structure_score += 3

        signals.append(
            "OBV di atas rata-rata"
        )

    # CMF
    cmf = safe_float(
        latest["CMF20"]
    )

    if (
        not np.isnan(cmf)
        and
        cmf > 0
    ):

        structure_score += 2

        signals.append(
            "CMF positif"
        )

    structure_score = min(
        structure_score,
        15,
    )

    # --------------------------------------------------------
    # FUNDAMENTAL SCORE
    # --------------------------------------------------------

    fundamental_score, fund_signals = (
        score_fundamental(
            per,
            pbv,
            roe_percent,
        )
    )

    signals.extend(
        fund_signals
    )

    # --------------------------------------------------------
    # COMPOSITE
    # --------------------------------------------------------

    technical_total = min(
        tech_score
        +
        momentum_score * 0.5
        +
        structure_score * 0.4,
        60,
    )

    regime_adjustment = {

        "Bullish": 3,

        "Neutral": 0,

        "Bearish": -3,

        "Unknown": 0,

    }.get(
        market_regime,
        0,
    )

    composite_score = float(
        np.clip(
            technical_total
            +
            fundamental_score
            +
            regime_adjustment,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    recent = df.tail(
        support_window
    )

    support = safe_float(
        recent["Low"].min()
    )

    resistance = safe_float(
        recent["High"].max()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_value = safe_float(
        latest["ATR"]
    )

    if (
        np.isnan(atr_value)
        or atr_value <= 0
    ):

        atr_value = max(
            price * 0.03,
            1,
        )

    # --------------------------------------------------------
    # BUY ZONE
    # --------------------------------------------------------

    buy_low = max(
        support,
        price
        - 0.5 * atr_value,
    )

    buy_high = min(
        price,
        resistance
        if resistance >= price
        else price,
    )

    if buy_low > buy_high:

        buy_low = (
            price
            - 0.5 * atr_value
        )

        buy_high = price

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    stop_loss = max(
        support
        - 0.25 * atr_value,

        price
        - atr_multiplier * atr_value,
    )

    if stop_loss >= price:

        stop_loss = (
            price
            - atr_multiplier * atr_value
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    risk_per_share = max(
        price - stop_loss,
        0.01,
    )

    tp1 = (
        price
        + 1.5
        * risk_per_share
    )

    tp2 = (
        price
        + 2.5
        * risk_per_share
    )

    rr_tp1 = (
        tp1 - price
    ) / risk_per_share

    rr_tp2 = (
        tp2 - price
    ) / risk_per_share

    risk_pct = (
        (
            price
            - stop_loss
        )
        / price
    ) * 100

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    if breakout:

        setup = "Breakout"

    elif (
        price
        > latest["SMA50"]
        and
        latest["MACD"]
        > latest["MACDSignal"]
    ):

        setup = "Trend Continuation"

    elif (
        price
        >= latest["SMA50"] * 0.97
    ):

        setup = "Pullback Watch"

    else:

        setup = "Wait / Weak Setup"

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    if risk_pct <= 4:

        risk_label = "Low"

    elif risk_pct <= 8:

        risk_label = "Medium"

    else:

        risk_label = "High"

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    missing_fundamental = sum(
        np.isnan(value)
        for value in [
            per,
            pbv,
            roe_percent,
        ]
    )

    if missing_fundamental >= 2:

        data_quality = (
            "Limited fundamental data"
        )

    elif missing_fundamental == 1:

        data_quality = (
            "Partial fundamental data"
        )

    else:

        data_quality = "Good"

    return {

        "Ticker":
            ticker,

        "Name":
            fundamentals[
                "longName"
            ],

        "Sector":
            fundamentals[
                "sector"
            ],

        "Industry":
            fundamentals[
                "industry"
            ],

        "Price":
            price,

        "Score":
            round(
                composite_score,
                1,
            ),

        "TechnicalScore":
            round(
                technical_total,
                1,
            ),

        "FundamentalScore":
            round(
                fundamental_score,
                1,
            ),

        "MomentumScore":
            round(
                momentum_score,
                1,
            ),

        "StructureScore":
            round(
                structure_score,
                1,
            ),

        "PER":
            per,

        "PBV":
            pbv,

        "ROE":
            roe_percent,

        "DividendYield":
            dividend_yield,

        "RSI":
            safe_float(
                latest["RSI"]
            ),

        "SMA50":
            safe_float(
                latest["SMA50"]
            ),

        "SMA200":
            safe_float(
                latest["SMA200"]
            ),

        "MACD":
            safe_float(
                latest["MACD"]
            ),

        "MACDSignal":
            safe_float(
                latest["MACDSignal"]
            ),

        "VolumeRatio":
            volume_ratio,

        "ATR":
            atr_value,

        "StockReturn60D":
            stock_return_60,

        "IHSGReturn60D":
            ihsg_return_60,

        "RelativeStrength":
            relative_strength,

        "Support":
            support,

        "Resistance":
            resistance,

        "BuyLow":
            buy_low,

        "BuyHigh":
            buy_high,

        "StopLoss":
            stop_loss,

        "TP1":
            tp1,

        "TP2":
            tp2,

        "RRTP1":
            rr_tp1,

        "RRTP2":
            rr_tp2,

        "RiskPct":
            risk_pct,

        "RiskLabel":
            risk_label,

        "Setup":
            setup,

        "Breakout":
            breakout,

        "DataQuality":
            data_quality,

        "Signals":
            signals,

        "DF":
            df,
    }


# ============================================================
# RUN SCREENING
# ============================================================

def run_screening(
    tickers,
    minimum_score,
    support_window,
    breakout_window,
    atr_mult,
    sector_filter,
    min_price,
    min_avg_volume,
):

    market = (
        get_market_regime()
    )

    market_df = market[
        "df"
    ]

    market_regime = market[
        "regime"
    ]

    results = []

    errors = []

    progress = st.progress(
        0
    )

    status = st.empty()

    total = max(
        len(tickers),
        1,
    )

    for index, ticker in enumerate(
        tickers,
        start=1,
    ):

        status.info(
            f"Menganalisis {ticker} "
            f"({index}/{total})..."
        )

        try:

            # ------------------------------------------------
            # Quick price pre-filter
            # ------------------------------------------------

            quick_df = get_price_data(
                f"{ticker}.JK",
                550,
            )

            if (
                quick_df.empty
                or len(quick_df) < 220
            ):

                errors.append(
                    f"{ticker}: "
                    "historical data tidak cukup"
                )

                progress.progress(
                    index / total
                )

                continue

            latest_close = safe_float(
                quick_df[
                    "Close"
                ].iloc[-1]
            )

            avg_volume_20 = safe_float(
                quick_df[
                    "Volume"
                ]
                .rolling(20)
                .mean()
                .iloc[-1]
            )

            # Price filter
            if (
                not np.isnan(min_price)
                and
                latest_close < min_price
            ):

                progress.progress(
                    index / total
                )

                continue

            # Volume filter
            if (
                not np.isnan(
                    min_avg_volume
                )
                and
                avg_volume_20
                < min_avg_volume
            ):

                progress.progress(
                    index / total
                )

                continue

            # ------------------------------------------------
            # Full analysis
            # ------------------------------------------------

            result = analyze_stock(
                ticker=ticker,
                market_df=market_df,
                market_regime=market_regime,
                support_window=support_window,
                breakout_window=breakout_window,
                atr_multiplier=atr_mult,
            )

            if result is None:

                errors.append(
                    f"{ticker}: "
                    "analysis gagal"
                )

            else:

                # Sector filter
                if (
                    sector_filter
                    != "All"
                    and
                    result["Sector"]
                    != sector_filter
                ):

                    progress.progress(
                        index / total
                    )

                    continue

                # Score filter
                if (
                    result["Score"]
                    >= minimum_score
                ):

                    results.append(
                        result
                    )

        except Exception as exc:

            errors.append(
                f"{ticker}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        progress.progress(
            index / total
        )

    status.empty()

    progress.empty()

    # Sort score
    results.sort(
        key=lambda x: x["Score"],
        reverse=True,
    )

    return (
        results,
        errors,
        market,
    )


# ============================================================
# SUMMARY DATAFRAME
# ============================================================

def create_summary_dataframe(
    results
):

    rows = []

    for rank, stock in enumerate(
        results,
        start=1,
    ):

        rows.append(
            {

                "Rank":
                    rank,

                "Ticker":
                    stock["Ticker"],

                "Score":
                    stock["Score"],

                "Setup":
                    stock["Setup"],

                "Technical":
                    stock["TechnicalScore"],

                "Fundamental":
                    stock["FundamentalScore"],

                "Momentum":
                    stock["MomentumScore"],

                "Sector":
                    stock["Sector"],

                "PER":
                    (
                        None
                        if np.isnan(
                            stock["PER"]
                        )
                        else round(
                            stock["PER"],
                            2,
                        )
                    ),

                "PBV":
                    (
                        None
                        if np.isnan(
                            stock["PBV"]
                        )
                        else round(
                            stock["PBV"],
                            2,
                        )
                    ),

                "ROE %":
                    (
                        None
                        if np.isnan(
                            stock["ROE"]
                        )
                        else round(
                            stock["ROE"],
                            2,
                        )
                    ),

                "RSI":
                    round(
                        stock["RSI"],
                        2,
                    ),

                "Volume x":
                    (
                        None
                        if np.isnan(
                            stock["VolumeRatio"]
                        )
                        else round(
                            stock["VolumeRatio"],
                            2,
                        )
                    ),

                "RS vs IHSG %":
                    (
                        None
                        if np.isnan(
                            stock[
                                "RelativeStrength"
                            ]
                        )
                        else round(
                            stock[
                                "RelativeStrength"
                            ],
                            2,
                        )
                    ),

                "Risk":
                    stock["RiskLabel"],

                "Price":
                    round(
                        stock["Price"],
                        2,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# AI PROMPT
# ============================================================

def create_ai_prompt(
    stock,
    market,
    horizon_days,
):

    per_text = (
        "N/A"
        if np.isnan(
            stock["PER"]
        )
        else
        f"{stock['PER']:.2f}x"
    )

    pbv_text = (
        "N/A"
        if np.isnan(
            stock["PBV"]
        )
        else
        f"{stock['PBV']:.2f}x"
    )

    roe_text = (
        "N/A"
        if np.isnan(
            stock["ROE"]
        )
        else
        f"{stock['ROE']:.2f}%"
    )

    volume_text = (
        "N/A"
        if np.isnan(
            stock["VolumeRatio"]
        )
        else
        f"{stock['VolumeRatio']:.2f}x"
    )

    return f"""
[SYSTEM]
Anda adalah Equity Research Analyst yang melakukan
analisis objektif terhadap saham di Bursa Efek Indonesia.

Jangan mengarang data yang tidak tersedia.
Gunakan data yang diberikan.
Pisahkan fakta, interpretasi dan ketidakpastian.

==================================================
STOCK
==================================================

Ticker:
{stock["Ticker"]}

Nama:
{stock["Name"]}

Sector:
{stock["Sector"]}

Industry:
{stock["Industry"]}

Harga:
{format_rupiah(stock["Price"])}

==================================================
FUNDAMENTAL
==================================================

PER:
{per_text}

PBV:
{pbv_text}

ROE:
{roe_text}

Dividend Yield:
{stock["DividendYield"]:.2f}%

==================================================
TECHNICAL
==================================================

RSI:
{stock["RSI"]:.2f}

SMA 50:
{stock["SMA50"]:.2f}

SMA 200:
{stock["SMA200"]:.2f}

MACD:
{stock["MACD"]:.4f}

MACD Signal:
{stock["MACDSignal"]:.4f}

Volume Ratio:
{volume_text}

ATR:
{stock["ATR"]:.2f}

==================================================
RELATIVE STRENGTH
==================================================

Return Saham 60D:
{
    "N/A"
    if np.isnan(stock["StockReturn60D"])
    else f"{stock['StockReturn60D']:.2f}%"
}

Return IHSG 60D:
{
    "N/A"
    if np.isnan(stock["IHSGReturn60D"])
    else f"{stock['IHSGReturn60D']:.2f}%"
}

Relative Strength:
{
    "N/A"
    if np.isnan(stock["RelativeStrength"])
    else f"{stock['RelativeStrength']:.2f}%"
}

==================================================
MARKET REGIME
==================================================

IHSG:
{market["regime"]}

Market Score:
{market["score"]:.1f}/100

==================================================
STRUCTURE
==================================================

Setup:
{stock["Setup"]}

Breakout:
{
    "YES"
    if stock["Breakout"]
    else "NO"
}

Support:
{stock["Support"]:.2f}

Resistance:
{stock["Resistance"]:.2f}

==================================================
RISK / REWARD
==================================================

Buy Zone:
{stock["BuyLow"]:.2f}
-
{stock["BuyHigh"]:.2f}

Stop Loss:
{stock["StopLoss"]:.2f}

TP1:
{stock["TP1"]:.2f}

TP2:
{stock["TP2"]:.2f}

Risk:
{stock["RiskPct"]:.2f}%

R:R TP1:
1:{stock["RRTP1"]:.2f}

R:R TP2:
1:{stock["RRTP2"]:.2f}

==================================================
SCORE
==================================================

Technical:
{stock["TechnicalScore"]}/60

Fundamental:
{stock["FundamentalScore"]}/40

Composite:
{stock["Score"]}/100

==================================================
TASK
==================================================

Buat analisis terstruktur:

1. Ringkasan kondisi saham.

2. Analisis fundamental:
   - PER
   - PBV
   - ROE
   - Dividend Yield

3. Analisis teknikal:
   - Trend
   - RSI
   - MACD
   - Volume
   - SMA50
   - SMA200

4. Analisis relative strength terhadap IHSG.

5. Analisis support dan resistance.

6. Evaluasi setup:
   - Breakout
   - Trend Continuation
   - Pullback
   - Wait

7. Evaluasi risk/reward.

8. Buat tiga skenario:

BULL:
- trigger
- area penting
- potensi target

BASE:
- kondisi yang perlu bertahan

BEAR:
- trigger negatif
- invalidation

9. Jelaskan risiko utama.

10. Beri daftar data tambahan yang perlu diverifikasi.

Jangan memberikan kepastian keuntungan.
Jangan mengarang berita atau katalis yang tidak tersedia.
"""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Screening Parameters"
)

universe_mode = st.sidebar.radio(
    "Universe Saham",
    [
        "Default Watchlist",
        "Input Manual",
        "Seluruh IDX",
    ],
)

# ------------------------------------------------------------
# Watchlist
# ------------------------------------------------------------

if universe_mode == "Default Watchlist":

    selected_tickers = (
        st.sidebar.multiselect(
            "Pilih Saham",
            options=DEFAULT_TICKERS,
            default=DEFAULT_TICKERS[:12],
        )
    )

# ------------------------------------------------------------
# Manual
# ------------------------------------------------------------

elif universe_mode == "Input Manual":

    manual_text = (
        st.sidebar.text_area(
            "Masukkan kode saham",
            placeholder=(
                "Contoh:\n"
                "ACES,GJTL,SMGR,ERAA\n"
                "atau satu ticker per baris"
            ),
            height=120,
        )
    )

    # Support commas / spaces / newline
    manual_tickers = re.split(
        r"[\s,;]+",
        manual_text,
    )

    selected_tickers = sorted(
        {
            clean_ticker(
                ticker
            )
            for ticker in manual_tickers
            if clean_ticker(
                ticker
            )
        }
    )

# ------------------------------------------------------------
# Entire IDX
# ------------------------------------------------------------

else:

    if not st.session_state[
        "universe"
    ]:

        with st.spinner(
            "Memuat daftar saham IDX..."
        ):

            st.session_state[
                "universe"
            ] = (
                get_all_idx_tickers()
            )

    all_idx_tickers = (
        st.session_state[
            "universe"
        ]
    )

    if all_idx_tickers:

        selected_tickers = (
            all_idx_tickers
        )

        st.sidebar.success(
            f"{len(all_idx_tickers)} "
            "ticker berhasil dimuat"
        )

    else:

        selected_tickers = []

        st.sidebar.error(
            "Daftar IDX tidak berhasil "
            "dibaca otomatis. Gunakan "
            "'Input Manual'."
        )

# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------

st.sidebar.divider()

min_score = st.sidebar.slider(
    "Minimum Composite Score",
    0,
    100,
    55,
)

min_price = st.sidebar.number_input(
    "Minimum Harga",
    min_value=0.0,
    value=0.0,
    step=100.0,
)

min_avg_volume = st.sidebar.number_input(
    "Minimum Average Volume 20D",
    min_value=0.0,
    value=0.0,
    step=100000.0,
    help=(
        "Gunakan 0 untuk menonaktifkan filter."
    ),
)

support_lookback = st.sidebar.slider(
    "Support / Resistance Lookback",
    20,
    120,
    60,
    5,
)

breakout_lookback = st.sidebar.slider(
    "Breakout Lookback",
    10,
    100,
    20,
    5,
)

atr_multiplier = st.sidebar.slider(
    "ATR Multiplier",
    0.5,
    3.0,
    1.5,
    0.1,
)

holding_days = st.sidebar.slider(
    "Swing Horizon / Trading Days",
    10,
    90,
    60,
    5,
)


# ============================================================
# SECTOR FILTER OPTIONS
# ============================================================

SECTOR_OPTIONS = [
    "All",
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown",
]

sector_filter = st.sidebar.selectbox(
    "Filter Sector",
    SECTOR_OPTIONS,
)


# ============================================================
# CACHE / RESET
# ============================================================

st.sidebar.divider()

if st.sidebar.button(
    "🔄 Refresh IDX Universe"
):

    get_all_idx_tickers.clear()

    st.session_state[
        "universe"
    ] = []

    st.rerun()


if st.sidebar.button(
    "🧹 Clear All Cache"
):

    st.cache_data.clear()

    st.session_state[
        "results"
    ] = []

    st.session_state[
        "errors"
    ] = []

    st.session_state[
        "universe"
    ] = []

    st.rerun()


# ============================================================
# RUN BUTTON
# ============================================================

run_clicked = st.button(
    "🚀 JALANKAN HYBRID SCREENING",
    type="primary",
    use_container_width=True,
)


if run_clicked:

    if not selected_tickers:

        st.warning(
            "Tidak ada ticker untuk dianalisis."
        )

    else:

        with st.spinner(
            "Menjalankan screening..."
        ):

            (
                results,
                errors,
                market,
            ) = run_screening(

                tickers=
                selected_tickers,

                minimum_score=
                min_score,

                support_window=
                support_lookback,

                breakout_window=
                breakout_lookback,

                atr_mult=
                atr_multiplier,

                sector_filter=
                sector_filter,

                min_price=
                min_price,

                min_avg_volume=
                min_avg_volume,
            )

            st.session_state[
                "results"
            ] = results

            st.session_state[
                "errors"
            ] = errors

            st.session_state[
                "market"
            ] = market


# ============================================================
# LOAD RESULTS
# ============================================================

results = st.session_state[
    "results"
]

errors = st.session_state[
    "errors"
]

market = st.session_state[
    "market"
]


# ============================================================
# MARKET STATUS
# ============================================================

m1, m2, m3, m4 = (
    st.columns(4)
)

m1.metric(
    "IHSG Market Regime",
    market.get(
        "regime",
        "Unknown",
    ),
)

m2.metric(
    "Market Score",
    (
        f"{market.get('score', 50):.0f}"
        "/100"
    ),
)

m3.metric(
    "Stocks Passed",
    len(results),
)

m4.metric(
    "Minimum Score",
    min_score,
)


# ============================================================
# EMPTY
# ============================================================

if not results:

    st.info(
        "Belum ada hasil. "
        "Pilih universe kemudian klik "
        "'JALANKAN HYBRID SCREENING'."
    )

    if errors:

        with st.expander(
            f"⚠️ Data issues ({len(errors)})"
        ):

            for error in errors:

                st.write(
                    f"- {error}"
                )

    st.stop()


# ============================================================
# RESULT HEADER
# ============================================================

st.success(
    f"{len(results)} saham lolos "
    f"minimum score {min_score}."
)


# ============================================================
# ERRORS
# ============================================================

if errors:

    with st.expander(
        f"⚠️ Data issues ({len(errors)})"
    ):

        for error in errors:

            st.write(
                f"- {error}"
            )


# ============================================================
# RANKING
# ============================================================

st.subheader(
    "🏆 Ranking Screening"
)

summary_df = (
    create_summary_dataframe(
        results
    )
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# EXPORT CSV
# ============================================================

csv_data = (
    summary_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)

st.download_button(
    "⬇️ Download CSV",
    data=csv_data,
    file_name=(
        "idx_screener_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    ),
    mime="text/csv",
)


# ============================================================
# DETAIL SELECTOR
# ============================================================

st.divider()

st.subheader(
    "🔎 Analisis Detail Saham"
)

selected_stock = st.selectbox(
    "Pilih Saham",
    options=[
        result["Ticker"]
        for result in results
    ],
)


stock = next(
    result
    for result in results
    if result["Ticker"]
    ==
    selected_stock
)


# ============================================================
# STOCK SUMMARY
# ============================================================

s1, s2, s3, s4, s5 = (
    st.columns(5)
)

s1.metric(
    "Harga",
    format_rupiah(
        stock["Price"]
    ),
)

s2.metric(
    "Composite Score",
    f"{stock['Score']:.1f}/100",
)

s3.metric(
    "Setup",
    stock["Setup"],
)

s4.metric(
    "Risk",
    stock["RiskLabel"],
)

s5.metric(
    "RS vs IHSG",
    (
        "N/A"
        if np.isnan(
            stock["RelativeStrength"]
        )
        else
        f"{stock['RelativeStrength']:.2f}%"
    ),
)


# ============================================================
# FUNDAMENTAL
# ============================================================

st.markdown(
    "### 💰 Fundamental"
)

f1, f2, f3, f4, f5 = (
    st.columns(5)
)

f1.metric(
    "PER",
    (
        "N/A"
        if np.isnan(stock["PER"])
        else
        f"{stock['PER']:.2f}x"
    ),
)

f2.metric(
    "PBV",
    (
        "N/A"
        if np.isnan(stock["PBV"])
        else
        f"{stock['PBV']:.2f}x"
    ),
)

f3.metric(
    "ROE",
    (
        "N/A"
        if np.isnan(stock["ROE"])
        else
        f"{stock['ROE']:.2f}%"
    ),
)

f4.metric(
    "Dividend Yield",
    f"{stock['DividendYield']:.2f}%",
)

f5.metric(
    "Sector",
    stock["Sector"],
)


# ============================================================
# TECHNICAL METRICS
# ============================================================

st.markdown(
    "### 📈 Technical"
)

t1, t2, t3, t4, t5, t6 = (
    st.columns(6)
)

t1.metric(
    "RSI",
    f"{stock['RSI']:.2f}",
)

t2.metric(
    "SMA 50",
    f"{stock['SMA50']:.2f}",
)

t3.metric(
    "SMA 200",
    f"{stock['SMA200']:.2f}",
)

t4.metric(
    "MACD",
    f"{stock['MACD']:.4f}",
)

t5.metric(
    "Volume Ratio",
    (
        "N/A"
        if np.isnan(
            stock["VolumeRatio"]
        )
        else
        f"{stock['VolumeRatio']:.2f}x"
    ),
)

t6.metric(
    "ATR",
    f"{stock['ATR']:.2f}",
)


# ============================================================
# PRICE CHART
# ============================================================

chart_df = (
    stock["DF"]
    .tail(180)
)

fig = go.Figure()

fig.add_trace(
    go.Candlestick(

        x=chart_df.index,

        open=chart_df[
            "Open"
        ],

        high=chart_df[
            "High"
        ],

        low=chart_df[
            "Low"
        ],

        close=chart_df[
            "Close"
        ],

        name="Price",
    )
)

fig.add_trace(
    go.Scatter(

        x=chart_df.index,

        y=chart_df[
            "SMA20"
        ],

        name="SMA 20",

        line=dict(
            width=1
        ),
    )
)

fig.add_trace(
    go.Scatter(

        x=chart_df.index,

        y=chart_df[
            "SMA50"
        ],

        name="SMA 50",

        line=dict(
            width=1.5
        ),
    )
)

fig.add_trace(
    go.Scatter(

        x=chart_df.index,

        y=chart_df[
            "SMA200"
        ],

        name="SMA 200",

        line=dict(
            width=1.5
        ),
    )
)


# Support
fig.add_hline(
    y=stock[
        "Support"
    ],
    line_dash="dot",
    annotation_text="Support",
)


# Resistance
fig.add_hline(
    y=stock[
        "Resistance"
    ],
    line_dash="dot",
    annotation_text="Resistance",
)


# Stop
fig.add_hline(
    y=stock[
        "StopLoss"
    ],
    line_dash="dash",
    annotation_text="SL",
)


# TP1
fig.add_hline(
    y=stock[
        "TP1"
    ],
    line_dash="dash",
    annotation_text="TP1",
)


# TP2
fig.add_hline(
    y=stock[
        "TP2"
    ],
    line_dash="dash",
    annotation_text="TP2",
)


fig.update_layout(

    title=(
        f"{selected_stock} — "
        "Price Structure"
    ),

    xaxis_rangeslider_visible=False,

    template="plotly_white",

    height=650,

    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# RSI
# ============================================================

rsi_fig = go.Figure()

rsi_fig.add_trace(
    go.Scatter(

        x=chart_df.index,

        y=chart_df[
            "RSI"
        ],

        name="RSI",
    )
)

rsi_fig.add_hline(
    y=70,
    line_dash="dot",
    annotation_text="70",
)

rsi_fig.add_hline(
    y=30,
    line_dash="dot",
    annotation_text="30",
)

rsi_fig.update_layout(

    title="RSI 14",

    template="plotly_white",

    height=300,

    yaxis=dict(
        range=[
            0,
            100,
        ]
    ),
)

st.plotly_chart(
    rsi_fig,
    use_container_width=True,
)


# ============================================================
# SIGNAL + RISK REWARD
# ============================================================

left, right = st.columns(
    2
)


with left:

    st.markdown(
        "### 🚦 Signals"
    )

    if stock["Signals"]:

        for signal in stock[
            "Signals"
        ][:15]:

            st.write(
                f"• {signal}"
            )

    else:

        st.write(
            "Tidak ada signal khusus."
        )

    st.caption(
        "Data quality: "
        f"{stock['DataQuality']}"
    )


with right:

    st.markdown(
        "### 🎯 Risk / Reward"
    )

    rr_df = pd.DataFrame(
        {

            "Level": [
                "Buy Low",
                "Buy High",
                "Support",
                "Resistance",
                "Stop Loss",
                "TP1",
                "TP2",
            ],

            "Price": [
                stock["BuyLow"],
                stock["BuyHigh"],
                stock["Support"],
                stock["Resistance"],
                stock["StopLoss"],
                stock["TP1"],
                stock["TP2"],
            ],
        }
    )

    st.dataframe(
        rr_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Risk: {stock['RiskPct']:.2f}% "
        f"| R:R TP1 = 1:{stock['RRTP1']:.2f} "
        f"| R:R TP2 = 1:{stock['RRTP2']:.2f}"
    )


# ============================================================
# AI PROMPT
# ============================================================

st.divider()

st.subheader(
    "🤖 AI Equity Analyst Prompt"
)

ai_prompt = create_ai_prompt(
    stock,
    market,
    holding_days,
)

st.code(
    ai_prompt,
    language="markdown",
)

st.download_button(
    "⬇️ Download AI Prompt",
    data=ai_prompt.encode(
        "utf-8"
    ),
    file_name=(
        f"AI_prompt_"
        f"{stock['Ticker']}.txt"
    ),
    mime="text/plain",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Disclaimer: aplikasi ini adalah alat research/screening. "
    "Data dari provider eksternal dapat terlambat, tidak lengkap, "
    "atau berubah. Score, entry, SL dan target merupakan hasil "
    "formula aplikasi dan bukan jaminan hasil investasi."
)
```
