import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="IDX Swing & Fundamental Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 IDX Stock Screener — Technical + Fundamental")
st.caption(
    "Hybrid screener untuk research saham IDX: trend, momentum, volume, valuasi, "
    "relative strength, breakout, risk/reward, dan auto AI prompt."
)


# =========================================================
# CONSTANTS
# =========================================================
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


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parameter Screening")

selected_tickers = st.sidebar.multiselect(
    "Pilih Daftar Saham IDX",
    options=DEFAULT_TICKERS,
    default=DEFAULT_TICKERS[:12],
)

min_score = st.sidebar.slider(
    "Minimal Composite Score",
    min_value=0,
    max_value=100,
    value=55,
    step=1,
)

holding_days = st.sidebar.slider(
    "Horizon Swing (hari bursa)",
    min_value=10,
    max_value=90,
    value=60,
    step=5,
)

atr_multiplier = st.sidebar.slider(
    "ATR Multiplier untuk SL",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
)

support_lookback = st.sidebar.slider(
    "Lookback Support / Resistance",
    min_value=20,
    max_value=120,
    value=60,
    step=5,
)

breakout_lookback = st.sidebar.slider(
    "Breakout Lookback",
    min_value=10,
    max_value=100,
    value=20,
    step=5,
)

st.sidebar.divider()

if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info(
    "Data harga dan fundamental berasal dari Yahoo Finance melalui yfinance. "
    "Gunakan hasil sebagai alat research, bukan jaminan hasil investasi."
)


# =========================================================
# UTILS
# =========================================================
def safe_float(value, default=np.nan):
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


# =========================================================
# PRICE DATA
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_price_data(symbol: str, period_days: int = 550) -> pd.DataFrame:
    """
    Download OHLCV data.

    symbol:
        BBCA.JK
        BBRI.JK
        ^JKSE

    period_days:
        History buffer untuk SMA200 dan indikator lainnya.
    """

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

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

    # Handle MultiIndex dari yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        return pd.DataFrame()

    df = df[required_columns].copy()

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    df.index = pd.to_datetime(df.index)

    return df


# =========================================================
# FUNDAMENTAL DATA
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_data(ticker: str) -> dict:

    symbol = f"{ticker}.JK"

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}

    return {
        "trailingPE": safe_float(
            info.get("trailingPE")
        ),

        "forwardPE": safe_float(
            info.get("forwardPE")
        ),

        "priceToBook": safe_float(
            info.get("priceToBook")
        ),

        "returnOnEquity": safe_float(
            info.get("returnOnEquity")
        ),

        "dividendYield": safe_float(
            info.get("dividendYield"),
            0.0,
        ),

        "marketCap": safe_float(
            info.get("marketCap"),
            0.0,
        ),

        "sector": info.get("sector") or "Unknown",

        "industry": info.get("industry") or "Unknown",

        "longName": info.get("longName") or ticker,
    }


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def ema(series, span):
    return series.ewm(
        span=span,
        adjust=False,
    ).mean()


def calculate_rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(50)


def calculate_atr(df, length=14):

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],

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

    # Moving averages
    output["SMA_20"] = (
        output["Close"]
        .rolling(20)
        .mean()
    )

    output["SMA_50"] = (
        output["Close"]
        .rolling(50)
        .mean()
    )

    output["SMA_200"] = (
        output["Close"]
        .rolling(200)
        .mean()
    )

    # EMA
    output["EMA_20"] = ema(
        output["Close"],
        20,
    )

    output["EMA_50"] = ema(
        output["Close"],
        50,
    )

    # MACD
    ema12 = ema(
        output["Close"],
        12,
    )

    ema26 = ema(
        output["Close"],
        26,
    )

    output["MACD"] = (
        ema12 - ema26
    )

    output["MACD_Signal"] = ema(
        output["MACD"],
        9,
    )

    output["MACD_Hist"] = (
        output["MACD"]
        - output["MACD_Signal"]
    )

    # RSI
    output["RSI"] = calculate_rsi(
        output["Close"],
        14,
    )

    # ATR
    output["ATR"] = calculate_atr(
        output,
        14,
    )

    # Volume
    output["Vol_SMA20"] = (
        output["Volume"]
        .rolling(20)
        .mean()
    )

    output["Volume_Ratio"] = (
        output["Volume"]
        / output["Vol_SMA20"]
        .replace(0, np.nan)
    )

    # =====================================================
    # OBV
    # =====================================================
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

    # =====================================================
    # CMF
    # =====================================================
    spread = (
        output["High"]
        - output["Low"]
    ).replace(
        0,
        np.nan,
    )

    money_flow_multiplier = (
        (
            output["Close"]
            - output["Low"]
        )
        - (
            output["High"]
            - output["Close"]
        )
    ) / spread

    money_flow_volume = (
        money_flow_multiplier
        .fillna(0)
        * output["Volume"]
    )

    output["CMF20"] = (
        money_flow_volume
        .rolling(20)
        .sum()
        /
        output["Volume"]
        .rolling(20)
        .sum()
    )

    # =====================================================
    # BREAKOUT
    # =====================================================
    output["Breakout_High"] = (
        output["High"]
        .shift(1)
        .rolling(
            breakout_window
        )
        .max()
    )

    output["Breakout_Low"] = (
        output["Low"]
        .shift(1)
        .rolling(
            breakout_window
        )
        .min()
    )

    return output


# =========================================================
# IHSG MARKET REGIME
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_market_regime():

    # Important:
    # IHSG menggunakan ^JKSE langsung,
    # bukan ^JKSE.JK
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
        > latest["SMA_50"]
    ):
        score += 15
    else:
        score -= 15

    if (
        latest["Close"]
        > latest["SMA_200"]
    ):
        score += 20
    else:
        score -= 20

    if (
        latest["MACD"]
        > latest["MACD_Signal"]
    ):
        score += 10
    else:
        score -= 10

    if latest["RSI"] >= 50:
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


# =========================================================
# RSI SCORE
# =========================================================
def score_rsi(value):

    if np.isnan(value):
        return 0

    if 45 <= value <= 60:
        return 10

    if (
        40 <= value < 45
        or 60 < value <= 65
    ):
        return 7

    if (
        35 <= value < 40
        or 65 < value <= 70
    ):
        return 4

    return 0


# =========================================================
# FUNDAMENTAL SCORE
# =========================================================
def score_fundamental(
    per,
    pbv,
    roe_percent,
):

    score = 0.0

    signals = []

    # =====================================================
    # ROE
    # Maximum = 15
    # =====================================================
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

    # =====================================================
    # PER
    # Maximum = 12.5
    # =====================================================
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

    # =====================================================
    # PBV
    # Maximum = 12.5
    # =====================================================
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


# =========================================================
# MAIN STOCK ANALYSIS
# =========================================================
def analyze_stock(
    ticker,
    market_df,
    market_regime,
    support_window,
    breakout_window,
    atr_mult,
):

    raw_df = get_price_data(
        f"{ticker}.JK"
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
        get_fundamental_data(ticker)
    )

    latest = df.iloc[-1]

    price = safe_float(
        latest["Close"]
    )

    if np.isnan(price):
        return None

    # =====================================================
    # FUNDAMENTAL
    # =====================================================
    per = safe_float(
        fundamentals["trailingPE"]
    )

    pbv = safe_float(
        fundamentals["priceToBook"]
    )

    raw_roe = fundamentals[
        "returnOnEquity"
    ]

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
            0,
        )
        * 100
    )

    # =====================================================
    # TECHNICAL SCORE
    # Maximum target around 60
    # =====================================================
    tech_score = 0.0

    signals = []

    # SMA50
    if (
        latest["Close"]
        > latest["SMA_50"]
    ):

        tech_score += 12

        signals.append(
            "Harga di atas SMA 50"
        )

    # SMA200
    if (
        latest["Close"]
        > latest["SMA_200"]
    ):

        tech_score += 12

        signals.append(
            "Harga di atas SMA 200"
        )

    # Trend alignment
    if (
        latest["SMA_50"]
        > latest["SMA_200"]
    ):

        tech_score += 6

        signals.append(
            "Golden trend: SMA50 > SMA200"
        )

    # MACD
    if (
        latest["MACD"]
        > latest["MACD_Signal"]
    ):

        tech_score += 10

        signals.append(
            "MACD bullish"
        )

    # RSI
    rsi_score = score_rsi(
        latest["RSI"]
    )

    tech_score += rsi_score

    if rsi_score >= 7:

        signals.append(
            f"RSI mendukung ({latest['RSI']:.1f})"
        )

    # Volume
    volume_ratio = safe_float(
        latest["Volume_Ratio"]
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

    # =====================================================
    # RELATIVE STRENGTH
    # =====================================================
    stock_return_60 = np.nan
    ihsg_return_60 = np.nan
    relative_strength = np.nan

    if len(df) > 60:

        old_stock_price = safe_float(
            df["Close"].iloc[-61]
        )

        if (
            old_stock_price > 0
        ):

            stock_return_60 = (
                (
                    price
                    / old_stock_price
                )
                - 1
            ) * 100

    if (
        not market_df.empty
        and len(market_df) > 60
    ):

        market_now = safe_float(
            market_df[
                "Close"
            ].iloc[-1]
        )

        market_then = safe_float(
            market_df[
                "Close"
            ].iloc[-61]
        )

        if (
            market_now > 0
            and market_then > 0
        ):

            ihsg_return_60 = (
                (
                    market_now
                    / market_then
                )
                - 1
            ) * 100

            if not np.isnan(
                stock_return_60
            ):

                relative_strength = (
                    stock_return_60
                    - ihsg_return_60
                )

    # =====================================================
    # MOMENTUM SCORE
    # Maximum 20
    # =====================================================
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

    # =====================================================
    # BREAKOUT / STRUCTURE
    # =====================================================
    structure_score = 0.0

    breakout = False

    breakout_high = safe_float(
        latest["Breakout_High"]
    )

    if (
        not np.isnan(
            breakout_high
        )
        and price > breakout_high
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
        and price >= breakout_high * 0.98
    ):

        structure_score += 6

        signals.append(
            f"Near breakout {breakout_window}D"
        )

    # OBV
    obv_average = (
        df["OBV"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    if (
        not np.isnan(obv_average)
        and latest["OBV"]
        > obv_average
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
        and cmf > 0
    ):

        structure_score += 2

        signals.append(
            "CMF positif"
        )

    structure_score = min(
        structure_score,
        15,
    )

    # =====================================================
    # FUNDAMENTAL SCORE
    # =====================================================
    fund_score, fund_signals = (
        score_fundamental(
            per,
            pbv,
            roe_percent,
        )
    )

    signals.extend(
        fund_signals
    )

    # =====================================================
    # TECHNICAL TOTAL
    # =====================================================
    technical_total = min(
        tech_score
        + momentum_score * 0.5
        + structure_score * 0.4,
        60,
    )

    # Market regime modifier
    regime_adjustment = {
        "Bullish": 3,
        "Neutral": 0,
        "Bearish": -3,
        "Unknown": 0,
    }.get(
        market_regime,
        0,
    )

    composite = float(
        np.clip(
            technical_total
            + fund_score
            + regime_adjustment,
            0,
            100,
        )
    )

    # =====================================================
    # SUPPORT / RESISTANCE
    # =====================================================
    recent = df.tail(
        support_window
    )

    support = safe_float(
        recent["Low"].min()
    )

    resistance = safe_float(
        recent["High"].max()
    )

    # =====================================================
    # ATR
    # =====================================================
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

    # =====================================================
    # BUY ZONE
    # =====================================================
    buy_low = max(
        support,
        price - 0.5 * atr_value,
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

    # =====================================================
    # STOP LOSS
    # =====================================================
    stop_loss = max(
        support
        - 0.25 * atr_value,

        price
        - atr_mult * atr_value,
    )

    if stop_loss >= price:

        stop_loss = (
            price
            - atr_mult * atr_value
        )

    # =====================================================
    # TAKE PROFIT
    # =====================================================
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

    # =====================================================
    # SETUP CLASSIFICATION
    # =====================================================
    if breakout:

        setup = "Breakout"

    elif (
        price
        > safe_float(
            latest["SMA_50"]
        )
        and latest["MACD"]
        > latest["MACD_Signal"]
    ):

        setup = "Trend Continuation"

    elif (
        price
        >= safe_float(
            latest["SMA_50"]
        ) * 0.97
    ):

        setup = "Pullback Watch"

    else:

        setup = "Wait / Weak Setup"

    # =====================================================
    # RISK LABEL
    # =====================================================
    if risk_pct <= 4:

        risk_label = "Low"

    elif risk_pct <= 8:

        risk_label = "Medium"

    else:

        risk_label = "High"

    # =====================================================
    # DATA QUALITY
    # =====================================================
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

    # =====================================================
    # RETURN
    # =====================================================
    return {

        "Ticker": ticker,

        "Name": fundamentals[
            "longName"
        ],

        "Sector": fundamentals[
            "sector"
        ],

        "Industry": fundamentals[
            "industry"
        ],

        "Price": price,

        "Score": round(
            composite,
            1,
        ),

        "TechnicalScore": round(
            technical_total,
            1,
        ),

        "FundamentalScore": round(
            fund_score,
            1,
        ),

        "MomentumScore": round(
            momentum_score,
            1,
        ),

        "StructureScore": round(
            structure_score,
            1,
        ),

        "PER": per,

        "PBV": pbv,

        "ROE": roe_percent,

        "DividendYield": dividend_yield,

        "RSI": safe_float(
            latest["RSI"]
        ),

        "SMA50": safe_float(
            latest["SMA_50"]
        ),

        "SMA200": safe_float(
            latest["SMA_200"]
        ),

        "MACD": safe_float(
            latest["MACD"]
        ),

        "MACDSignal": safe_float(
            latest["MACD_Signal"]
        ),

        "VolumeRatio": volume_ratio,

        "ATR": atr_value,

        "StockReturn60D":
            stock_return_60,

        "IHSGReturn60D":
            ihsg_return_60,

        "RelativeStrength":
            relative_strength,

        "Support": support,

        "Resistance": resistance,

        "BuyLow": buy_low,

        "BuyHigh": buy_high,

        "StopLoss": stop_loss,

        "TP1": tp1,

        "TP2": tp2,

        "RRTP1": rr_tp1,

        "RRTP2": rr_tp2,

        "RiskPct": risk_pct,

        "RiskLabel":
            risk_label,

        "Setup": setup,

        "Breakout":
            breakout,

        "DataQuality":
            data_quality,

        "Signals":
            signals,

        "DF": df,
    }


# =========================================================
# SCREENING ENGINE
# =========================================================
def run_screening(
    tickers,
    minimum_score,
    support_window,
    breakout_window,
    atr_mult,
):

    market = (
        get_market_regime()
    )

    market_df = market["df"]

    market_regime = (
        market["regime"]
    )

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

            result = analyze_stock(
                ticker=ticker,
                market_df=market_df,
                market_regime=market_regime,
                support_window=support_window,
                breakout_window=breakout_window,
                atr_mult=atr_mult,
            )

            if result is None:

                errors.append(
                    f"{ticker}: "
                    "data tidak cukup / tidak tersedia"
                )

            elif (
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

    results.sort(
        key=lambda x: x["Score"],
        reverse=True,
    )

    return (
        results,
        errors,
        market,
    )


# =========================================================
# SUMMARY DATAFRAME
# =========================================================
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
                            stock[
                                "VolumeRatio"
                            ],
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

    return pd.DataFrame(rows)


# =========================================================
# CSV EXPORT
# =========================================================
def dataframe_to_csv(
    dataframe
):

    buffer = io.StringIO()

    dataframe.to_csv(
        buffer,
        index=False,
    )

    return buffer.getvalue().encode(
        "utf-8"
    )


# =========================================================
# AI PROMPT
# =========================================================
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
        else f"{stock['PER']:.2f}x"
    )

    pbv_text = (
        "N/A"
        if np.isnan(
            stock["PBV"]
        )
        else f"{stock['PBV']:.2f}x"
    )

    roe_text = (
        "N/A"
        if np.isnan(
            stock["ROE"]
        )
        else f"{stock['ROE']:.2f}%"
    )

    volume_text = (
        "N/A"
        if np.isnan(
            stock["VolumeRatio"]
        )
        else f"{stock['VolumeRatio']:.2f}x"
    )

    return f"""
[SYSTEM INSTRUCTION:
EXPERT HYBRID EQUITY RESEARCH ANALYST - IDX
]

Anda adalah analis pasar modal Indonesia yang melakukan
research objektif berbasis data.

Jangan mengarang data yang tidak tersedia.

Jelaskan keterbatasan data jika diperlukan.

=========================================================
INSTRUMEN
=========================================================

Ticker:
{stock["Ticker"]}

Nama:
{stock["Name"]}

Sektor:
{stock["Sector"]}

Industri:
{stock["Industry"]}

Harga terakhir:
{format_rupiah(stock["Price"])}


=========================================================
FUNDAMENTAL
=========================================================

PER:
{per_text}

PBV:
{pbv_text}

ROE:
{roe_text}

Dividend Yield:
{stock["DividendYield"]:.2f}%


=========================================================
TEKNIKAL
=========================================================

RSI 14:
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

Return 60D:
{
    "N/A"
    if np.isnan(stock["StockReturn60D"])
    else f"{stock['StockReturn60D']:.2f}%"
}

Relative Strength vs IHSG:
{
    "N/A"
    if np.isnan(stock["RelativeStrength"])
    else f"{stock['RelativeStrength']:.2f}%"
}


=========================================================
MARKET REGIME
=========================================================

IHSG Regime:
{market["regime"]}

IHSG Regime Score:
{market["score"]:.1f}/100


=========================================================
PRICE STRUCTURE
=========================================================

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


=========================================================
RISK / REWARD ENGINE
=========================================================

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


=========================================================
COMPOSITE SCORE
=========================================================

Technical:
{stock["TechnicalScore"]}/60

Fundamental:
{stock["FundamentalScore"]}/40

Composite:
{stock["Score"]}/100


=========================================================
HORIZON
=========================================================

Research horizon:
{horizon_days} hari bursa.


=========================================================
TUGAS AI
=========================================================

1. Jelaskan trend dan momentum.

2. Evaluasi fundamental:
   PER
   PBV
   ROE
   Dividend Yield

3. Pertimbangkan karakteristik sektor ketika
   menginterpretasikan valuasi.

4. Jelaskan apakah setup lebih dekat ke:
   - Breakout
   - Trend Continuation
   - Pullback Watch
   - Wait / Weak Setup

5. Evaluasi:
   - Support
   - Resistance
   - Buy Zone
   - Stop Loss
   - TP1
   - TP2
   - Risk/Reward

6. Buat tiga skenario:

   BULL
   Trigger
   Target / scenario

   BASE
   Trigger
   Target / scenario

   BEAR
   Trigger
   Invalidation

7. Jelaskan risiko utama.

8. Sebutkan data tambahan yang harus diverifikasi
   sebelum keputusan dibuat.

9. Jangan memberikan kepastian keuntungan.

10. Jangan menyamakan composite score dengan
    rekomendasi investasi personal.

Tampilkan dalam Markdown profesional.
"""


# =========================================================
# SESSION STATE
# =========================================================
if "results" not in st.session_state:

    st.session_state[
        "results"
    ] = []


if "errors" not in st.session_state:

    st.session_state[
        "errors"
    ] = []


if "market" not in st.session_state:

    st.session_state[
        "market"
    ] = {
        "regime": "Unknown",
        "score": 50,
        "df": pd.DataFrame(),
    }


# =========================================================
# RUN BUTTON
# =========================================================
run_clicked = st.button(
    "🚀 Jalankan Hybrid Screening",
    type="primary",
    use_container_width=True,
)


if run_clicked:

    if not selected_tickers:

        st.warning(
            "Pilih minimal satu ticker IDX terlebih dahulu."
        )

    else:

        with st.spinner(
            "Mengunduh data dan menjalankan screening..."
        ):

            (
                results,
                errors,
                market,
            ) = run_screening(
                tickers=selected_tickers,
                minimum_score=min_score,
                support_window=support_lookback,
                breakout_window=breakout_lookback,
                atr_mult=atr_multiplier,
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


# =========================================================
# LOAD STATE
# =========================================================
results = st.session_state[
    "results"
]

errors = st.session_state[
    "errors"
]

market = st.session_state[
    "market"
]


# =========================================================
# MARKET HEADER
# =========================================================
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Market Regime",
    market.get(
        "regime",
        "Unknown",
    ),
)

m2.metric(
    "IHSG Regime Score",
    f"{market.get('score', 50):.0f}/100",
)

m3.metric(
    "Stocks Passed",
    len(results),
)

m4.metric(
    "Minimum Score",
    min_score,
)


# =========================================================
# EMPTY STATE
# =========================================================
if not results:

    st.info(
        "Klik **Jalankan Hybrid Screening** "
        "untuk memulai."
    )

    st.stop()


# =========================================================
# SUCCESS
# =========================================================
st.success(
    f"Ditemukan {len(results)} saham "
    f"dengan score ≥ {min_score}."
)


# =========================================================
# ERRORS
# =========================================================
if errors:

    with st.expander(
        f"⚠️ Data issues ({len(errors)})"
    ):

        for error in errors:

            st.write(
                f"- {error}"
            )


# =========================================================
# RANKING
# =========================================================
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
    column_config={

        "Score":
            st.column_config.NumberColumn(
                "Score",
                format="%.1f",
            ),

        "Technical":
            st.column_config.NumberColumn(
                "Technical",
                format="%.1f",
            ),

        "Fundamental":
            st.column_config.NumberColumn(
                "Fundamental",
                format="%.1f",
            ),

        "Momentum":
            st.column_config.NumberColumn(
                "Momentum",
                format="%.1f",
            ),

        "Price":
            st.column_config.NumberColumn(
                "Price",
                format="%.2f",
            ),
    },
)


# =========================================================
# EXPORT
# =========================================================
st.download_button(
    "⬇️ Download Screening CSV",

    data=dataframe_to_csv(
        summary_df
    ),

    file_name=(
        f"idx_screener_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    ),

    mime="text/csv",
)


# =========================================================
# DETAIL
# =========================================================
st.divider()

st.subheader(
    "🔎 Analisis Detail"
)

selected_stock = st.selectbox(
    "Pilih saham",

    options=[
        result["Ticker"]
        for result in results
    ],
)


stock = next(
    result
    for result in results
    if result["Ticker"]
    == selected_stock
)


# =========================================================
# TOP METRICS
# =========================================================
d1, d2, d3, d4, d5 = st.columns(5)

d1.metric(
    "Harga",
    format_rupiah(
        stock["Price"]
    ),
)

d2.metric(
    "Score",
    f"{stock['Score']:.1f}/100",
)

d3.metric(
    "Setup",
    stock["Setup"],
)

d4.metric(
    "Risk",
    stock["RiskLabel"],
)

d5.metric(
    "Relative Strength",
    (
        "N/A"
        if np.isnan(
            stock[
                "RelativeStrength"
            ]
        )
        else
        f"{stock['RelativeStrength']:.2f}%"
    ),
)


# =========================================================
# FUNDAMENTAL
# =========================================================
st.markdown(
    "### 💰 Fundamental"
)

f1, f2, f3, f4, f5 = st.columns(5)

f1.metric(
    "PER",
    (
        "N/A"
        if np.isnan(
            stock["PER"]
        )
        else
        f"{stock['PER']:.2f}x"
    ),
)

f2.metric(
    "PBV",
    (
        "N/A"
        if np.isnan(
            stock["PBV"]
        )
        else
        f"{stock['PBV']:.2f}x"
    ),
)

f3.metric(
    "ROE",
    (
        "N/A"
        if np.isnan(
            stock["ROE"]
        )
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
    stock["Sector"][:20],
)


# =========================================================
# TECHNICAL
# =========================================================
st.markdown(
    "### 📈 Technical Dashboard"
)

t1, t2, t3, t4, t5, t6 = st.columns(6)

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
    "Volume x",
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


# =========================================================
# CANDLESTICK CHART
# =========================================================
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

        name="OHLC",
    )
)


fig.add_trace(
    go.Scatter(

        x=chart_df.index,

        y=chart_df[
            "SMA_20"
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
            "SMA_50"
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
            "SMA_200"
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

    annotation_text="Support",

    line_dash="dot",
)


# Resistance
fig.add_hline(
    y=stock[
        "Resistance"
    ],

    annotation_text="Resistance",

    line_dash="dot",
)


# Stop loss
fig.add_hline(
    y=stock[
        "StopLoss"
    ],

    annotation_text="Engine SL",

    line_dash="dash",
)


# TP1
fig.add_hline(
    y=stock[
        "TP1"
    ],

    annotation_text="TP1",

    line_dash="dash",
)


# TP2
fig.add_hline(
    y=stock[
        "TP2"
    ],

    annotation_text="TP2",

    line_dash="dash",
)


fig.update_layout(

    title=(
        f"Candlestick & Trend — "
        f"{selected_stock}"
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


# =========================================================
# RSI CHART
# =========================================================
rsi_df = (
    stock["DF"]
    .tail(180)
)

rsi_fig = go.Figure()


rsi_fig.add_trace(
    go.Scatter(

        x=rsi_df.index,

        y=rsi_df[
            "RSI"
        ],

        name="RSI 14",
    )
)


rsi_fig.add_hline(
    y=70,
    line_dash="dot",
    annotation_text="RSI 70",
)


rsi_fig.add_hline(
    y=30,
    line_dash="dot",
    annotation_text="RSI 30",
)


rsi_fig.update_layout(

    title="RSI 14",

    template="plotly_white",

    height=300,

    yaxis_title="RSI",
)


st.plotly_chart(
    rsi_fig,
    use_container_width=True,
)


# =========================================================
# SIGNAL + RISK REWARD
# =========================================================
left, right = st.columns(
    [1, 1]
)


with left:

    st.markdown(
        "### 🚦 Signals"
    )

    for signal in stock[
        "Signals"
    ][:12]:

        st.write(
            f"• {signal}"
        )

    st.caption(
        f"Data quality: "
        f"{stock['DataQuality']}"
    )


with right:

    st.markdown(
        "### 🎯 Risk / Reward Engine"
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
        f"Risk to SL: "
        f"{stock['RiskPct']:.2f}% "
        f"| R:R TP1 1:{stock['RRTP1']:.2f} "
        f"| R:R TP2 1:{stock['RRTP2']:.2f}"
    )


# =========================================================
# AI PROMPT
# =========================================================
st.divider()

st.subheader(
    "🤖 Master Prompt AI"
)

st.caption(
    "Prompt memakai output engine analitik. "
    "AI menjadi layer interpretasi, bukan sumber data mentah."
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


# =========================================================
# DISCLAIMER
# =========================================================
st.divider()

st.caption(
    "Disclaimer: aplikasi ini adalah alat screening/research. "
    "Data provider eksternal dapat terlambat, tidak lengkap, atau berubah. "
    "Score dan level risk/reward bukan jaminan hasil dan bukan nasihat investasi personal."
)