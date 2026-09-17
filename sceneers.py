import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="IDX Stock Screener", page_icon="📈", layout="wide")

st.title("📈 IDX Stock Screener")
st.caption("Technical + Fundamental screening using Yahoo Finance data.")

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "TLKM", "ASII", "UNVR",
    "ICBP", "AMRT", "ADRO", "PTBA", "ITMG", "PGAS", "ANTM", "INCO",
    "CPIN", "MEDC", "MDKA", "AKRA", "AMMN", "BREN", "TPIA", "GOTO"
]


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def normalize_columns(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    return df.dropna(subset=["Close"]).copy()


@st.cache_data(ttl=900, show_spinner=False)
def get_price_data(ticker):
    try:
        end = datetime.now()
        start = end - timedelta(days=420)
        df = yf.download(
            f"{ticker}.JK",
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        df = normalize_columns(df)
        return df if len(df) >= 100 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_fundamental(ticker):
    data = {
        "name": ticker,
        "sector": "N/A",
        "industry": "N/A",
        "per": None,
        "pbv": None,
        "roe": None,
        "dividend": 0.0,
    }
    try:
        info = yf.Ticker(f"{ticker}.JK").info
        data["name"] = info.get("longName") or info.get("shortName") or ticker
        data["sector"] = info.get("sector") or "N/A"
        data["industry"] = info.get("industry") or "N/A"
        data["per"] = info.get("trailingPE")
        data["pbv"] = info.get("priceToBook")
        data["roe"] = info.get("returnOnEquity")
        data["dividend"] = info.get("dividendYield") or 0.0
    except Exception:
        pass
    return data


def analyze(ticker):
    ticker = ticker.strip().upper().replace(".JK", "")
    df = get_price_data(ticker)
    if df.empty:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = rsi(df["Close"])
    df["VolumeMA20"] = df["Volume"].rolling(20).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    last = df.iloc[-1]
    fund = get_fundamental(ticker)

    close = float(last["Close"])
    score_tech = 0
    score_fund = 0
    signals = []

    if pd.notna(last["SMA50"]) and close > float(last["SMA50"]):
        score_tech += 15
        signals.append("Harga di atas SMA 50")
    if pd.notna(last["SMA200"]) and close > float(last["SMA200"]):
        score_tech += 10
        signals.append("Harga di atas SMA 200")
    if pd.notna(last["MACD"]) and pd.notna(last["MACDSignal"]) and last["MACD"] > last["MACDSignal"]:
        score_tech += 15
        signals.append("MACD bullish")
    if pd.notna(last["RSI"]) and 40 <= float(last["RSI"]) <= 65:
        score_tech += 10
        signals.append(f"RSI sehat ({float(last['RSI']):.1f})")
    if pd.notna(last["VolumeMA20"]) and float(last["Volume"]) > 1.2 * float(last["VolumeMA20"]):
        score_tech += 10
        signals.append("Volume > 1.2x rata-rata 20 hari")

    roe_pct = fund["roe"] * 100 if fund["roe"] is not None else None
    if roe_pct is not None:
        if roe_pct >= 15:
            score_fund += 15
            signals.append(f"ROE tinggi ({roe_pct:.1f}%)")
        elif roe_pct >= 10:
            score_fund += 10
            signals.append(f"ROE sehat ({roe_pct:.1f}%)")

    if fund["per"] is not None and 0 < float(fund["per"]) <= 20:
        score_fund += 15
        signals.append(f"PER <= 20x ({float(fund['per']):.2f}x)")
    if fund["pbv"] is not None and 0 < float(fund["pbv"]) <= 3:
        score_fund += 10
        signals.append(f"PBV <= 3x ({float(fund['pbv']):.2f}x)")

    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else close * 0.02
    buy_low = close - 0.50 * atr
    buy_high = close + 0.25 * atr
    tp1 = close + 2 * atr
    tp2 = close + 3.5 * atr
    sl = close - 1.5 * atr

    return {
        "ticker": ticker,
        "name": fund["name"],
        "sector": fund["sector"],
        "industry": fund["industry"],
        "price": close,
        "score": score_tech + score_fund,
        "tech": score_tech,
        "fund": score_fund,
        "per": fund["per"],
        "pbv": fund["pbv"],
        "roe": roe_pct,
        "dividend": fund["dividend"] * 100,
        "rsi": float(last["RSI"]) if pd.notna(last["RSI"]) else None,
        "sma20": float(last["SMA20"]) if pd.notna(last["SMA20"]) else None,
        "sma50": float(last["SMA50"]) if pd.notna(last["SMA50"]) else None,
        "sma200": float(last["SMA200"]) if pd.notna(last["SMA200"]) else None,
        "buy_low": buy_low,
        "buy_high": buy_high,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "signals": signals,
        "df": df,
    }


# SIDEBAR
st.sidebar.header("⚙️ Parameter")
mode = st.sidebar.radio("Sumber ticker", ["Daftar bawaan", "Input manual"])

if mode == "Daftar bawaan":
    tickers = st.sidebar.multiselect(
        "Pilih saham",
        DEFAULT_TICKERS,
        default=DEFAULT_TICKERS[:12],
    )
else:
    text = st.sidebar.text_area(
        "Ticker saham",
        "BBCA, BBRI, BMRI, BBNI",
        help="Pisahkan dengan koma atau baris baru. Contoh: ARTO, BUKA, GOTO",
    )
    tickers = [x.strip().upper().replace(".JK", "") for x in text.replace("\n", ",").split(",") if x.strip()]

min_score = st.sidebar.slider("Minimal skor", 0, 100, 55)

if st.button("🚀 Jalankan Screening", type="primary", use_container_width=True):
    if not tickers:
        st.warning("Masukkan minimal satu ticker.")
    else:
        results = []
        failed = []
        progress = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(tickers, 1):
            status.write(f"Menganalisis **{ticker}** ...")
            try:
                result = analyze(ticker)
                if result is None:
                    failed.append(ticker)
                elif result["score"] >= min_score:
                    results.append(result)
            except Exception as e:
                failed.append(f"{ticker}: {str(e)[:60]}")
            progress.progress(i / len(tickers))

        progress.empty()
        status.empty()
        results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state["results"] = results
        st.session_state["failed"] = failed


if "results" in st.session_state:
    results = st.session_state["results"]
    failed = st.session_state.get("failed", [])

    if results:
        st.success(f"Ditemukan {len(results)} saham dengan skor minimal {min_score}.")

        table = []
        for r in results:
            table.append({
                "Kode": r["ticker"],
                "Perusahaan": r["name"],
                "Harga": f"Rp {r['price']:,.0f}",
                "Skor": r["score"],
                "Teknikal": r["tech"],
                "Fundamental": r["fund"],
                "PER": round(float(r["per"]), 2) if r["per"] is not None else "N/A",
                "PBV": round(float(r["pbv"]), 2) if r["pbv"] is not None else "N/A",
                "ROE": f"{r['roe']:.2f}%" if r["roe"] is not None else "N/A",
                "Div Yield": f"{r['dividend']:.2f}%",
                "RSI": round(r["rsi"], 2) if r["rsi"] is not None else "N/A",
                "Sinyal": ", ".join(r["signals"][:3]),
            })
        st.subheader("📋 Hasil Screening")
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        selected = st.selectbox("Pilih saham untuk detail", [r["ticker"] for r in results])
        d = next(r for r in results if r["ticker"] == selected)

        st.subheader(f"📊 {d['ticker']} — {d['name']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Harga", f"Rp {d['price']:,.0f}")
        c2.metric("Skor", f"{d['score']}/100")
        c3.metric("PER", f"{d['per']:.2f}x" if d["per"] is not None else "N/A")
        c4.metric("PBV", f"{d['pbv']:.2f}x" if d["pbv"] is not None else "N/A")
        c5.metric("ROE", f"{d['roe']:.2f}%" if d["roe"] is not None else "N/A")

        chart = d["df"].tail(180)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart.index, open=chart["Open"], high=chart["High"],
            low=chart["Low"], close=chart["Close"], name="OHLC"
        ))
        fig.add_trace(go.Scatter(x=chart.index, y=chart["SMA20"], name="SMA 20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=chart.index, y=chart["SMA50"], name="SMA 50", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=chart.index, y=chart["SMA200"], name="SMA 200", line=dict(width=1.5)))
        fig.update_layout(title=f"{d['ticker']} - 180 Hari", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Area Teknis Berbasis ATR")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Buy Zone", f"Rp {d['buy_low']:,.0f} - Rp {d['buy_high']:,.0f}")
        a2.metric("TP 1", f"Rp {d['tp1']:,.0f}")
        a3.metric("TP 2", f"Rp {d['tp2']:,.0f}")
        a4.metric("Stop Loss", f"Rp {d['sl']:,.0f}")
        st.caption("Area teknis dihitung secara mekanis menggunakan ATR; bukan jaminan harga atau nasihat investasi.")

        st.subheader("🔎 Sinyal")
        if d["signals"]:
            for signal in d["signals"]:
                st.write(f"• {signal}")
        else:
            st.info("Tidak ada sinyal khusus.")

        prompt = f"""
Anda adalah analis saham IDX. Analisis saham {d['ticker']} ({d['name']}) secara objektif.

FUNDAMENTAL
Harga: Rp {d['price']:,.0f}
PER: {d['per'] if d['per'] is not None else 'N/A'}
PBV: {d['pbv'] if d['pbv'] is not None else 'N/A'}
ROE: {f'{d["roe"]:.2f}%' if d['roe'] is not None else 'N/A'}
Dividend Yield: {d['dividend']:.2f}%

TEKNIKAL
RSI: {f'{d["rsi"]:.2f}' if d['rsi'] is not None else 'N/A'}
SMA20: {f'{d["sma20"]:.2f}' if d['sma20'] is not None else 'N/A'}
SMA50: {f'{d["sma50"]:.2f}' if d['sma50'] is not None else 'N/A'}
SMA200: {f'{d["sma200"]:.2f}' if d['sma200'] is not None else 'N/A'}
Sinyal: {', '.join(d['signals']) if d['signals'] else 'Tidak ada'}

AREA TEKNIS
Buy Zone: Rp {d['buy_low']:,.0f} - Rp {d['buy_high']:,.0f}
TP1: Rp {d['tp1']:,.0f}
TP2: Rp {d['tp2']:,.0f}
Stop Loss: Rp {d['sl']:,.0f}

Tugas:
1. Jelaskan fundamental.
2. Jelaskan trend dan momentum teknikal.
3. Jelaskan risiko utama.
4. Bedakan fakta, asumsi, dan interpretasi.
5. Jangan mengarang data yang tidak tersedia.
""".strip()

        st.subheader("🤖 Prompt AI")
        st.code(prompt, language="text")
        st.download_button("⬇️ Download Prompt", prompt, f"prompt_{d['ticker']}.txt", "text/plain")

    else:
        st.warning("Tidak ada saham yang memenuhi skor minimum.")

    if failed:
        st.warning("Ticker gagal dianalisis: " + ", ".join(failed))
else:
    st.info("Pilih ticker lalu klik '🚀 Jalankan Screening'.")

st.markdown("---")
st.caption("Data Yahoo Finance dapat kosong/terlambat. Aplikasi ini adalah alat screening, bukan nasihat investasi.")
