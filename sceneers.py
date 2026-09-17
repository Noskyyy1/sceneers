import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="IDX Swing & Fundamental Screener", layout="wide")

st.title("📈 IDX Stock Screener: Teknikal & Fundamental (2-3 Bulan Potential)")
st.caption("Screening gabungan Trend/Momentum Teknikal + Valuasi Fundamental (PER, PBV, ROE) untuk Saham IDX.")

# --- SIDEBAR: INPUT & PARAMETER ---
st.sidebar.header("⚙️ Parameter Screening")

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP", 
    "AMRT", "ADRO", "PTBA", "ITMG", "PGAS", "ANTM", "INCO", "CPIN",
    "BRIS", "MEDC", "MDKA", "AKRA", "AMMN", "BREN", "TPIA", "GOTO"
]

selected_tickers = st.sidebar.multiselect(
    "Pilih Daftar Saham IDX:",
    options=DEFAULT_TICKERS,
    default=DEFAULT_TICKERS[:12]
)

min_score = st.sidebar.slider("Minimal Skor Gabungan (0 - 100):", 0, 100, 55)

# --- FUNGSI AMBIL DATA & ANALISIS ---
def analyze_stock(ticker_code):
    symbol = f"{ticker_code}.JK"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 1. Fetch Historical Data (Teknikal)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty or len(df) < 100:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Fetch Fundamental Data (Info)
    tk = yf.Ticker(symbol)
    info = tk.info if tk else {}

    # Ekstraksi Parameter Fundamental
    per = info.get('trailingPE', None)
    pbv = info.get('priceToBook', None)
    roe = info.get('returnOnEquity', None) # Nilai berupa desimal (misal 0.15 = 15%)
    dividend_yield = info.get('dividendYield', 0) or 0
    market_cap = info.get('marketCap', 0)

    roe_percent = round(roe * 100, 2) if roe is not None else None
    div_percent = round(dividend_yield * 100, 2) if dividend_yield else 0.0

    # --- INDIKATOR TEKNIKAL ---
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['Vol_SMA20'] = ta.sma(df['Volume'], length=20)

    latest = df.iloc[-1]

    # --- ALGORITMA SCORING (MAX 100) ---
    tech_score = 0
    fund_score = 0
    signals = []

    # A. Skor Teknikal (Bobot Max 60 Poin)
    if latest['Close'] > latest['SMA_50']:
        tech_score += 15
        signals.append("Harga di atas SMA 50 (Uptrend Jangka Menengah)")
    if latest['Close'] > latest['SMA_200']:
        tech_score += 10
        signals.append("Harga di atas SMA 200 (Major Uptrend)")
    if latest['MACD'] > latest['MACD_Signal']:
        tech_score += 15
        signals.append("MACD Bullish Crossover")
    if 40 <= latest['RSI'] <= 65:
        tech_score += 10
        signals.append(f"RSI Ideal ({latest['RSI']:.1f})")
    if latest['Volume'] > (1.2 * latest['Vol_SMA20']):
        tech_score += 10
        signals.append("Volume Spurt / Akumulasi")

    # B. Skor Fundamental (Bobot Max 40 Poin)
    # 1. Profitabilitas (ROE > 10% Sangat Baik)
    if roe_percent is not None:
        if roe_percent >= 15:
            fund_score += 15
            signals.append(f"ROE Sangat Tinggi ({roe_percent}%)")
        elif roe_percent >= 10:
            fund_score += 10
            signals.append(f"ROE Sehat ({roe_percent}%)")

    # 2. Valuasi PER & PBV
    if per is not None and 0 < per <= 20:
        fund_score += 15
        signals.append(f"PER Wajar/Murah ({round(per, 2)}x)")
    if pbv is not None and 0 < pbv <= 3.0:
        fund_score += 10
        signals.append(f"PBV Terjangkau ({round(pbv, 2)}x)")

    total_score = tech_score + fund_score

    return {
        "Ticker": ticker_code,
        "Price": int(latest['Close']),
        "Score": total_score,
        "TechScore": tech_score,
        "FundScore": fund_score,
        "PER": round(per, 2) if per else "N/A",
        "PBV": round(pbv, 2) if pbv else "N/A",
        "ROE": f"{roe_percent}%" if roe_percent is not None else "N/A",
        "DivYield": f"{div_percent}%",
        "RSI": round(latest['RSI'], 2),
        "SMA_50": round(latest['SMA_50'], 2),
        "SMA_200": round(latest['SMA_200'], 2),
        "Signals": signals,
        "DF": df
    }

# --- PROCESS SCREENING ---
if st.button("🚀 Jalankan Screening Hybrid (Teknikal + Fundamental)"):
    results = []
    
    with st.spinner("Mengunduh & menganalisis data teknikal & laporan keuangan..."):
        for t in selected_tickers:
            res = analyze_stock(t)
            if res and res['Score'] >= min_score:
                results.append(res)
    
    st.session_state['results'] = results

# --- DISPLAY RESULTS ---
if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']
    st.success(f"Ditemukan **{len(results)}** saham yang memenuhi kriteria minimum skor {min_score}!")

    # Display Tabel Ringkasan dengan Metrik Fundamental
    summary_data = []
    for r in results:
        summary_data.append({
            "Kode": r['Ticker'],
            "Harga Terakhir": f"Rp {r['Price']:,}",
            "Total Skor": f"{r['Score']}%",
            "PER": r['PER'],
            "PBV": r['PBV'],
            "ROE": r['ROE'],
            "Dividend Yield": r['DivYield'],
            "RSI (14)": r['RSI'],
            "Highlights Sinyal": ", ".join(r['Signals'][:3])
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # Analisis Detail
    st.subheader("📊 Analisis Detail & Auto AI Prompt Generator")
    
    selected_stock = st.selectbox("Pilih Saham untuk Analisis Detail & Generate Prompt:", [r['Ticker'] for r in results])
    stock_detail = next(item for item in results if item["Ticker"] == selected_stock)
    
    # Metrik Utama Tampilan Khas Dashboard
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Harga Terakhir", f"Rp {stock_detail['Price']:,}")
    c2.metric("PER (Price/Earnings)", f"{stock_detail['PER']}")
    c3.metric("PBV (Price/Book)", f"{stock_detail['PBV']}")
    c4.metric("ROE (Return on Equity)", f"{stock_detail['ROE']}")
    c5.metric("Div Yield", f"{stock_detail['DivYield']}")

    # Chart Candlestick
    df_chart = stock_detail['DF'].tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'], high=df_chart['High'],
        low=df_chart['Low'], close=df_chart['Close'],
        name="OHLC"
    ))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], name="SMA 50", line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200'], name="SMA 200", line=dict(color='blue', width=1.5)))
    fig.update_layout(title=f"Chart Candlestick 6 Bulan - {selected_stock}", xaxis_rangeslider_visible=False, template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)

    # --- AUTO GENERATED PROMPT AI WITH FUNDAMENTAL DATA ---
    st.subheader("🤖 Master Prompt AI (Memuat Data Teknikal & Fundamental)")
    
    ai_prompt = f"""
[SYSTEM INSTRUCTION: EXPERT HYBRID EQUITY ANALYST (IDX)]

Anda adalah Analis Saham Senior Pasar Modal Indonesia (IDX) yang berpengalaman menggabungkan Analisis Fundamental & Teknikal untuk keputusan Swing Trading Jangka Menengah (2-3 bulan).

Berikut adalah data kuantitatif komprehensif terbaru untuk saham {stock_detail['Ticker']}:

--- DATA FUNDAMENTAL ---
- Harga Terakhir: Rp {stock_detail['Price']}
- Valuation (PER): {stock_detail['PER']}x
- Valuation (PBV): {stock_detail['PBV']}x
- Profitabilitas (ROE): {stock_detail['ROE']}
- Dividend Yield: {stock_detail['DivYield']}

--- DATA TEKNIKAL ---
- RSI (14): {stock_detail['RSI']}
- Posisi SMA 50: {stock_detail['SMA_50']}
- Posisi SMA 200: {stock_detail['SMA_200']}
- Sinyal Kunci Terdeteksi: {", ".join(stock_detail['Signals'])}

TUGAS ANDA:
1. **Analisis Valuasi & Kinerja**: Berikan ulasan singkat mengenai kesehatan fundamental saham ini berdasarkan PER, PBV, dan ROE-nya.
2. **Prospek Teknikal (2-3 Bulan)**: Apakah setup teknikal mendukung kenaikan harga dalam kurun waktu 2-3 bulan ke depan?
3. **Rencana Eksekusi Trading**:
   - Area Beli (Buy Zone Range)
   - Target Harga / Take Profit (TP 1 & TP 2)
   - Batas Stop Loss (SL)
4. **Analisis Katalis Sektoral**: Sebutkan 2-3 katalis makro ekonomi/isu sektoral di Indonesia yang bisa menjadi pendorong harga saham ini.

Tampilkan jawaban dalam format Markdown profesional dan terstruktur.
"""
    st.code(ai_prompt, language="markdown")

else:
    st.info("Klik tombol 'Jalankan Screening Hybrid' di atas untuk memulai.")