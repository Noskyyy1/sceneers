import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="IDX Auto-Screener Alerts", layout="wide")

class AutoSmartMoneyScanner:
    """Mesin Pemindai Otomatis Tanpa Input Kode Saham."""
    @staticmethod
    def get_tracked_tickers():
        # Daftar otomatis saham likuid IDX (Gabungan LQ45 & Kompas100 teraktif)
        return [
            "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "AMMN", "GOTO", 
            "ADRO", "PTBA", "ANTM", "BRIS", "BRPT", "TPIA", "PGAS", "INKP", 
            "MEDC", "UNVR", "AKRA", "EXCL", "ISAT", "KLBF", "MYOR", "SMGR"
        ]

    def scan_all_markets(self) -> list:
        tickers = self.get_tracked_tickers()
        alerts = []
        
        for t in tickers:
            try:
                ticker_symbol = f"{t}.JK"
                ticker_obj = yf.Ticker(ticker_symbol)
                df = ticker_obj.history(period="3m") # data 3 bulan terakhir
                
                if df.empty or len(df) < 20:
                    continue
                
                # Hitung Rata-rata Volume 20 hari
                df['Vol_Avg20'] = ta.sma(df['Volume'], length=20)
                df['MA50'] = ta.sma(df['Close'], length=50) if len(df) >= 50 else df['Close']
                
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                # UKUR LONJAKAN VOLUME (Deteksi Uang Besar Masuk)
                volume_multiplier = latest['Volume'] / latest['Vol_Avg20'] if latest['Vol_Avg20'] > 0 else 1
                
                # Kriteria Alert: Volume naik > 2x lipat rata-rata & harga naik (Akumulasi Broker Besar)
                if volume_multiplier >= 2.0 and latest['Close'] > prev['Close']:
                    info = ticker_obj.info
                    alerts.append({
                        "Kode": t,
                        "Nama": info.get("longName", "N/A"),
                        "Harga": f"Rp {latest['Close']:,.0f}",
                        "Lonjakan Volume": f"{volume_multiplier:.2f}x Lipat!",
                        "Status Tren": "📈 Uptrend (Di atas MA50)" if latest['Close'] > latest['MA50'] else "📉 Downtrend / Rebound",
                        "Raw_Data": df
                    })
            except:
                continue
        return alerts

# --- TAMPILAN DASHBOARD YANG DIPERCANTIK ---
st.title("🚨 IDX Automated Smart Money Alert System")
st.markdown("Sistem ini memindai saham-saham utama di Bursa Efek Indonesia secara otomatis. Anda tidak perlu memasukkan kode saham lagi.")
st.markdown("---")

# Tombol Pemicu di Layar Utama
if st.button("🔄 Klik untuk Mulai Scan & Cek Notifikasi Uang Besar", use_container_width=True):
    with st.spinner("Sistem sedang menganalisis aktivitas transaksi bursa..."):
        scanner = AutoSmartMoneyScanner()
        detected_alerts = scanner.scan_all_markets()
        
        if detected_alerts:
            # TAMPILAN NOTIFIKASI POP-UP KELAP-KELIP
            st.toast("🔥 WARNING: Ada aktivitas Uang Besar terdeteksi masuk pasar!", icon="⚠️")
            
            st.markdown(f"### 🔔 NOTIFIKASI: Terdeteksi {len(detected_alerts)} Saham Sedang Diakumulasi Broker Besar harian!")
            
            # Tampilkan dalam bentuk kartu yang rapi
            for alert in detected_alerts:
                with st.expander(f"🔴 ALERT NOTIFIKASI: {alert['Kode']} - {alert['Nama']} (Volume Loncat {alert['Lonjakan Volume']})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Harga Terakhir", alert["Harga"])
                    col2.metric("Rasio Uang Masuk", alert["Lonjakan Volume"])
                    col3.metric("Kondisi Tren", alert["Status Tren"])
                    
                    # Tampilkan grafik lilin untuk saham yang memicu notifikasi
                    df_chart = alert["Raw_Data"]
                    fig = go.Figure(data=[go.Candlestick(
                        x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                        low=df_chart['Low'], close=df_chart['Close'], name="Harga"
                    )])
                    fig.update_layout(height=300, xaxis_rangeslider_visible=False, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kondisi Pasar Tenang: Saat ini tidak terdeteksi adanya lonjakan volume transaksi tidak wajar (Smart Money) di saham-saham utama.")
