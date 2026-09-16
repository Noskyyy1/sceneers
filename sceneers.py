import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# =====================================================================
# 1. MODUL: DATA ENGINE (SCREENER & INSTITUTIONAL TRACKER)
# =====================================================================
class IDXMarketScanner:
    """Mesin pengumpul data pasar untuk memindai banyak saham sekaligus."""
    def __init__(self, tickers_list):
        # Format ticker agar sesuai standar Yahoo Finance Indonesia (.JK)
        self.tickers = [t if t.endswith('.JK') else f"{t.upper()}.JK" for t in tickers_list]

    def scan_stock(self, ticker_symbol) -> dict:
        """Memindai satu saham untuk aspek Teknikal, Fundamental & Smart Money."""
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            # Ambil data 6 bulan terakhir untuk melihat tren jangka pendek-menengah
            df = ticker_obj.history(period="6m")
            
            if df.empty or len(df) < 50:
                return None
            
            # --- Perhitungan Teknikal & Smart Money Flow ---
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA50'] = ta.sma(df['Close'], length=50)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['Vol_Avg20'] = ta.sma(df['Volume'], length=20)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Indikator Smart Money Sederhana (Volume & Price Action)
            # Jika harga naik tinggi ditemani volume > rata-rata, tandanya ada akumulasi besar
            smart_money_accum = latest['Volume'] > (latest['Vol_Avg20'] * 1.5) and latest['Close'] > prev['Close']
            
            # --- Mengambil Jejak Pemegang Saham Kakap (>5% / Venture / Institutional) ---
            # Menggunakan properti institutional_holders atau major_holders dari yfinance
            try:
                major_holders = ticker_obj.major_holders
                # Alternatif jika institutional_holders tersedia
                inst_holders = ticker_obj.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    top_investors = ", ".join(inst_holders['Holder'].head(3).tolist())
                else:
                    top_investors = "Institusi Domestik / Ritel Besar"
            except:
                top_investors = "Data Pemegang Saham Tersembunyi/Ritel"
                
            info = ticker_obj.info
            
            return {
                "Ticker": ticker_symbol.replace(".JK", ""),
                "Nama Perusahaan": info.get("longName", "N/A"),
                "Harga": latest['Close'],
                "RSI": latest['RSI'],
                "MA20": latest['MA20'],
                "MA50": latest['MA50'],
                "Smart Money Signal": "🔥 AKUMULASI BESAR" if smart_money_accum else "Neutral / Distribusi",
                "Top Institutional/VC Holders": top_investors,
                "Market Cap (T)": info.get("marketCap", 0) / 1e12,
                "Raw_Data": df # Disimpan untuk visualisasi grafik nanti
            }
        except Exception:
            return None

# =====================================================================
# 2. MODUL: LOGIKA SELEKSI INVESTASI (2-3 BULAN)
# =====================================================================
class InvestmentScreener:
    """Menyaring saham-saham hasil scan yang paling layak invest 2-3 bulan ke depan."""
    @staticmethod
    def filter_candidates(scanned_results: list) -> pd.DataFrame:
        candidates = []
        for stock in scanned_results:
            if stock is None:
                continue
            
            # Kriteria Layak Investasi Jangka Pendek:
            # 1. Sedang Uptrend (Harga di atas MA50)
            # 2. Momentum Kuat (RSI antara 45 hingga 65, tidak overbought)
            # 3. Likuiditas Bagus (Market Cap > 1 Triliun Rupiah)
            is_uptrend = stock["Harga"] > stock["MA50"]
            healthy_momentum = 45 <= stock["RSI"] <= 68
            is_liquid = stock["Market Cap (T)"] > 1.0 
            
            status = "❌ Kurang Ideal"
            score = 0
            
            if is_uptrend and healthy_momentum and is_liquid:
                status = "🎯 LAYAK INVESTASI (2-3 Bulan)"
                score += 2
            elif is_uptrend and is_liquid:
                status = "🗒️ Pantau (Tunggu Koreksi)"
                score += 1
                
            if stock["Smart Money Signal"] == "🔥 AKUMULASI BESAR":
                score += 1

            candidates.append({
                "Ticker": stock["Ticker"],
                "Nama Perusahaan": stock["Nama Perusahaan"],
                "Harga Terakhir": f"Rp {stock['Harga']:,.0f}",
                "RSI": f"{stock['RSI']:.1f}",
                "Smart Money": stock["Smart Money Signal"],
                "Jejak Inst/VC Utama": stock["Top Institutional/VC Holders"],
                "Status Kelayakan": status,
                "Score": score,
                "Raw_Data": stock["Raw_Data"] # untuk grafik
            })
            
        df_res = pd.DataFrame(candidates)
        if not df_res.empty:
            return df_res.sort_values(by="Score", ascending=False)
        return df_res

# =====================================================================
# 3. INTERFACES / DASHBOARD
# =====================================================================
def main():
    st.set_page_config(page_title="IDX Smart Money Screener", layout="wide")
    st.title("🔎 IDX Smart Money & Institutional Investment Screener")
    st.caption("Aplikasi pemindai saham kelayakan investasi 2-3 bulan berdasarkan data tren dan jejak kepemilikan modal besar.")
    st.markdown("---")
    
    # Kumpulan Saham Pilihan untuk di-scan (Bisa Anda tambah sesuai keinginan)
    default_watchlists = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "UNVR", "GOTO", "AMMN", "BBNI", "ADRO", "PTBA", "ANTM"]
    
    st.sidebar.header("⚙️ Menu Screener")
    selected_stocks = st.sidebar.text_area(
        "Daftar Kode Saham yang Di-scan (Pisahkan dengan koma):", 
        value=", ".join(default_watchlists)
    )
    
    tickers_to_scan = [t.strip() for t in selected_stocks.split(",")]
    
    if st.sidebar.button("🚀 Mulai Pemindaian Pasar"):
        scanner = IDXMarketScanner(tickers_to_scan)
        
        with st.spinner(f"Memindai {len(tickers_to_scan)} saham di Bursa Efek Indonesia..."):
            raw_results = [scanner.scan_stock(t) for t in scanner.tickers]
            summary_df = InvestmentScreener.filter_candidates(raw_results)
            
            if summary_df.empty:
                st.warning("Tidak ada saham yang memenuhi kriteria pemindaian.")
                return
            
            # Tampilkan Tabel Hasil Analisis
            st.subheader("📋 Tabel Hasil Screener & Ranking Kelayakan")
            # Sembunyikan kolom objek grafik 'Raw_Data' dari tabel publik
            display_df = summary_df.drop(columns=['Raw_Data'])
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Tampilkan Grafik untuk Saham Peringkat Teratas (Skor Tertinggi)
            st.markdown("---")
            top_stock = summary_df.iloc[0]
            st.subheader(f"📈 Grafik Tren Saham Potensial Teratas: {top_stock['Ticker']} ({top_stock['Status Kelayakan']})")
            
            df_chart = top_stock['Raw_Data']
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                low=df_chart['Low'], close=df_chart['Close'], name="Harga"
            ))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], name='MA 50 (Batas Tren)', line=dict(color='blue', width=2)))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
