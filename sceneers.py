import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="IDX Analytics Pro", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp{background:#0b1220}.block-container{max-width:1500px;padding-top:1.2rem}
[data-testid="stSidebar"]{background:#0f172a;border-right:1px solid #263653}
.hero{background:linear-gradient(135deg,#111c31,#0f172a);border:1px solid #263653;border-radius:18px;padding:24px 28px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.2rem}.hero p{color:#94a3b8;margin:6px 0 0}
.card{background:#111c31;border:1px solid #263653;border-radius:15px;padding:16px 18px;min-height:100px}
.label{color:#94a3b8;font-size:.8rem}.value{color:#f8fafc;font-size:1.55rem;font-weight:800;margin-top:6px}
.note{color:#64748b;font-size:.75rem}.signal{background:#111c31;border:1px solid #263653;border-radius:10px;padding:10px;margin:6px 0}
.plan{background:#0f1a2e;border:1px solid #2b3f63;border-radius:12px;padding:12px 14px;margin:8px 0}
.plan b{color:#93c5fd}
</style>
""", unsafe_allow_html=True)

STARTER = sorted(set("""AADI ACES ADCP ADHI ADMR ADRO AGRO AKRA AMMN AMRT ANTM APLN ARCI ARTO ASII ASLC ASSA AUTO AVIA BBCA BBHI BBNI BBRI BBTN BCAP BDMN BELI BFIN BHIT BIKA BIRD BJBR BJTM BKSL BMRI BNGA BREN BRIS BRMS BSDE BTPS BUKA BUMI BUVA CARS CBRE CDIA CENT CFIN CINT CLEO CMRY CPIN CTRA CYBR DEWA DOID DSNG ELSA EMTK ENRG ERAA ESSA EXCL FAST FILM FREN GGRM GIAA GJTL GOTO HEAL HMSP HRUM ICBP IMAS INCO INDF INDY INKP INTP ISAT ITMG JSMR JPFA KBLI KIJA KLBF KPIG KRAS LINK LPKR LSIP MAPA MAPI MARK MBMA MCAS MDKA MEDC MIKA MLPL MNCN MPMX MTEL MYOR NCKL NISP PANI PGAS PGEO PNBN PNLF PPRE PPRO PTBA PTPP PTRO PWON RAJA RALS SCMA SIDO SMBR SMGR SMRA SMSM SRTG SSIA STAA SUPR TAPG TBIG TINS TKIM TLKM TOWR TPIA TRIM TUGU ULTJ UNTR UNVR WIKA WTON ZYRX""".split()))
INDEXES={
"LQ45 (starter)":"ACES ADRO AKRA AMMN AMRT ANTM ASII BBCA BBNI BBRI BBTN BMRI BRIS CPIN EMTK GOTO ICBP INCO INDF INKP ISAT ITMG JSMR KLBF MAPI MBMA MDKA MEDC PGAS PGEO PTBA PTRO SMGR TLKM TOWR TPIA UNTR UNVR".split(),
"IDX30 (starter)":"ADRO AKRA AMMN AMRT ANTM ASII BBCA BBNI BBRI BBTN BMRI BRIS CPIN GOTO ICBP INDF INCO ISAT ITMG KLBF MDKA PGAS PTBA SMGR TLKM TPIA UNTR UNVR".split(),
}

# ────────────────────────────── DATA LAYER ──────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def load_universe():
    p=Path(__file__).parent/"data"/"idx_universe.csv"
    if p.exists():
        try:
            d=pd.read_csv(p); c="Ticker" if "Ticker" in d.columns else d.columns[0]
            x=d[c].dropna().astype(str).str.upper().str.replace(".JK","",regex=False).str.strip()
            if len(x): return sorted(set(x))
        except Exception: pass
    return STARTER

@st.cache_data(ttl=900, show_spinner=False)
def download_batch(tickers, chunk=20):
    """Ambil OHLCV banyak ticker sekaligus. Jauh lebih cepat dari 1 request per ticker."""
    tickers=list(tickers); out={}
    start=datetime.now()-timedelta(days=430); end=datetime.now()
    for i in range(0,len(tickers),chunk):
        part=tickers[i:i+chunk]; syms=[f"{t}.JK" for t in part]
        try:
            raw=yf.download(syms,start=start,end=end,progress=False,auto_adjust=False,
                            threads=True,group_by="ticker")
        except Exception:
            raw=pd.DataFrame()
        if raw is None or raw.empty: continue
        for t,s in zip(part,syms):
            try:
                if isinstance(raw.columns,pd.MultiIndex):
                    lvl0=set(raw.columns.get_level_values(0))
                    if s in lvl0: d=raw[s]
                    elif t in lvl0: d=raw[t]
                    else: continue
                else:
                    d=raw
                d=d.copy()
                if d.dropna(how="all").empty: continue
                out[t]=d
            except Exception: continue
    return out

def add_indicators(raw):
    req=["Open","High","Low","Close","Volume"]
    if raw is None or raw.empty or not all(c in raw.columns for c in req): return pd.DataFrame()
    d=raw.dropna(subset=["Close"]).copy()
    for c in req: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["Close"])
    if d.empty: return pd.DataFrame()
    d["SMA20"]=d.Close.rolling(20).mean(); d["SMA50"]=d.Close.rolling(50).mean(); d["SMA200"]=d.Close.rolling(200).mean()
    delta=d.Close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d["RSI"]=100-(100/(1+(ag/al.replace(0,np.nan))))
    e12=d.Close.ewm(span=12,adjust=False).mean(); e26=d.Close.ewm(span=26,adjust=False).mean()
    d["MACD"]=e12-e26; d["MACDSignal"]=d.MACD.ewm(span=9,adjust=False).mean(); d["MACDHist"]=d.MACD-d.MACDSignal
    d["VolSMA20"]=d.Volume.rolling(20).mean()
    d["Value20"]=(d.Close*d.Volume).rolling(20).mean()
    tr=pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
    d["ATR14"]=tr.rolling(14).mean()
    return d

def _fund_raw(t):
    out={"Company":t,"Sector":"N/A","Industry":"N/A","PER":None,"PBV":None,"ROE":None,"Div":0.0,"MarketCap":None}
    try:
        i=yf.Ticker(f"{t}.JK").info
        out.update({"Company":i.get("longName") or i.get("shortName") or t,
                    "Sector":i.get("sector") or "N/A","Industry":i.get("industry") or "N/A",
                    "PER":i.get("trailingPE"),"PBV":i.get("priceToBook"),
                    "ROE":i.get("returnOnEquity"),"Div":i.get("dividendYield") or 0.0,
                    "MarketCap":i.get("marketCap")})
    except Exception: pass
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals_many(tickers, workers=8):
    tickers=list(tickers); out={}
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for t,res in zip(tickers, ex.map(_fund_raw,tickers)): out[t]=res
    except Exception:
        for t in tickers: out[t]=_fund_raw(t)
    return out

# ────────────────────────────── HELPERS ──────────────────────────────

def num(x):
    try:
        x=float(x); return x if np.isfinite(x) else None
    except Exception: return None

def _tick(p):
    if p is None or p<=0: return 1
    if p<200: return 1
    if p<500: return 2
    if p<2000: return 5
    if p<5000: return 10
    return 25

def tick_round(p, mode="nearest"):
    """Bulatkan ke fraksi harga IDX supaya level bisa langsung dipakai di order book."""
    p=num(p)
    if p is None or p<=0: return None
    t=_tick(p)
    if mode=="down": return float(math.floor(p/t)*t)
    if mode=="up": return float(math.ceil(p/t)*t)
    return float(round(p/t)*t)

def rup(x): return "N/A" if x is None else f"Rp {x:,.0f}"

# ────────────────────────────── ANALYSIS ──────────────────────────────

def analyze(ticker, raw, f, cfg):
    d=add_indicators(raw)
    if d.empty or len(d)<100: return None
    q=d.iloc[-1]
    price=num(q.Close)
    if price is None or price<=0: return None
    rsi=num(q.RSI); s20=num(q.SMA20); s50=num(q.SMA50); s200=num(q.SMA200)
    macd=num(q.MACD); ms=num(q.MACDSignal); vol=num(q.Volume); va=num(q.VolSMA20)
    atr=num(q.ATR14); val20=num(q.Value20)
    per=num(f["PER"]); pbv=num(f["PBV"]); roe=num(f["ROE"]); div=num(f["Div"]) or 0.0
    roe_pct=roe*100 if roe is not None else None
    # yfinance kadang kirim 0.045, kadang 4.5 — normalkan
    div_pct=div*100 if div<=1 else div

    ts=fs=0; signals=[]
    if s50 is not None and price>s50: ts+=15; signals.append("Harga > SMA50")
    if s200 is not None and price>s200: ts+=10; signals.append("Harga > SMA200")
    if macd is not None and ms is not None and macd>ms: ts+=15; signals.append("MACD bullish")
    if rsi is not None and 40<=rsi<=65: ts+=10; signals.append(f"RSI sehat ({rsi:.1f})")
    if vol is not None and va is not None and va>0 and vol>1.2*va: ts+=10; signals.append("Volume > 1.2x rata-rata")
    if roe_pct is not None:
        if roe_pct>=15: fs+=15; signals.append(f"ROE {roe_pct:.1f}%")
        elif roe_pct>=10: fs+=10; signals.append(f"ROE {roe_pct:.1f}%")
    if per is not None and 0<per<=20: fs+=15; signals.append(f"PER {per:.1f}x")
    if pbv is not None and 0<pbv<=3: fs+=10; signals.append(f"PBV {pbv:.1f}x")

    if atr is None or atr<=0: atr=price*.03

    # ── Level rencana transaksi ──
    sup20=num(d.Low.tail(20).min()); res60=num(d.High.tail(60).max())
    buy_low=tick_round(price-cfg["buy_k"]*atr,"down")
    buy_high=tick_round(price+cfg["add_k"]*atr,"up")
    sl_raw=price-cfg["sl_k"]*atr
    if sup20 is not None and sup20<price: sl_raw=min(sl_raw, sup20-0.25*atr)
    sl=tick_round(sl_raw,"down")
    tp1=tick_round(price+cfg["tp1_k"]*atr,"down")
    tp2=tick_round(price+cfg["tp2_k"]*atr,"down")
    entry_ref=tick_round(price,"nearest") or price
    risk=entry_ref-sl if sl is not None else None
    rrr=(tp1-entry_ref)/risk if (risk and risk>0 and tp1 is not None) else None
    risk_pct=(risk/entry_ref*100) if (risk and entry_ref) else None

    return {"Ticker":ticker,"Company":f["Company"],"Sector":f["Sector"],"Industry":f["Industry"],
            "Price":price,"Score":ts+fs,"Tech":ts,"Fund":fs,"PER":per,"PBV":pbv,"ROE":roe_pct,"Div":div_pct,
            "RSI":rsi,"SMA20":s20,"SMA50":s50,"SMA200":s200,"MACD":macd,"MACDSignal":ms,"ATR":atr,
            "ATRpct":atr/price*100,"Value20":val20,"Sup20":sup20,"Res60":res60,
            "Entry":entry_ref,"BuyLow":buy_low,"BuyHigh":buy_high,"TP1":tp1,"TP2":tp2,"SL":sl,
            "Risk":risk,"RiskPct":risk_pct,"RRR":rrr,"Signals":signals,"DF":d}

# ────────────────────────────── SIDEBAR ──────────────────────────────

universe_all=load_universe()
st.sidebar.markdown("## ⚙️ Screening Engine")
mode=st.sidebar.selectbox("Universe",["IDX Universe (CSV)","IDX Core (starter)","LQ45 (starter)","IDX30 (starter)","Manual"])
if mode=="IDX Universe (CSV)": universe=universe_all
elif mode=="IDX Core (starter)": universe=STARTER
elif mode in INDEXES: universe=INDEXES[mode]
else:
    raw_in=st.sidebar.text_area("Ticker manual","BBCA, BBRI, BMRI, BBNI",height=100)
    universe=[x.strip().upper().replace(".JK","") for x in raw_in.replace("\n",",").split(",") if x.strip()]
universe=sorted(set(universe))

max_scan=st.sidebar.number_input("Max ticker per scan",1,max(1,len(universe)),
                                 min(100,max(1,len(universe))),10)

st.sidebar.markdown("### 🎚️ Filter")
min_score=st.sidebar.slider("Minimum Hybrid Score",0,100,55,5)
rmin,rmax=st.sidebar.slider("RSI range",0,100,(30,70))
min_rrr=st.sidebar.slider("Minimum Risk/Reward (TP1)",0.0,5.0,0.0,0.25)
min_value=st.sidebar.number_input("Min. nilai transaksi harian (Rp miliar)",0.0,100.0,0.0,0.5,
                                  help="Filter likuiditas: rata-rata Close × Volume 20 hari.")
req50=st.sidebar.checkbox("Wajib > SMA50")
req200=st.sidebar.checkbox("Wajib > SMA200")
reqmacd=st.sidebar.checkbox("Wajib MACD bullish")

st.sidebar.markdown("### 🎯 Rencana Transaksi (ATR)")
cfg={
 "buy_k":st.sidebar.slider("Batas bawah buy zone (× ATR)",0.0,2.0,0.5,0.1),
 "add_k":st.sidebar.slider("Batas atas buy zone (× ATR)",0.0,1.0,0.25,0.05),
 "sl_k": st.sidebar.slider("Stop loss (× ATR)",0.5,4.0,1.5,0.1),
 "tp1_k":st.sidebar.slider("Take profit 1 (× ATR)",0.5,8.0,2.0,0.5),
 "tp2_k":st.sidebar.slider("Take profit 2 (× ATR)",1.0,12.0,3.5,0.5),
}
st.sidebar.markdown("### 💼 Position Sizing")
capital=st.sidebar.number_input("Modal (Rp)",0,10_000_000_000,100_000_000,1_000_000)
risk_pct_cap=st.sidebar.slider("Risiko per transaksi (% modal)",0.1,5.0,1.0,0.1)

if st.sidebar.button("🧹 Clear cache"):
    st.cache_data.clear(); st.rerun()

# ────────────────────────────── MAIN ──────────────────────────────

st.markdown('<div class="hero"><h1>📈 IDX Analytics Pro</h1><p>Hybrid Technical + Fundamental Stock Screener · Streamlit Dashboard</p></div>',unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results=[]
if "failed" not in st.session_state: st.session_state.failed=[]

c1,c2=st.columns([5,1])
c1.caption(f"Universe aktif: {len(universe)} ticker · Scan maksimum: {int(max_scan)}")
run=c2.button("🚀 RUN SCAN",type="primary",use_container_width=True)

if run:
    results=[]; failed=[]; selected=universe[:int(max_scan)]
    bar=st.progress(0); status=st.empty()

    status.write(f"📥 Mengunduh data harga untuk {len(selected)} ticker ...")
    prices=download_batch(tuple(selected))
    bar.progress(0.45)

    status.write("📊 Mengambil data fundamental ...")
    funds=fundamentals_many(tuple(selected))
    bar.progress(0.75)

    status.write("🧮 Menghitung skor dan level transaksi ...")
    for n,t in enumerate(selected,1):
        try:
            raw=prices.get(t)
            if raw is None: failed.append(t); continue
            r=analyze(t,raw,funds.get(t,_fund_raw(t)),cfg)
            if r is None: failed.append(t); continue
            ok=r["Score"]>=min_score
            if req50: ok=ok and r["SMA50"] is not None and r["Price"]>r["SMA50"]
            if req200: ok=ok and r["SMA200"] is not None and r["Price"]>r["SMA200"]
            if reqmacd: ok=ok and r["MACD"] is not None and r["MACDSignal"] is not None and r["MACD"]>r["MACDSignal"]
            if r["RSI"] is not None: ok=ok and rmin<=r["RSI"]<=rmax
            if min_rrr>0: ok=ok and r["RRR"] is not None and r["RRR"]>=min_rrr
            if min_value>0: ok=ok and r["Value20"] is not None and r["Value20"]>=min_value*1e9
            if ok: results.append(r)
        except Exception:
            failed.append(t)
        bar.progress(0.75+0.25*n/len(selected))

    status.empty(); bar.empty()
    results.sort(key=lambda x:x["Score"],reverse=True)
    st.session_state.results=results; st.session_state.failed=failed

results=st.session_state.results; failed=st.session_state.failed

if not results:
    st.info("Atur parameter di sidebar lalu klik **RUN SCAN**.")
    st.markdown("### 🧭 Struktur aplikasi\n**Overview** ringkasan · **Screener** tabel + level buy/exit · **Stock Detail** chart & position sizing · **AI Analysis** prompt.")
else:
    avg=np.mean([r["Score"] for r in results])
    above=sum(r["SMA50"] is not None and r["Price"]>r["SMA50"] for r in results)
    bull=sum(r["MACD"] is not None and r["MACDSignal"] is not None and r["MACD"]>r["MACDSignal"] for r in results)
    rs=[r["RSI"] for r in results if r["RSI"] is not None]
    rr=[r["RRR"] for r in results if r["RRR"] is not None]
    k=st.columns(6)
    k[0].metric("Passed",len(results)); k[1].metric("Avg Score",f"{avg:.1f}/100")
    k[2].metric("> SMA50",above); k[3].metric("MACD Bullish",bull)
    k[4].metric("Avg RSI",f"{np.mean(rs):.1f}" if rs else "N/A")
    k[5].metric("Avg R:R",f"{np.mean(rr):.2f}" if rr else "N/A")

    tabs=st.tabs(["🏠 Overview","🔎 Screener","🎯 Trade Plan","📊 Stock Detail","🤖 AI Analysis"])

    with tabs[0]:
        left,right=st.columns([1.5,1])
        with left:
            dfscore=pd.DataFrame({"Ticker":[r["Ticker"] for r in results],"Score":[r["Score"] for r in results]}).sort_values("Score")
            fig=go.Figure(go.Bar(x=dfscore.Score,y=dfscore.Ticker,orientation="h",text=dfscore.Score,textposition="outside"))
            fig.update_layout(template="plotly_dark",height=max(380,len(results)*26),margin=dict(l=10,r=30,t=20,b=20),xaxis_range=[0,105])
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.markdown("#### Screening Snapshot")
            st.dataframe(pd.DataFrame({"Metric":["Minimum Score","Passed","Technical Avg","Fundamental Avg","Unavailable"],
                                       "Value":[min_score,len(results),round(np.mean([r["Tech"] for r in results]),1),
                                                round(np.mean([r["Fund"] for r in results]),1),len(failed)]}),
                         hide_index=True,use_container_width=True)
            if failed: st.caption("Unavailable: "+", ".join(failed[:40]))

    with tabs[1]:
        rows=[{"Ticker":r["Ticker"],"Company":r["Company"],"Price":r["Price"],"Score":r["Score"],
               "Technical":r["Tech"],"Fundamental":r["Fund"],"RSI":r["RSI"],"PER":r["PER"],"PBV":r["PBV"],
               "ROE %":r["ROE"],"Div Yield %":r["Div"],"Sector":r["Sector"]} for r in results]
        table=pd.DataFrame(rows)
        st.dataframe(table,use_container_width=True,hide_index=True,column_config={
            "Price":st.column_config.NumberColumn(format="Rp %.0f"),
            "Score":st.column_config.ProgressColumn(min_value=0,max_value=100),
            "Technical":st.column_config.NumberColumn(format="%d/60"),
            "Fundamental":st.column_config.NumberColumn(format="%d/40"),
            "RSI":st.column_config.NumberColumn(format="%.2f"),
            "PER":st.column_config.NumberColumn(format="%.2fx"),
            "PBV":st.column_config.NumberColumn(format="%.2fx"),
            "ROE %":st.column_config.NumberColumn(format="%.2f%%"),
            "Div Yield %":st.column_config.NumberColumn(format="%.2f%%")})
        st.download_button("⬇️ Export CSV",table.to_csv(index=False).encode(),"idx_screening_results.csv","text/csv")

    with tabs[2]:
        st.markdown("#### 🎯 Level beli & keluar (berbasis ATR, dibulatkan ke fraksi harga IDX)")
        plan_rows=[]
        for r in results:
            lots=None
            if r["Risk"] and r["Risk"]>0:
                shares=(capital*risk_pct_cap/100)/r["Risk"]
                lots=int(shares//100)
            plan_rows.append({"Ticker":r["Ticker"],"Last":r["Price"],
                              "Buy Zone ⬇":r["BuyLow"],"Buy Zone ⬆":r["BuyHigh"],
                              "Stop Loss":r["SL"],"TP1":r["TP1"],"TP2":r["TP2"],
                              "Risk %":r["RiskPct"],"R:R":r["RRR"],"ATR %":r["ATRpct"],
                              "Support 20D":r["Sup20"],"Resist 60D":r["Res60"],
                              "Lot (sizing)":lots})
        plan=pd.DataFrame(plan_rows)
        st.dataframe(plan,use_container_width=True,hide_index=True,column_config={
            "Last":st.column_config.NumberColumn(format="Rp %.0f"),
            "Buy Zone ⬇":st.column_config.NumberColumn(format="Rp %.0f"),
            "Buy Zone ⬆":st.column_config.NumberColumn(format="Rp %.0f"),
            "Stop Loss":st.column_config.NumberColumn(format="Rp %.0f"),
            "TP1":st.column_config.NumberColumn(format="Rp %.0f"),
            "TP2":st.column_config.NumberColumn(format="Rp %.0f"),
            "Support 20D":st.column_config.NumberColumn(format="Rp %.0f"),
            "Resist 60D":st.column_config.NumberColumn(format="Rp %.0f"),
            "Risk %":st.column_config.NumberColumn(format="%.2f%%"),
            "R:R":st.column_config.NumberColumn(format="%.2f"),
            "ATR %":st.column_config.NumberColumn(format="%.2f%%")})
        st.download_button("⬇️ Export Trade Plan CSV",plan.to_csv(index=False).encode(),"idx_trade_plan.csv","text/csv")
        st.caption(f"Lot dihitung dari modal Rp {capital:,.0f} dan risiko {risk_pct_cap:.1f}% per transaksi (1 lot = 100 lembar). "
                   "Semua level bersifat mekanis, bukan rekomendasi investasi.")

    detail=None
    with tabs[3]:
        choice=st.selectbox("Pilih saham",[r["Ticker"] for r in results],key="detail_choice")
        detail=next(r for r in results if r["Ticker"]==choice)
        st.markdown(f"### {detail['Ticker']} — {detail['Company']}")
        st.caption(f"{detail['Sector']} · {detail['Industry']}")
        q=st.columns(5)
        q[0].metric("Price",rup(detail["Price"])); q[1].metric("Score",f"{detail['Score']}/100")
        q[2].metric("PER",f"{detail['PER']:.2f}x" if detail["PER"] is not None else "N/A")
        q[3].metric("PBV",f"{detail['PBV']:.2f}x" if detail["PBV"] is not None else "N/A")
        q[4].metric("R:R",f"{detail['RRR']:.2f}" if detail["RRR"] is not None else "N/A")

        period=st.radio("Chart",["3M","6M","1Y"],horizontal=True,index=1)
        days={"3M":90,"6M":180,"1Y":365}[period]; d=detail["DF"].tail(days)
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.62,.18,.20])
        fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="OHLC"),row=1,col=1)
        for col in ["SMA20","SMA50","SMA200"]:
            fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col,line=dict(width=1.4)),row=1,col=1)
        for lvl,txt,dash in [(detail["BuyHigh"],"Buy zone","dot"),(detail["SL"],"Stop loss","dash"),
                             (detail["TP1"],"TP1","dash"),(detail["TP2"],"TP2","dash")]:
            if lvl: fig.add_hline(y=lvl,line_dash=dash,line_width=1,annotation_text=txt,
                                  annotation_position="right",row=1,col=1)
        fig.add_trace(go.Bar(x=d.index,y=d.Volume,name="Volume"),row=2,col=1)
        fig.add_trace(go.Scatter(x=d.index,y=d.RSI,name="RSI",line=dict(width=1.4)),row=3,col=1)
        fig.add_hline(y=70,line_dash="dot",row=3,col=1); fig.add_hline(y=30,line_dash="dot",row=3,col=1)
        fig.update_layout(template="plotly_dark",height=760,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10))
        fig.update_yaxes(range=[0,100],row=3,col=1)
        st.plotly_chart(fig,use_container_width=True)

        a,b,c=st.columns(3)
        with a:
            st.markdown("#### 📈 Technical")
            for label,value in [("RSI",detail["RSI"]),("SMA20",rup(detail["SMA20"])),("SMA50",rup(detail["SMA50"])),
                                ("SMA200",rup(detail["SMA200"])),("MACD",detail["MACD"]),
                                ("ATR14",detail["ATR"]),("ATR %",detail["ATRpct"])]:
                shown=value if isinstance(value,str) else (f"{value:.2f}" if value is not None else "N/A")
                st.markdown(f'<div class="signal"><b>{label}</b><br>{shown}</div>',unsafe_allow_html=True)
        with b:
            st.markdown("#### 💰 Fundamental")
            for label,value in [("PER",f"{detail['PER']:.2f}x" if detail['PER'] is not None else "N/A"),
                                ("PBV",f"{detail['PBV']:.2f}x" if detail['PBV'] is not None else "N/A"),
                                ("ROE",f"{detail['ROE']:.2f}%" if detail['ROE'] is not None else "N/A"),
                                ("Dividend Yield",f"{detail['Div']:.2f}%"),
                                ("Nilai transaksi 20D",rup(detail["Value20"])),
                                ("Sektor",detail['Sector'])]:
                st.markdown(f'<div class="signal"><b>{label}</b><br>{value}</div>',unsafe_allow_html=True)
        with c:
            st.markdown("#### 🎯 Rencana Transaksi")
            lots=None; nominal=None
            if detail["Risk"] and detail["Risk"]>0:
                shares=(capital*risk_pct_cap/100)/detail["Risk"]
                lots=int(shares//100); nominal=lots*100*(detail["Entry"] or detail["Price"])
            plan_items=[("Harga Beli (buy zone)",f"{rup(detail['BuyLow'])} – {rup(detail['BuyHigh'])}"),
                        ("Stop Loss (keluar rugi)",f"{rup(detail['SL'])}  ({detail['RiskPct']:.2f}% risiko)" if detail["RiskPct"] else rup(detail["SL"])),
                        ("Take Profit 1",rup(detail["TP1"])),
                        ("Take Profit 2",rup(detail["TP2"])),
                        ("Risk : Reward (TP1)",f"{detail['RRR']:.2f} : 1" if detail["RRR"] is not None else "N/A"),
                        ("Support 20 hari",rup(detail["Sup20"])),
                        ("Resistance 60 hari",rup(detail["Res60"])),
                        ("Ukuran posisi",f"{lots} lot ≈ {rup(nominal)}" if lots else "N/A")]
            for label,value in plan_items:
                st.markdown(f'<div class="plan"><b>{label}</b><br>{value}</div>',unsafe_allow_html=True)
            st.caption("Level mekanis berbasis ATR + support terdekat; bukan jaminan atau rekomendasi investasi.")
        st.markdown("#### 🔔 Signals")
        st.write(" · ".join(detail["Signals"]) if detail["Signals"] else "Tidak ada sinyal khusus.")

    with tabs[4]:
        ai_choice=st.selectbox("Saham",[r["Ticker"] for r in results],key="ai_choice")
        r=next(x for x in results if x["Ticker"]==ai_choice)
        rsi_txt=f"{r['RSI']:.2f}" if r['RSI'] is not None else "N/A"
        macd_txt=f"{r['MACD']:.4f}" if r['MACD'] is not None else "N/A"
        mds_txt=f"{r['MACDSignal']:.4f}" if r['MACDSignal'] is not None else "N/A"
        rrr_txt=f"{r['RRR']:.2f}" if r['RRR'] is not None else "N/A"
        sig_txt="\n".join("- "+s for s in r["Signals"]) if r["Signals"] else "- Tidak ada"
        prompt=f"""Anda adalah equity analyst yang melakukan analisis objektif saham IDX.

SAHAM: {r['Ticker']} — {r['Company']}
SEKTOR: {r['Sector']} / {r['Industry']}

FUNDAMENTAL
- Harga: {rup(r['Price'])}
- PER: {r['PER'] if r['PER'] is not None else 'N/A'}x
- PBV: {r['PBV'] if r['PBV'] is not None else 'N/A'}x
- ROE: {f"{r['ROE']:.2f}%" if r['ROE'] is not None else 'N/A'}
- Dividend Yield: {r['Div']:.2f}%

TEKNIKAL
- RSI: {rsi_txt}
- SMA20: {rup(r['SMA20'])}
- SMA50: {rup(r['SMA50'])}
- SMA200: {rup(r['SMA200'])}
- MACD: {macd_txt} / Signal: {mds_txt}
- ATR14: {r['ATR']:.2f} ({r['ATRpct']:.2f}% dari harga)
- Hybrid Score: {r['Score']}/100 (Technical {r['Tech']}/60, Fundamental {r['Fund']}/40)

LEVEL MEKANIS
- Buy zone: {rup(r['BuyLow'])} – {rup(r['BuyHigh'])}
- Stop loss: {rup(r['SL'])}
- TP1 / TP2: {rup(r['TP1'])} / {rup(r['TP2'])}
- Risk:Reward (TP1): {rrr_txt}
- Support 20D / Resistance 60D: {rup(r['Sup20'])} / {rup(r['Res60'])}

SIGNAL
{sig_txt}

TUGAS
1. Jelaskan kondisi fundamental.
2. Jelaskan trend dan momentum.
3. Nilai apakah level buy/stop/TP di atas masuk akal terhadap struktur harga.
4. Identifikasi risiko dan kondisi yang membatalkan setup.
5. Bedakan data, asumsi, dan interpretasi.
6. Jangan mengarang data yang tidak tersedia.
7. Gunakan Markdown profesional.
"""
        st.code(prompt,language="text")
        st.download_button("⬇️ Download AI Prompt",prompt,f"AI_prompt_{r['Ticker']}.txt","text/plain")

st.markdown('<div style="text-align:center;color:#64748b;font-size:.75rem;padding:20px">IDX Analytics Pro · Data via Yahoo Finance · Untuk edukasi dan analisis, bukan rekomendasi investasi.</div>',unsafe_allow_html=True)