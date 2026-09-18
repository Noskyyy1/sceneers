import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

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
</style>
""", unsafe_allow_html=True)

STARTER = sorted(set("""AADI ACES ADCP ADHI ADMR ADRO AGRO AKRA AMMN AMRT ANTM APLN ARCI ARTO ASII ASLC ASSA AUTO AVIA BBCA BBHI BBNI BBRI BBTN BCAP BDMN BELI BFIN BHIT BIKA BIRD BJBR BJTM BKSL BMRI BNGA BREN BRIS BRMS BSDE BTPS BUKA BUMI BUVA CARS CBRE CDIA CENT CFIN CINT CLEO CMRY CPIN CTRA CYBR DEWA DOID DSNG ELSA EMTK ENRG ERAA ESSA EXCL FAST FILM FREN GGRM GIAA GJTL GOTO HEAL HMSP HRUM ICBP IMAS INCO INDF INDY INKP INTP ISAT ITMG JSMR JPFA KBLI KIJA KLBF KPIG KRAS LINK LPKR LSIP MAPA MAPI MARK MBMA MCAS MDKA MEDC MIKA MLPL MNCN MPMX MTEL MYOR NCKL NISP PANI PGAS PGEO PNBN PNLF PPRE PPRO PTBA PTPP PTRO PWON RAJA RALS SCMA SIDO SMBR SMGR SMRA SMSM SRTG SSIA STAA SUPR TAPG TBIG TINS TKIM TLKM TOWR TPIA TRIM TUGU ULTJ UNTR UNVR WIKA WTON ZYRX""".split()))
INDEXES={
"LQ45 (starter)":"ACES ADRO AKRA AMMN AMRT ANTM ASII BBCA BBNI BBRI BBTN BMRI BRIS CPIN EMTK GOTO ICBP INCO INDF INKP ISAT ITMG JSMR KLBF MAPI MBMA MDKA MEDC PGAS PGEO PTBA PTRO SMGR TLKM TOWR TPIA UNTR UNVR".split(),
"IDX30 (starter)":"ADRO AKRA AMMN AMRT ANTM ASII BBCA BBNI BBRI BBTN BMRI BRIS CPIN GOTO ICBP INDF INCO ISAT ITMG KLBF MDKA PGAS PTBA SMGR TLKM TPIA UNTR UNVR".split(),
}

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
def history(ticker):
    try:
        raw=yf.download(f"{ticker}.JK", start=datetime.now()-timedelta(days=430), end=datetime.now(),
                        progress=False, auto_adjust=False, threads=False)
        if isinstance(raw.columns,pd.MultiIndex):
            raw.columns=raw.columns.get_level_values(0)
        req=["Open","High","Low","Close","Volume"]
        if raw.empty or not all(c in raw.columns for c in req): return pd.DataFrame()
        d=raw.dropna(subset=["Close"]).copy()
        for c in req: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["SMA20"]=d.Close.rolling(20).mean(); d["SMA50"]=d.Close.rolling(50).mean(); d["SMA200"]=d.Close.rolling(200).mean()
        delta=d.Close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
        ag=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        d["RSI"]=100-(100/(1+(ag/al.replace(0,np.nan))))
        e12=d.Close.ewm(span=12,adjust=False).mean(); e26=d.Close.ewm(span=26,adjust=False).mean()
        d["MACD"]=e12-e26; d["MACDSignal"]=d.MACD.ewm(span=9,adjust=False).mean(); d["MACDHist"]=d.MACD-d.MACDSignal
        d["VolSMA20"]=d.Volume.rolling(20).mean()
        tr=pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
        d["ATR14"]=tr.rolling(14).mean()
        return d
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals(ticker):
    out={"Company":ticker,"Sector":"N/A","Industry":"N/A","PER":None,"PBV":None,"ROE":None,"Div":0.0,"MarketCap":None}
    try:
        i=yf.Ticker(f"{ticker}.JK").info
        out.update({"Company":i.get("longName") or i.get("shortName") or ticker,"Sector":i.get("sector") or "N/A","Industry":i.get("industry") or "N/A","PER":i.get("trailingPE"),"PBV":i.get("priceToBook"),"ROE":i.get("returnOnEquity"),"Div":i.get("dividendYield") or 0.0,"MarketCap":i.get("marketCap")})
    except Exception: pass
    return out

def num(x):
    try:
        x=float(x); return x if np.isfinite(x) else None
    except Exception: return None

def analyze(ticker):
    d=history(ticker)
    if d.empty or len(d)<100: return None
    f=fundamentals(ticker); q=d.iloc[-1]
    price=num(q.Close); rsi=num(q.RSI); s20=num(q.SMA20); s50=num(q.SMA50); s200=num(q.SMA200); macd=num(q.MACD); ms=num(q.MACDSignal); vol=num(q.Volume); va=num(q.VolSMA20); atr=num(q.ATR14)
    if price is None:return None
    per=num(f["PER"]); pbv=num(f["PBV"]); roe=num(f["ROE"]); div=num(f["Div"]) or 0
    roe_pct=roe*100 if roe is not None else None; div_pct=div*100
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
    return {"Ticker":ticker,"Company":f["Company"],"Sector":f["Sector"],"Industry":f["Industry"],"Price":price,"Score":ts+fs,"Tech":ts,"Fund":fs,"PER":per,"PBV":pbv,"ROE":roe_pct,"Div":div_pct,"RSI":rsi,"SMA20":s20,"SMA50":s50,"SMA200":s200,"MACD":macd,"MACDSignal":ms,"ATR":atr,"BuyLow":price-.5*atr,"BuyHigh":price+.25*atr,"TP1":price+2*atr,"TP2":price+3.5*atr,"SL":price-1.5*atr,"Signals":signals,"DF":d}

def rup(x): return "N/A" if x is None else f"Rp {x:,.0f}"

universe_all=load_universe()
st.sidebar.markdown("## ⚙️ Screening Engine")
mode=st.sidebar.selectbox("Universe",["IDX Universe (CSV)","IDX Core (starter)","LQ45 (starter)","IDX30 (starter)","Manual"])
if mode=="IDX Universe (CSV)": universe=universe_all
elif mode=="IDX Core (starter)": universe=STARTER
elif mode in INDEXES: universe=INDEXES[mode]
else:
    raw=st.sidebar.text_area("Ticker manual","BBCA, BBRI, BMRI, BBNI",height=100)
    universe=[x.strip().upper().replace(".JK","") for x in raw.replace("\n",",").split(",") if x.strip()]
universe=sorted(set(universe))
min_score=st.sidebar.slider("Minimum Hybrid Score",0,100,55,5)
rmin,rmax=st.sidebar.slider("RSI range",0,100,(30,70))
req50=st.sidebar.checkbox("Wajib > SMA50")
req200=st.sidebar.checkbox("Wajib > SMA200")
reqmacd=st.sidebar.checkbox("Wajib MACD bullish")
max_scan=st.sidebar.number_input("Max ticker per scan",1,max(1,len(universe)),min(50,max(1,len(universe))),10)
if st.sidebar.button("🧹 Clear cache"):
    st.cache_data.clear(); st.rerun()

st.markdown('<div class="hero"><h1>📈 IDX Analytics Pro</h1><p>Hybrid Technical + Fundamental Stock Screener · Streamlit Dashboard</p></div>',unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results=[]
if "failed" not in st.session_state: st.session_state.failed=[]

c1,c2=st.columns([5,1]); c1.caption(f"Universe aktif: {len(universe)} ticker · Scan maksimum: {int(max_scan)}")
run=c2.button("🚀 RUN SCAN",type="primary",use_container_width=True)
if run:
    results=[]; failed=[]; selected=universe[:int(max_scan)]; bar=st.progress(0); status=st.empty()
    for n,t in enumerate(selected,1):
        status.write(f"Scanning **{t}** ...")
        try:
            r=analyze(t)
            if r is None: failed.append(t); continue
            ok=r["Score"]>=min_score
            if req50: ok=ok and r["SMA50"] is not None and r["Price"]>r["SMA50"]
            if req200: ok=ok and r["SMA200"] is not None and r["Price"]>r["SMA200"]
            if reqmacd: ok=ok and r["MACD"] is not None and r["MACDSignal"] is not None and r["MACD"]>r["MACDSignal"]
            if r["RSI"] is not None: ok=ok and rmin<=r["RSI"]<=rmax
            if ok: results.append(r)
        except Exception: failed.append(t)
        bar.progress(n/len(selected))
    status.empty(); bar.empty(); results.sort(key=lambda x:x["Score"],reverse=True); st.session_state.results=results; st.session_state.failed=failed

results=st.session_state.results; failed=st.session_state.failed
if not results:
    st.info("Atur parameter di sidebar lalu klik **RUN SCAN**.")
    st.markdown("### 🧭 Struktur aplikasi\n**Overview** untuk ringkasan · **Screener** untuk tabel · **Stock Detail** untuk chart · **AI Analysis** untuk prompt.")
else:
    avg=np.mean([r["Score"] for r in results]); above=sum(r["SMA50"] is not None and r["Price"]>r["SMA50"] for r in results); bull=sum(r["MACD"] is not None and r["MACDSignal"] is not None and r["MACD"]>r["MACDSignal"] for r in results); rs=[r["RSI"] for r in results if r["RSI"] is not None]
    k=st.columns(5)
    k[0].metric("Passed",len(results)); k[1].metric("Avg Score",f"{avg:.1f}/100"); k[2].metric("> SMA50",above); k[3].metric("MACD Bullish",bull); k[4].metric("Avg RSI",f"{np.mean(rs):.1f}" if rs else "N/A")
    tabs=st.tabs(["🏠 Overview","🔎 Screener","📊 Stock Detail","🤖 AI Analysis"])

    with tabs[0]:
        left,right=st.columns([1.5,1])
        with left:
            dfscore=pd.DataFrame({"Ticker":[r["Ticker"] for r in results],"Score":[r["Score"] for r in results]}).sort_values("Score")
            fig=go.Figure(go.Bar(x=dfscore.Score,y=dfscore.Ticker,orientation="h",text=dfscore.Score,textposition="outside")); fig.update_layout(template="plotly_dark",height=max(380,len(results)*30),margin=dict(l=10,r=30,t=20,b=20),xaxis_range=[0,105]); st.plotly_chart(fig,use_container_width=True)
        with right:
            st.markdown("#### Screening Snapshot")
            st.dataframe(pd.DataFrame({"Metric":["Minimum Score","Passed","Technical Avg","Fundamental Avg","Unavailable"],"Value":[min_score,len(results),round(np.mean([r["Tech"] for r in results]),1),round(np.mean([r["Fund"] for r in results]),1),len(failed)]}),hide_index=True,use_container_width=True)
            if failed: st.caption("Unavailable: "+", ".join(failed[:25]))

    with tabs[1]:
        rows=[{"Ticker":r["Ticker"],"Company":r["Company"],"Price":r["Price"],"Score":r["Score"],"Technical":r["Tech"],"Fundamental":r["Fund"],"RSI":r["RSI"],"PER":r["PER"],"PBV":r["PBV"],"ROE %":r["ROE"],"Div Yield %":r["Div"],"Sector":r["Sector"]} for r in results]
        table=pd.DataFrame(rows)
        st.dataframe(table,use_container_width=True,hide_index=True,column_config={"Price":st.column_config.NumberColumn(format="Rp %.0f"),"Score":st.column_config.ProgressColumn(min_value=0,max_value=100),"Technical":st.column_config.NumberColumn(format="%d/60"),"Fundamental":st.column_config.NumberColumn(format="%d/40"),"RSI":st.column_config.NumberColumn(format="%.2f"),"PER":st.column_config.NumberColumn(format="%.2fx"),"PBV":st.column_config.NumberColumn(format="%.2fx"),"ROE %":st.column_config.NumberColumn(format="%.2f%%"),"Div Yield %":st.column_config.NumberColumn(format="%.2f%%")})
        st.download_button("⬇️ Export CSV",table.to_csv(index=False).encode(),"idx_screening_results.csv","text/csv")

    detail=None
    with tabs[2]:
        choice=st.selectbox("Pilih saham",[r["Ticker"] for r in results],key="detail_choice"); detail=next(r for r in results if r["Ticker"]==choice)
        st.markdown(f"### {detail['Ticker']} — {detail['Company']}"); st.caption(f"{detail['Sector']} · {detail['Industry']}")
        q=st.columns(5); q[0].metric("Price",rup(detail["Price"])); q[1].metric("Score",f"{detail['Score']}/100"); q[2].metric("PER",f"{detail['PER']:.2f}x" if detail["PER"] is not None else "N/A"); q[3].metric("PBV",f"{detail['PBV']:.2f}x" if detail["PBV"] is not None else "N/A"); q[4].metric("ROE",f"{detail['ROE']:.2f}%" if detail["ROE"] is not None else "N/A")
        period=st.radio("Chart",["3M","6M","1Y"],horizontal=True,index=1); days={"3M":90,"6M":180,"1Y":365}[period]; d=detail["DF"].tail(days)
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.62,.18,.20])
        fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="OHLC"),row=1,col=1)
        for col,name in [("SMA20","SMA20"),("SMA50","SMA50"),("SMA200","SMA200")]: fig.add_trace(go.Scatter(x=d.index,y=d[col],name=name,line=dict(width=1.4)),row=1,col=1)
        fig.add_trace(go.Bar(x=d.index,y=d.Volume,name="Volume"),row=2,col=1); fig.add_trace(go.Scatter(x=d.index,y=d.RSI,name="RSI",line=dict(width=1.4)),row=3,col=1)
        fig.add_hline(y=70,line_dash="dot",row=3,col=1); fig.add_hline(y=30,line_dash="dot",row=3,col=1)
        fig.update_layout(template="plotly_dark",height=740,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10)); fig.update_yaxes(range=[0,100],row=3,col=1); st.plotly_chart(fig,use_container_width=True)
        a,b,c=st.columns(3)
        with a:
            st.markdown("#### 📈 Technical")
            for label,value in [("RSI",detail["RSI"]),("SMA20",rup(detail["SMA20"])),("SMA50",rup(detail["SMA50"])),("SMA200",rup(detail["SMA200"])),("MACD",detail["MACD"]),("ATR14",detail["ATR"])]: st.markdown(f'<div class="signal"><b>{label}</b><br>{value if isinstance(value,str) else f"{value:.2f}" if value is not None else "N/A"}</div>',unsafe_allow_html=True)
        with b:
            st.markdown("#### 💰 Fundamental")
            for label,value in [("PER",f"{detail['PER']:.2f}x" if detail['PER'] is not None else "N/A"),("PBV",f"{detail['PBV']:.2f}x" if detail['PBV'] is not None else "N/A"),("ROE",f"{detail['ROE']:.2f}%" if detail['ROE'] is not None else "N/A"),("Dividend Yield",f"{detail['Div']:.2f}%"),("Market",detail['Sector'])]: st.markdown(f'<div class="signal"><b>{label}</b><br>{value}</div>',unsafe_allow_html=True)
        with c:
            st.markdown("#### 🎯 ATR Reference")
            for label,value in [("Buy Zone",f"{rup(detail['BuyLow'])} – {rup(detail['BuyHigh'])}"),("TP1",rup(detail["TP1"])),("TP2",rup(detail["TP2"])),("Stop Reference",rup(detail["SL"]))]: st.markdown(f'<div class="signal"><b>{label}</b><br>{value}</div>',unsafe_allow_html=True)
            st.caption("Level mekanis berbasis ATR; bukan jaminan atau rekomendasi investasi.")
        st.markdown("#### 🔔 Signals"); st.write(" · ".join(detail["Signals"]) if detail["Signals"] else "Tidak ada sinyal khusus.")

    with tabs[3]:
        ai_choice=st.selectbox("Saham",[r["Ticker"] for r in results],key="ai_choice"); r=next(x for x in results if x["Ticker"]==ai_choice)
        prompt=f'''Anda adalah equity analyst yang melakukan analisis objektif saham IDX.\n\nSAHAM: {r['Ticker']} — {r['Company']}\nSEKTOR: {r['Sector']} / {r['Industry']}\n\nFUNDAMENTAL\n- Harga: {rup(r['Price'])}\n- PER: {r['PER'] if r['PER'] is not None else 'N/A'}x\n- PBV: {r['PBV'] if r['PBV'] is not None else 'N/A'}x\n- ROE: {f"{r['ROE']:.2f}%" if r['ROE'] is not None else 'N/A'}\n- Dividend Yield: {r['Div']:.2f}%\n\nTEKNIKAL\n- RSI: {r['RSI']:.2f} jika tersedia\n- SMA20: {rup(r['SMA20'])}\n- SMA50: {rup(r['SMA50'])}\n- SMA200: {rup(r['SMA200'])}\n- MACD: {r['MACD']:.4f} jika tersedia\n- MACD Signal: {r['MACDSignal']:.4f} jika tersedia\n- ATR14: {r['ATR']:.2f}\n- Hybrid Score: {r['Score']}/100\n- Technical: {r['Tech']}/60\n- Fundamental: {r['Fund']}/40\n\nSIGNAL\n{chr(10).join('- '+s for s in r['Signals']) if r['Signals'] else '- Tidak ada'}\n\nTUGAS\n1. Jelaskan kondisi fundamental.\n2. Jelaskan trend dan momentum.\n3. Identifikasi risiko dan kondisi yang membatalkan setup.\n4. Bedakan data, asumsi, dan interpretasi.\n5. Jangan mengarang data yang tidak tersedia.\n6. Gunakan Markdown profesional.\n'''
        st.code(prompt,language="text"); st.download_button("⬇️ Download AI Prompt",prompt,f"AI_prompt_{r['Ticker']}.txt","text/plain")

st.markdown('<div style="text-align:center;color:#64748b;font-size:.75rem;padding:20px">IDX Analytics Pro · Data via Yahoo Finance · Untuk edukasi dan analisis.</div>',unsafe_allow_html=True)
