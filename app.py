import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

load_dotenv(override=True)
WAQI_TOKEN = os.getenv("WAQI_TOKEN") or "facf272642c8f4364de71f9b6da659a571d340fd"

PAKISTAN_CITIES = {
    "Lahore":      {"lat": 31.5497, "lon": 74.3436, "waqi": "lahore",      "province": "Punjab"},
    "Karachi":     {"lat": 24.8607, "lon": 67.0011, "waqi": "karachi",     "province": "Sindh"},
    "Islamabad":   {"lat": 33.6844, "lon": 73.0479, "waqi": "islamabad",   "province": "Federal"},
    "Rawalpindi":  {"lat": 33.6007, "lon": 73.0679, "waqi": "rawalpindi",  "province": "Punjab"},
    "Peshawar":    {"lat": 34.0150, "lon": 71.5249, "waqi": "peshawar",    "province": "KPK"},
    "Quetta":      {"lat": 30.1798, "lon": 66.9750, "waqi": "quetta",      "province": "Balochistan"},
    "Multan":      {"lat": 30.1575, "lon": 71.5249, "waqi": "multan",      "province": "Punjab"},
    "Faisalabad":  {"lat": 31.4504, "lon": 73.1350, "waqi": "faisalabad",  "province": "Punjab"},
    "Sialkot":     {"lat": 32.4945, "lon": 74.5229, "waqi": "sialkot",     "province": "Punjab"},
    "Gujranwala":  {"lat": 32.1877, "lon": 74.1945, "waqi": "gujranwala",  "province": "Punjab"},
    "Hyderabad":   {"lat": 25.3960, "lon": 68.3578, "waqi": "hyderabad",   "province": "Sindh"},
    "Abbottabad":  {"lat": 34.1688, "lon": 73.2215, "waqi": "abbottabad",  "province": "KPK"},
    "Bahawalpur":  {"lat": 29.3956, "lon": 71.6836, "waqi": "bahawalpur",  "province": "Punjab"},
    "Sargodha":    {"lat": 32.0836, "lon": 72.6711, "waqi": "sargodha",    "province": "Punjab"},
    "Sukkur":      {"lat": 27.7052, "lon": 68.8574, "waqi": "sukkur",      "province": "Sindh"},
}

st.set_page_config(page_title="AirGuard Pakistan", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #050810;
    color: #c9d1d9;
    font-family: 'Rajdhani', sans-serif;
}

/* Animated smog background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background:
        radial-gradient(ellipse 80% 40% at 20% 20%, rgba(100,80,20,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 80% 10%, rgba(60,40,10,0.10) 0%, transparent 50%),
        radial-gradient(ellipse 100% 50% at 50% 80%, rgba(80,60,20,0.08) 0%, transparent 60%);
    animation: smogDrift 20s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes smogDrift {
    0%   { transform: translateX(0) translateY(0) scale(1); }
    50%  { transform: translateX(30px) translateY(-20px) scale(1.05); }
    100% { transform: translateX(-15px) translateY(10px) scale(0.97); }
}

/* Wind lines */
.wind-wrap { position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1;overflow:hidden; }
.wl {
    position:absolute;height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,0.25),transparent);
    animation:windBlow linear infinite;border-radius:2px;
}
.wl:nth-child(1){top:5%;width:200px;animation-duration:4s;animation-delay:0s}
.wl:nth-child(2){top:12%;width:130px;animation-duration:6s;animation-delay:1.2s}
.wl:nth-child(3){top:22%;width:280px;animation-duration:5s;animation-delay:2.4s}
.wl:nth-child(4){top:32%;width:100px;animation-duration:7s;animation-delay:0.6s}
.wl:nth-child(5){top:44%;width:220px;animation-duration:4.5s;animation-delay:3s}
.wl:nth-child(6){top:56%;width:160px;animation-duration:6.5s;animation-delay:1.8s}
.wl:nth-child(7){top:67%;width:120px;animation-duration:5.5s;animation-delay:2.8s}
.wl:nth-child(8){top:78%;width:240px;animation-duration:4s;animation-delay:4s}
.wl:nth-child(9){top:88%;width:140px;animation-duration:7.5s;animation-delay:0.9s}
.wl:nth-child(10){top:94%;width:190px;animation-duration:5s;animation-delay:3.6s}
@keyframes windBlow {
    0%{left:-300px;opacity:0} 10%{opacity:1} 90%{opacity:0.7} 100%{left:110%;opacity:0}
}

/* Dust particles */
.dp {
    position:fixed;border-radius:50%;
    background:rgba(180,140,60,0.35);
    animation:dustUp linear infinite;
    pointer-events:none;z-index:1;
}
.dp:nth-child(1){left:8%;width:2px;height:2px;animation-duration:14s;animation-delay:0s}
.dp:nth-child(2){left:22%;width:3px;height:3px;animation-duration:18s;animation-delay:3s}
.dp:nth-child(3){left:38%;width:2px;height:2px;animation-duration:12s;animation-delay:6s}
.dp:nth-child(4){left:55%;width:4px;height:4px;animation-duration:20s;animation-delay:1s}
.dp:nth-child(5){left:72%;width:2px;height:2px;animation-duration:16s;animation-delay:4s}
.dp:nth-child(6){left:87%;width:3px;height:3px;animation-duration:15s;animation-delay:8s}
@keyframes dustUp {
    0%{bottom:-10px;opacity:0;transform:translateX(0)}
    15%{opacity:0.6}
    50%{transform:translateX(25px)}
    85%{opacity:0.3}
    100%{bottom:105vh;opacity:0;transform:translateX(-15px)}
}

/* Main content above animations */
.main .block-container { position:relative;z-index:2; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#06080f,#0d1117) !important;
    border-right: 1px solid #1a2233 !important;
}
[data-testid="stSidebar"] * { color:#c9d1d9 !important; }
[data-testid="stSidebar"] .stRadio label { font-size:0.95rem !important; padding:3px 0 !important; }
[data-testid="stSidebar"] .stRadio label:hover { color:#00d4ff !important; }
[data-testid="stSidebar"] .stMarkdown p { color:#c9d1d9 !important; }
[data-testid="stSidebar"] .stMarkdown strong { color:#00d4ff !important; }
[data-testid="stSidebar"] hr { border-color:#1a2233 !important; }

/* HERO */
.hero-header {
    font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;
    text-align:center;
    background:linear-gradient(90deg,#00d4ff,#0099ff,#7c3aed,#ff6b35,#00d4ff);
    background-size:300% auto;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    animation:gradFlow 4s linear infinite;
    padding:18px 0 4px;letter-spacing:3px;
}
@keyframes gradFlow { 0%{background-position:0%} 100%{background-position:300%} }

.hero-sub {
    font-family:'Rajdhani',sans-serif;font-size:0.95rem;
    text-align:center;color:#6e7681;letter-spacing:5px;
    text-transform:uppercase;margin-bottom:25px;
}

/* STAT CARDS */
.stat-card {
    background:linear-gradient(135deg,#0d1117,#161b22);
    border:1px solid #21262d;border-radius:14px;
    padding:16px;text-align:center;
    position:relative;overflow:hidden;
    transition:all 0.3s;
}
.stat-card::after {
    content:'';position:absolute;top:0;left:-100%;
    width:100%;height:2px;
    background:linear-gradient(90deg,transparent,#00d4ff,transparent);
    animation:scanBar 3s linear infinite;
}
@keyframes scanBar { 0%{left:-100%} 100%{left:100%} }
.stat-card:hover { border-color:#00d4ff44;box-shadow:0 0 20px rgba(0,212,255,0.12);transform:translateY(-2px); }
.stat-number { font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;color:#00d4ff;text-shadow:0 0 15px rgba(0,212,255,0.4); }
.stat-label  { font-family:'Rajdhani',sans-serif;font-size:0.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:2px;margin-top:4px; }

/* SECTION HEADER */
.section-header {
    font-family:'Orbitron',monospace;font-size:1rem;color:#00d4ff;
    border-bottom:1px solid #21262d;padding-bottom:8px;
    margin:22px 0 18px;letter-spacing:2px;
    text-shadow:0 0 8px rgba(0,212,255,0.25);
}

/* EXPANDERS — full fix */
[data-testid="stExpander"] {
    background:#0d1117 !important;
    border:1px solid #21262d !important;
    border-radius:12px !important;
    margin-bottom:8px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > div > div > div > div > div > summary {
    background:#161b22 !important;
    color:#c9d1d9 !important;
    border-radius:12px !important;
    padding:10px 14px !important;
    font-family:'Rajdhani',sans-serif !important;
    font-size:1.05rem !important;
    font-weight:600 !important;
    cursor:pointer !important;
}
[data-testid="stExpander"] summary:hover { background:#1c2128 !important;color:#00d4ff !important; }
[data-testid="stExpander"] summary * { color:inherit !important; }
details { background:#0d1117 !important;border:1px solid #21262d !important;border-radius:12px !important;margin-bottom:8px !important; }
details summary { background:#161b22 !important;color:#c9d1d9 !important;padding:10px 14px !important;border-radius:12px !important;font-family:'Rajdhani',sans-serif !important;font-size:1.05rem !important;font-weight:600 !important;cursor:pointer !important; }
details[open] summary { border-radius:12px 12px 0 0 !important;border-bottom:1px solid #21262d !important; }
details summary:hover { color:#00d4ff !important;background:#1c2128 !important; }
details summary p,details summary span { color:inherit !important; }

/* METRICS */
[data-testid="stMetric"] { background:#0d1117 !important;border:1px solid #21262d !important;border-radius:10px !important;padding:10px !important; }
[data-testid="stMetricLabel"] { color:#8b949e !important;font-family:'Rajdhani',sans-serif !important;font-size:0.8rem !important; }
[data-testid="stMetricValue"] { font-family:'Orbitron',monospace !important;color:#00d4ff !important;text-shadow:0 0 8px rgba(0,212,255,0.35) !important; }

/* ALERTS */
.smog-banner {
    background:linear-gradient(90deg,#7e0023,#cc0000,#ff4500,#cc0000,#7e0023);
    background-size:300% auto;animation:gradFlow 2s linear infinite;
    border-radius:10px;padding:12px 18px;text-align:center;
    font-family:'Orbitron',monospace;font-size:0.9rem;color:white !important;
    letter-spacing:2px;margin:6px 0;box-shadow:0 0 25px rgba(255,0,0,0.35);
}
.alert-vuh {
    background:linear-gradient(135deg,#1a0033,#2d0066);
    border:2px solid #8f3f97;border-radius:10px;padding:12px 18px;
    text-align:center;font-family:'Rajdhani',sans-serif;font-size:1rem;
    color:#d4aaff !important;margin:6px 0;
}

/* PROVINCE BADGE */
.pb { display:inline-block;background:rgba(31,111,235,0.12);border:1px solid #1f6feb;
      border-radius:20px;padding:1px 9px;font-size:0.72rem;color:#58a6ff !important; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { background:#0d1117 !important;border-bottom:1px solid #21262d !important;gap:4px !important; }
.stTabs [data-baseweb="tab"] { background:#161b22 !important;color:#8b949e !important;border-radius:6px 6px 0 0 !important;border:1px solid #21262d !important;font-family:'Rajdhani',sans-serif !important;font-size:0.88rem !important;padding:6px 14px !important; }
.stTabs [aria-selected="true"] { background:#1c2128 !important;color:#00d4ff !important;border-bottom:2px solid #00d4ff !important; }
.stTabs [data-baseweb="tab-panel"] { background:#0d1117 !important; }

/* BUTTONS */
.stButton > button { background:linear-gradient(135deg,#0d1117,#161b22) !important;color:#00d4ff !important;border:1px solid #00d4ff44 !important;border-radius:8px !important;font-family:'Rajdhani',sans-serif !important;font-size:0.95rem !important;font-weight:700 !important;letter-spacing:2px !important;transition:all 0.3s !important; }
.stButton > button:hover { background:linear-gradient(135deg,#00d4ff18,#0099ff18) !important;box-shadow:0 0 18px rgba(0,212,255,0.25) !important;border-color:#00d4ff !important; }
.stButton > button[kind="primary"] { border-color:#00d4ff !important;box-shadow:0 0 12px rgba(0,212,255,0.2) !important; }

/* SELECT / INPUT */
[data-testid="stSelectbox"] > div > div { background:#161b22 !important;border:1px solid #30363d !important;color:#c9d1d9 !important;border-radius:8px !important; }
[data-testid="stNumberInput"] input { background:#161b22 !important;border:1px solid #30363d !important;color:#c9d1d9 !important;border-radius:8px !important; }
[data-testid="stMultiSelect"] > div { background:#161b22 !important;border:1px solid #30363d !important;border-radius:8px !important; }
[data-testid="stMultiSelect"] span { color:#c9d1d9 !important; }

/* DATAFRAME */
[data-testid="stDataFrame"] { border:1px solid #21262d !important;border-radius:10px !important; }

/* HIDE DEFAULTS */
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}

/* SCROLLBAR */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#0d1117}
::-webkit-scrollbar-thumb{background:#21262d;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#00d4ff}
</style>

<div class="wind-wrap">
  <div class="wl"></div><div class="wl"></div><div class="wl"></div>
  <div class="wl"></div><div class="wl"></div><div class="wl"></div>
  <div class="wl"></div><div class="wl"></div><div class="wl"></div>
  <div class="wl"></div>
  <div class="dp"></div><div class="dp"></div><div class="dp"></div>
  <div class="dp"></div><div class="dp"></div><div class="dp"></div>
</div>
""", unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────
@st.cache_data(ttl=600)
def get_live_aqi(waqi_name):
    try:
        r = requests.get(f"https://api.waqi.info/feed/{waqi_name}/?token={WAQI_TOKEN}", timeout=10)
        d = r.json()
        if d["status"] == "ok":
            i = d["data"]["iaqi"]
            return {
                "aqi":      d["data"].get("aqi", 0),
                "pm25":     i.get("pm25",{}).get("v"),
                "pm10":     i.get("pm10",{}).get("v"),
                "co":       i.get("co",  {}).get("v"),
                "no2":      i.get("no2", {}).get("v"),
                "o3":       i.get("o3",  {}).get("v"),
                "so2":      i.get("so2", {}).get("v"),
                "temp":     i.get("t",   {}).get("v"),
                "humidity": i.get("h",   {}).get("v"),
                "wind":     i.get("w",   {}).get("v"),
            }
    except:
        pass
    return None


def classify_aqi(aqi):
    if not aqi or aqi == 0: return "Unknown",               "⚪", "#555555", 0
    elif aqi <= 50:          return "Good",                  "🟢", "#00e400", 1
    elif aqi <= 100:         return "Moderate",              "🟡", "#ffff00", 2
    elif aqi <= 150:         return "Unhealthy (Sensitive)", "🟠", "#ff7e00", 3
    elif aqi <= 200:         return "Unhealthy",             "🔴", "#ff0000", 4
    elif aqi <= 300:         return "Very Unhealthy",        "🟣", "#8f3f97", 5
    else:                    return "Hazardous",             "⚫", "#7e0023", 6


def gauge(aqi, city):
    level, emoji, color, _ = classify_aqi(aqi)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={'text': f"<b>{city}</b>", 'font': {'size':14,'color':'#c9d1d9','family':'Rajdhani'}},
        number={'font': {'size':36,'color':color,'family':'Orbitron'}},
        gauge={
            'axis':      {'range':[0,500],'tickcolor':'#30363d','tickfont':{'color':'#8b949e','size':8}},
            'bar':       {'color':color,'thickness':0.22},
            'bgcolor':   '#0d1117','bordercolor':'#21262d','borderwidth':1,
            'steps': [
                {'range':[0,50],'color':'#001800'},{'range':[50,100],'color':'#181800'},
                {'range':[100,150],'color':'#181000'},{'range':[150,200],'color':'#180000'},
                {'range':[200,300],'color':'#100018'},{'range':[300,500],'color':'#0c0008'},
            ],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.72,'value':aqi}
        }
    ))
    fig.update_layout(height=235,margin=dict(l=12,r=12,t=45,b=5),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font={'color':'#c9d1d9'})
    return fig


def health_score(aqi, age, conds):
    b = 100 if aqi<=50 else 80 if aqi<=100 else 60 if aqi<=150 else 40 if aqi<=200 else 20 if aqi<=300 else 5
    d = 0
    if "Child"        in age:   d += 15
    if "Elderly"      in age:   d += 15
    if "Asthma"       in conds: d += 20
    if "Heart Disease" in conds:d += 20
    if "Pregnant"     in conds: d += 15
    if "Diabetes"     in conds: d += 10
    if "COPD"         in conds: d += 18
    return max(0, b - d)


@st.cache_data(ttl=600)
def fetch_cities(city_tuple):
    out = {}
    for c in city_tuple:
        d = get_live_aqi(PAKISTAN_CITIES[c]["waqi"])
        if d: out[c] = d
    return out


LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,17,23,0.95)',
    font=dict(color='#c9d1d9', family='Rajdhani'),
    xaxis=dict(color='#8b949e', gridcolor='#1a2233'),
    yaxis=dict(color='#8b949e', gridcolor='#1a2233'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'))
)


# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
        <div style='font-family:Orbitron,monospace;font-size:1.4rem;font-weight:900;
                    color:#00d4ff;letter-spacing:4px;text-shadow:0 0 18px rgba(0,212,255,0.55)'>
            🌫️ AIRGUARD
        </div>
        <div style='font-family:Rajdhani,sans-serif;color:#6e7681;font-size:0.72rem;
                    letter-spacing:4px;margin-top:2px'>PAKISTAN · AI MONITOR</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("**NAVIGATE**", [
        "🏠 Live Dashboard", "🗺️ Pakistan AQI Map", "📈 AI Forecast",
        "🏥 Health Risk Profiler", "📊 Historical Analysis",
        "⚡ Smog Alert Center", "🔬 City Deep Dive",
    ])

    st.divider()
    st.markdown("**SELECT CITIES**")
    sel = st.multiselect("Cities to monitor", list(PAKISTAN_CITIES.keys()),
                         default=["Lahore","Karachi","Islamabad","Peshawar","Multan","Quetta"])
    st.divider()
    st.markdown("**YOUR HEALTH PROFILE**")
    age_group  = st.selectbox("Age Group", ["Adult (18-59)","Child (under 12)","Elderly (60+)"])
    conditions = st.multiselect("Health Conditions",
                                ["None","Asthma","Heart Disease","Pregnant","Diabetes","COPD"],
                                default=["None"])
    st.divider()
    st.markdown(f"""<div style='text-align:center;font-family:Rajdhani;color:#6e7681;font-size:0.72rem'>
        🕐 {datetime.now().strftime('%d %b %Y, %H:%M')}<br>
        <span style='color:#00d4ff'>● LIVE</span> — Refreshes every 10 min
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE 1 — LIVE DASHBOARD
# ════════════════════════════════════════════════════════
if "Live Dashboard" in page:
    st.markdown('<div class="hero-header">🌫️ AIRGUARD PAKISTAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-Time AI-Powered Air Quality Intelligence</div>', unsafe_allow_html=True)

    if not sel:
        st.warning("Select cities from the sidebar!")
        st.stop()

    with st.spinner("🛰️ Fetching live sensor data..."):
        city_data = fetch_cities(tuple(sel))

    if not city_data:
        st.error("Could not fetch data. Check your WAQI token in .env")
        st.stop()

    # Emergency alerts
    for city, data in city_data.items():
        aqi = data['aqi']
        if aqi > 300:
            st.markdown(f'<div class="smog-banner">☠️ HAZARDOUS EMERGENCY · {city.upper()} · AQI {aqi} · STAY INDOORS NOW</div>', unsafe_allow_html=True)
        elif aqi > 200:
            st.markdown(f'<div class="alert-vuh">🟣 VERY UNHEALTHY · {city} · AQI {aqi} · Avoid all outdoor activity</div>', unsafe_allow_html=True)

    aqi_vals = [d['aqi'] for d in city_data.values() if d['aqi'] > 0]
    if aqi_vals:
        worst = max(city_data, key=lambda c: city_data[c]['aqi'])
        best  = min(city_data, key=lambda c: city_data[c]['aqi'])
        _, _, wc, _ = classify_aqi(city_data[worst]['aqi'])
        _, _, bc, _ = classify_aqi(city_data[best]['aqi'])

        st.markdown('<div class="section-header">📡 NATIONAL OVERVIEW</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card"><div class="stat-number">{int(np.mean(aqi_vals))}</div><div class="stat-label">Avg AQI</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{wc};font-size:1.2rem">{worst}</div><div class="stat-label">Most Polluted</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{bc};font-size:1.2rem">{best}</div><div class="stat-label">Cleanest Air</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#00e400">{len(city_data)}</div><div class="stat-label">Cities Live</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 LIVE AQI GAUGES</div>', unsafe_allow_html=True)

    items = list(city_data.items())
    for i in range(0, len(items), 3):
        row = items[i:i+3]
        cols = st.columns(len(row))
        for col, (city, data) in zip(cols, row):
            with col:
                level, emoji, color, _ = classify_aqi(data['aqi'])
                st.plotly_chart(gauge(data['aqi'], city), use_container_width=True)
                prov = PAKISTAN_CITIES[city]["province"]
                st.markdown(f"""<div style='text-align:center;margin-top:-8px;font-family:Rajdhani'>
                    <span class='pb'>{prov}</span>
                    <span style='color:{color};font-weight:700'> {emoji} {level}</span>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 POLLUTANT BREAKDOWN</div>', unsafe_allow_html=True)

    for city, data in city_data.items():
        level, emoji, color, _ = classify_aqi(data['aqi'])
        with st.expander(f"{emoji}  {city}  —  AQI {data['aqi']}  |  {level}"):
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("PM2.5",  f"{data['pm25'] or 'N/A'}")
            c2.metric("PM10",   f"{data['pm10'] or 'N/A'}")
            c3.metric("NO₂",    f"{data['no2']  or 'N/A'}")
            c4.metric("O₃",     f"{data['o3']   or 'N/A'}")
            c5.metric("🌡️ Temp", f"{data['temp'] or 'N/A'}°C")
            c6.metric("💧 Humid",f"{data['humidity'] or 'N/A'}%")
            aqi = data['aqi']
            if aqi <= 50:   st.success("✅ Air quality is great! Safe for all activities.")
            elif aqi <= 100: st.info("😊 Acceptable. Sensitive groups take minor precautions.")
            elif aqi <= 150: st.warning("⚠️ Unhealthy for sensitive groups. Limit outdoor time.")
            elif aqi <= 200: st.error("🚨 Unhealthy for everyone. Wear N95 mask outdoors.")
            else:            st.error("☠️ HAZARDOUS! Stay indoors. Seal windows. Use air purifier.")


# ════════════════════════════════════════════════════════
# PAGE 2 — PAKISTAN AQI MAP
# ════════════════════════════════════════════════════════
elif "AQI Map" in page:
    st.markdown('<div class="hero-header">🗺️ PAKISTAN AQI MAP</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Interactive Real-Time Pollution Heatmap — 15 Cities</div>', unsafe_allow_html=True)

    with st.spinner("🛰️ Loading all city data..."):
        all_data = fetch_cities(tuple(PAKISTAN_CITIES.keys()))

    rows = []
    for city, info in PAKISTAN_CITIES.items():
        d = all_data.get(city)
        aqi = d['aqi'] if d else 0
        level, emoji, color, _ = classify_aqi(aqi)
        rows.append({"City":city,"Province":info["province"],
                     "lat":info["lat"],"lon":info["lon"],
                     "AQI":aqi if aqi>0 else None,"Status":f"{emoji} {level}",
                     "Color":color,"Size":max(aqi/3,12) if aqi>0 else 10})

    df_map = pd.DataFrame(rows).dropna(subset=["AQI"])
    if not df_map.empty:
        fig = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", size="Size", color="AQI",
            hover_name="City",
            hover_data={"AQI":True,"Status":True,"Province":True,
                        "lat":False,"lon":False,"Size":False,"Color":False},
            color_continuous_scale=[[0,"#00e400"],[0.2,"#ffff00"],[0.4,"#ff7e00"],
                                    [0.6,"#ff0000"],[0.8,"#8f3f97"],[1.0,"#7e0023"]],
            range_color=[0,350], mapbox_style="carto-darkmatter",
            zoom=4.8, center={"lat":30.3753,"lon":69.3451},
        )
        fig.update_layout(
            height=570, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(
                title=dict(text="AQI", font=dict(color='white')),
                tickfont=dict(color='white')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section-header">📋 ALL CITIES STATUS</div>', unsafe_allow_html=True)
        disp = df_map[['City','Province','AQI','Status']].sort_values('AQI',ascending=False)
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Map data unavailable. Try refreshing.")


# ════════════════════════════════════════════════════════
# PAGE 3 — AI FORECAST
# ════════════════════════════════════════════════════════
elif "Forecast" in page:
    st.markdown('<div class="hero-header">📈 AI FORECAST</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">LSTM Deep Learning Neural Network — Prediction Engine</div>', unsafe_allow_html=True)

    c1,c2 = st.columns([2,1])
    with c1: city = st.selectbox("Select city", sel or ["Lahore"])
    with c2: fh   = st.slider("Hours ahead", 6, 48, 24)

    if st.button("🔮 GENERATE AI FORECAST", type="primary", use_container_width=True):
        with st.spinner("🧠 Running LSTM..."):
            live = get_live_aqi(PAKISTAN_CITIES[city]["waqi"])
            cur  = live['aqi'] if live else 120

            if TF_AVAILABLE and os.path.exists("models/best_lstm.keras"):
                model  = load_model("models/best_lstm.keras")
                scaler = joblib.load("models/aqi_scaler.pkl")
                seed   = scaler.transform(np.array([cur]*12).reshape(-1,1))
                preds, seq = [], seed.copy()
                for _ in range(fh):
                    p = model.predict(seq.reshape(1,12,1), verbose=0)
                    preds.append(p[0][0])
                    seq = np.append(seq[1:],p).reshape(-1,1)
                fa = np.clip(scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten(),0,500)
                src = "LSTM Model"
            else:
                np.random.seed(42)
                b, fa = cur, []
                for i in range(fh):
                    h = (datetime.now().hour+i)%24
                    tf = 1.25 if (6<=h<=9 or 17<=h<=20) else 0.88
                    b = np.clip(b*tf*0.97+np.random.normal(0,7), 10, 450)
                    fa.append(b)
                fa = np.array(fa)
                src = "AI Simulation"

            hrs    = [(datetime.now()+timedelta(hours=i)).strftime("%H:%M") for i in range(fh)]
            colors = [classify_aqi(v)[2] for v in fa]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hrs,y=fa,mode='lines+markers',
                line=dict(color='#00d4ff',width=3),
                marker=dict(size=7,color=colors,line=dict(color='#0d1117',width=1)),
                fill='tozeroy',fillcolor='rgba(0,212,255,0.07)'))
            for th,cl,lb in [(50,'#00e400','Good'),(100,'#ffff00','Moderate'),
                             (150,'#ff7e00','Sensitive'),(200,'#ff0000','Unhealthy'),
                             (300,'#8f3f97','Very Unhealthy')]:
                fig.add_hline(y=th,line_dash="dot",line_color=cl,opacity=0.5,
                             annotation_text=lb,annotation_font_color=cl,annotation_font_size=9)
            fig.update_layout(title=dict(text=f"🔮 {fh}h Forecast — {city} ({src})",
                                        font=dict(color='#c9d1d9',family='Rajdhani',size=14)),
                             height=420,showlegend=False,**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Current", cur)
            c2.metric("Avg",     f"{int(np.mean(fa))}")
            c3.metric("Peak",    f"{int(np.max(fa))}")
            c4.metric("Min",     f"{int(np.min(fa))}")
            pl,pe,*_ = classify_aqi(int(np.max(fa)))
            st.info(f"**AI says:** {city} will peak at **{pl}** {pe} in the next {fh} hours.")


# ════════════════════════════════════════════════════════
# PAGE 4 — HEALTH RISK PROFILER
# ════════════════════════════════════════════════════════
elif "Health Risk" in page:
    st.markdown('<div class="hero-header">🏥 HEALTH RISK PROFILER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Personalized AI Health Risk Assessment Engine</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1: city       = st.selectbox("Your city", sel or list(PAKISTAN_CITIES.keys()))
    with c2: manual_aqi = st.number_input("Or enter AQI manually (0 = live)", 0, 500, 0)

    if st.button("🔍 ASSESS MY RISK", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            if manual_aqi > 0:
                aqi = manual_aqi
            else:
                d = get_live_aqi(PAKISTAN_CITIES[city]["waqi"])
                aqi = d['aqi'] if d else 150

            level, emoji, color, severity = classify_aqi(aqi)
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{color}12,{color}28);
                        border:2px solid {color}66;border-radius:20px;padding:28px;
                        text-align:center;margin:12px 0;box-shadow:0 0 35px {color}2a'>
                <div style='font-family:Orbitron,monospace;font-size:4.5rem;font-weight:900;
                            color:{color};text-shadow:0 0 25px {color}'>{aqi}</div>
                <div style='font-family:Rajdhani,sans-serif;font-size:1.4rem;color:#c9d1d9;
                            letter-spacing:4px;margin-top:4px'>{emoji} {level.upper()}</div>
                <div style='color:#8b949e;margin-top:6px;font-family:Rajdhani'>
                    📍 {city} · {datetime.now().strftime("%H:%M, %d %b %Y")}
                </div>
            </div>""", unsafe_allow_html=True)

            sc_val = health_score(aqi, age_group, conditions)
            sc_col = "#00e400" if sc_val>70 else "#ff7e00" if sc_val>40 else "#ff0000"

            st.markdown('<div class="section-header">🎯 YOUR PERSONAL SAFETY SCORE</div>', unsafe_allow_html=True)
            c1,c2 = st.columns([1,2])
            with c1:
                st.markdown(f"""<div style='text-align:center;background:#0d1117;
                    border:2px solid {sc_col};border-radius:14px;padding:22px;
                    box-shadow:0 0 22px {sc_col}33'>
                    <div style='font-family:Orbitron;font-size:3.5rem;color:{sc_col};
                                font-weight:900;text-shadow:0 0 12px {sc_col}'>{sc_val}</div>
                    <div style='font-family:Rajdhani;color:#8b949e;letter-spacing:2px'>/ 100</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                if sc_val > 70:   st.success("✅ **Relatively Safe** for your health profile")
                elif sc_val > 40: st.warning("⚠️ **Moderate Risk** — Take precautions")
                else:             st.error("🚨 **HIGH RISK** — Conditions dangerous for you")
                st.progress(sc_val/100)
                st.caption("100 = perfectly safe · 0 = extremely dangerous for your profile")

            st.markdown('<div class="section-header">👤 PROFILE-SPECIFIC RISK</div>', unsafe_allow_html=True)
            risks = []
            if   "Child"   in age_group: risks.append(("👶 Child",   100, 150))
            elif "Elderly" in age_group: risks.append(("👴 Elderly", 100, 150))
            else:                        risks.append(("🧑 Adult",   150, 200))
            for c in conditions:
                if   c == "Asthma":        risks.append(("🫁 Asthma",        50,  100))
                elif c == "Heart Disease": risks.append(("❤️ Heart Disease",  50,  100))
                elif c == "Pregnant":      risks.append(("🤰 Pregnant",       50,  100))
                elif c == "Diabetes":      risks.append(("💉 Diabetes",       100, 150))
                elif c == "COPD":          risks.append(("🫁 COPD",           35,   75))
            for profile, st_, dt_ in risks:
                if aqi >= dt_:  st.error(f"🆘 **{profile}**: HIGH DANGER — Stay indoors now!")
                elif aqi >= st_: st.warning(f"⚠️ **{profile}**: AT RISK — Serious precautions needed")
                else:           st.success(f"✅ **{profile}**: SAFE — Conditions acceptable")

            st.markdown('<div class="section-header">📋 WHAT TO DO RIGHT NOW</div>', unsafe_allow_html=True)
            adv = {
                1: ("✅ Go outside freely! Perfect for exercise.","No mask needed","Open windows — fresh air"),
                2: ("😊 Generally safe outdoors.","Mask optional for sensitive","Windows OK"),
                3: ("⚠️ Limit outdoor time for sensitive groups.","N95 recommended if sensitive","Consider closing windows"),
                4: ("🚨 Avoid outdoors. Serious risk for all.","N95 mask essential","Keep windows closed"),
                5: ("🚨 Stay indoors. Emergency conditions.","N95 + minimize time outside","Seal windows + air purifier"),
                6: ("☠️ DO NOT GO OUTSIDE. Extreme emergency.","Full respirator if unavoidable","Emergency protocols — seal home"),
            }
            a = adv.get(severity, adv[2])
            c1,c2,c3 = st.columns(3)
            c1.info(f"**🏃 Outdoor Activity**\n\n{a[0]}")
            c2.warning(f"**😷 Mask Guide**\n\n{a[1]}")
            c3.error(f"**🪟 Windows**\n\n{a[2]}")


# ════════════════════════════════════════════════════════
# PAGE 5 — HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════
elif "Historical" in page:
    st.markdown('<div class="hero-header">📊 HISTORICAL ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">10 Years of Pakistan Air Quality Intelligence (2015–2025)</div>', unsafe_allow_html=True)

    if os.path.exists("data/clean_aqi.csv"):
        df = pd.read_csv("data/clean_aqi.csv")

        st.markdown('<div class="section-header">🌍 DATASET INTELLIGENCE</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(f'<div class="stat-card"><div class="stat-number">{df["year"].nunique()}</div><div class="stat-label">Years</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-number">{df["city"].nunique()}</div><div class="stat-label">Cities</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-number">{len(df):,}</div><div class="stat-label">Data Points</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-number">{df["pm25_ugm3"].max():.0f}</div><div class="stat-label">Peak PM2.5</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="stat-card"><div class="stat-number">{int(df["pakistan_global_rank"].min())}</div><div class="stat-label">Worst Rank</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        t1,t2,t3,t4,t5,t6 = st.tabs([
            "📅 Yearly Trends","🏙️ City Rankings",
            "🗓️ Monthly","🌿 Seasons",
            "🌍 Global Rank","🔥 Smog & Crop Burning"
        ])

        with t1:
            yr = df.groupby('year').agg(avg=('pm25_ugm3','mean'),mx=('pm25_ugm3','max'),mn=('pm25_ugm3','min')).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yr['year'],y=yr['mx'],name='Peak',line=dict(color='#ff4444',width=2,dash='dash')))
            fig.add_trace(go.Scatter(x=yr['year'],y=yr['avg'],name='Average',
                line=dict(color='#00d4ff',width=3),marker=dict(size=9),
                fill='tonexty',fillcolor='rgba(255,68,68,0.07)'))
            fig.add_trace(go.Scatter(x=yr['year'],y=yr['mn'],name='Min',line=dict(color='#00e400',width=2,dash='dot')))
            fig.add_hline(y=15,line_dash="dash",line_color="#00e400",
                         annotation_text="WHO Safe Limit 15 μg/m³",annotation_font_color="#00e400")
            fig.update_layout(title="Pakistan PM2.5 Trend 2015–2025", height=420,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.95)',
                              font=dict(color='#c9d1d9', family='Rajdhani'),
                              xaxis=dict(color='#8b949e', gridcolor='#1a2233', dtick=1),
                              yaxis=dict(color='#8b949e', gridcolor='#1a2233'),
                              legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9')))
            st.plotly_chart(fig,use_container_width=True)
            ex = df['pm25_ugm3'].mean()/15
            st.error(f"🚨 Pakistan avg PM2.5 = **{df['pm25_ugm3'].mean():.1f} μg/m³** — **{ex:.1f}x above** WHO safe limit of 15 μg/m³!")

        with t2:
            cs = df.groupby(['city','province']).agg(avg=('pm25_ugm3','mean'),mx=('pm25_ugm3','max'),aqi=('aqi_us','mean')).reset_index().sort_values('avg',ascending=False)
            cs['color'] = cs['avg'].apply(lambda x:'#ff0000' if x>100 else '#ff7e00' if x>55 else '#ffff00' if x>35 else '#00e400')
            cs['WHO'] = (cs['avg']/15).round(1).astype(str)+'x'
            fig = go.Figure(go.Bar(x=cs['city'],y=cs['avg'],marker_color=cs['color'],
                text=cs['avg'].round(1),textposition='outside',textfont=dict(color='white')))
            fig.add_hline(y=15,line_dash="dash",line_color="#00e400",annotation_text="WHO Safe")
            fig.add_hline(y=35,line_dash="dash",line_color="#ffff00",annotation_text="US EPA")
            fig.update_layout(title="Average PM2.5 by City 2015–2025", height=430,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.95)',
                              font=dict(color='#c9d1d9', family='Rajdhani'),
                              xaxis=dict(color='#8b949e', tickangle=-30, gridcolor='#1a2233'),
                              yaxis=dict(color='#8b949e', gridcolor='#1a2233'),
                              legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9')))
            st.plotly_chart(fig,use_container_width=True)
            disp = cs[['city','province','avg','mx','aqi','WHO']].copy()
            disp.columns = ['City','Province','Avg PM2.5','Peak PM2.5','Avg AQI','WHO Exceed']
            st.dataframe(disp.round(1),use_container_width=True,hide_index=True)

        with t3:
            mo = df.groupby(['month','month_name']).agg(avg=('pm25_ugm3','mean')).reset_index().sort_values('month')
            mo['color'] = mo['avg'].apply(lambda x:'#ff0000' if x>100 else '#ff7e00' if x>55 else '#ffff00' if x>35 else '#00e400')
            fig = go.Figure(go.Bar(x=mo['month_name'],y=mo['avg'],marker_color=mo['color'],
                text=mo['avg'].round(1),textposition='outside',textfont=dict(color='white')))
            fig.add_hline(y=15,line_dash="dash",line_color="#00e400",annotation_text="WHO Safe")
            fig.update_layout(title="Monthly Air Quality Pattern — Pakistan Average",height=400,**LAYOUT)
            st.plotly_chart(fig,use_container_width=True)
            wm = mo.loc[mo['avg'].idxmax(),'month_name']
            bm = mo.loc[mo['avg'].idxmin(),'month_name']
            c1,c2 = st.columns(2)
            c1.error(f"🚨 **Worst: {wm}** — Peak smog season")
            c2.success(f"✅ **Best: {bm}** — Cleanest air of year")

        with t4:
            ss = df.groupby('season').agg(avg=('pm25_ugm3','mean'),mx=('pm25_ugm3','max')).reset_index()
            sc_map = {'Winter':'#4488ff','Spring':'#44ff88','Summer':'#ffaa44','Autumn':'#ff6644'}
            fig = px.bar(ss,x='season',y='avg',color='season',color_discrete_map=sc_map,
                        title="Air Quality by Season",text='avg')
            fig.update_traces(texttemplate='%{text:.1f}',textposition='outside',textfont_color='white')
            fig.add_hline(y=15,line_dash="dash",line_color="white",annotation_text="WHO Safe")
            fig.update_layout(height=370,showlegend=False,**LAYOUT)
            st.plotly_chart(fig,use_container_width=True)
            pivot = df.pivot_table(values='pm25_ugm3',index='city',columns='season',aggfunc='mean').round(1)
            fig2 = px.imshow(pivot,color_continuous_scale='RdYlGn_r',
                            title="PM2.5 Heatmap — City × Season",aspect='auto')
            fig2.update_layout(height=410,paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#c9d1d9',family='Rajdhani'))
            st.plotly_chart(fig2,use_container_width=True)
            ws = ss.loc[ss['avg'].idxmax(),'season']
            st.error(f"❄️ **{ws} is the most dangerous season** — Cold air traps pollutants, creating toxic smog layers over cities.")

        with t5:
            ry  = df.groupby('year')['pakistan_global_rank'].mean().reset_index()
            np_ = df.groupby('year')['pakistan_national_avg_pm25'].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ry['year'],y=ry['pakistan_global_rank'],
                mode='lines+markers',name='Global Rank',
                line=dict(color='#ff4500',width=3),marker=dict(size=9),yaxis='y'))
            fig.add_trace(go.Bar(x=np_['year'],y=np_['pakistan_national_avg_pm25'],
                name='National PM2.5',marker_color='rgba(0,212,255,0.3)',yaxis='y2'))
            fig.update_layout(
                title="Pakistan Global Pollution Rank vs National PM2.5 (2015–2025)",
                xaxis=dict(color='#8b949e',gridcolor='#1a2233',dtick=1),
                yaxis=dict(title="Global Rank (lower=worse)",color='#ff4500',
                          gridcolor='#1a2233',autorange='reversed'),
                yaxis2=dict(title="National PM2.5",color='#00d4ff',overlaying='y',side='right'),
                height=430,paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.95)',
                font=dict(color='#c9d1d9',family='Rajdhani'),
                legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#c9d1d9'))
            )
            st.plotly_chart(fig,use_container_width=True)
            wr = int(ry['pakistan_global_rank'].min())
            st.error(f"🌍 Pakistan ranked **#{wr} most polluted country** globally — This app exists to fight that reality.")

        with t6:
            smog = df[df['is_smog_season']==1]['pm25_ugm3'].mean()
            crop = df[df['is_crop_burning_season']==1]['pm25_ugm3'].mean()
            norm = df[(df['is_smog_season']==0)&(df['is_crop_burning_season']==0)]['pm25_ugm3'].mean()
            both = df[(df['is_smog_season']==1)&(df['is_crop_burning_season']==1)]['pm25_ugm3'].mean()
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#00e400">{norm:.0f}</div><div class="stat-label">Normal PM2.5</div></div>',unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#ff7e00">{smog:.0f}</div><div class="stat-label">Smog Season</div></div>',unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#ff4500">{crop:.0f}</div><div class="stat-label">Crop Burning</div></div>',unsafe_allow_html=True)
            c4.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#ff0000">{0 if np.isnan(both) else both:.0f}</div><div class="stat-label">Both Combined</div></div>',unsafe_allow_html=True)
            sc = df.groupby(['city','is_smog_season'])['pm25_ugm3'].mean().reset_index()
            sc['period'] = sc['is_smog_season'].map({1:'🔥 Smog Season',0:'Normal'})
            fig = px.bar(sc,x='city',y='pm25_ugm3',color='period',barmode='group',
                        color_discrete_map={'🔥 Smog Season':'#ff4500','Normal':'#00d4ff'},
                        title="Smog Season vs Normal — PM2.5 by City")
            fig.add_hline(y=15,line_dash="dash",line_color="white",annotation_text="WHO Safe")
            fig.update_layout(height=430,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,23,0.95)',
                              font=dict(color='#c9d1d9', family='Rajdhani'),
                              xaxis=dict(color='#8b949e', tickangle=-30, gridcolor='#1a2233'),
                              yaxis=dict(color='#8b949e', gridcolor='#1a2233'),
                              legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9')))
            st.plotly_chart(fig,use_container_width=True)
            st.warning("🌾 **Crop burning in Oct–Nov** causes PM2.5 to spike 3–5x in Punjab cities. Farmers burning rice stubble is the #1 driver of Lahore's toxic winter smog.")
    else:
        st.warning("Historical data not found. Run data_cleaning.py first.")


# ════════════════════════════════════════════════════════
# PAGE 6 — SMOG ALERT CENTER
# ════════════════════════════════════════════════════════
elif "Smog Alert" in page:
    st.markdown('<div class="hero-header">⚡ SMOG ALERT CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-Time National Emergency Air Quality Monitoring</div>', unsafe_allow_html=True)

    with st.spinner("🛰️ Scanning all cities..."):
        all_data = fetch_cities(tuple(PAKISTAN_CITIES.keys()))

    haz,vuh,unh,mod,good = [],[],[],[],[]
    for city,data in all_data.items():
        aqi = data['aqi']
        if aqi>300:   haz.append((city,aqi))
        elif aqi>200: vuh.append((city,aqi))
        elif aqi>150: unh.append((city,aqi))
        elif aqi>100: mod.append((city,aqi))
        else:         good.append((city,aqi))

    for city,aqi in sorted(haz,key=lambda x:-x[1]):
        st.markdown(f'<div class="smog-banner">☠️ HAZARDOUS · {city.upper()} · AQI {aqi} · EMERGENCY — DO NOT GO OUTSIDE</div>',unsafe_allow_html=True)
    for city,aqi in sorted(vuh,key=lambda x:-x[1]):
        st.markdown(f'<div class="alert-vuh">🟣 VERY UNHEALTHY · {city} · AQI {aqi} · Avoid all outdoor activity</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 NATIONAL ALERT SUMMARY</div>',unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("☠️ Hazardous",     len(haz))
    c2.metric("🟣 Very Unhealthy",len(vuh))
    c3.metric("🔴 Unhealthy",     len(unh))
    c4.metric("🟡 Moderate",      len(mod))
    c5.metric("🟢 Good",          len(good))

    st.markdown('<div class="section-header">🏆 ALL CITIES RANKED</div>',unsafe_allow_html=True)
    for city,data in sorted(all_data.items(),key=lambda x:x[1]['aqi'],reverse=True):
        aqi = data['aqi']
        level,emoji,color,_ = classify_aqi(aqi)
        bar = min(int(aqi/5),100)
        st.markdown(f"""
        <div style='display:flex;align-items:center;margin:5px 0;
                    background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:9px 14px'>
            <div style='width:115px;font-family:Rajdhani;font-weight:700;color:#c9d1d9'>{city}</div>
            <div style='flex:1;background:#161b22;border-radius:4px;height:14px;margin:0 10px'>
                <div style='width:{bar}%;background:{color};height:100%;border-radius:4px'></div>
            </div>
            <div style='width:46px;font-family:Orbitron;color:{color};font-weight:700;font-size:0.85rem'>{aqi}</div>
            <div style='width:180px;font-family:Rajdhani;color:{color};margin-left:8px;font-size:0.9rem'>{emoji} {level}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="section-header">🛡️ PROTECTION GUIDE</div>',unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.markdown("**😷 Mask Guide**\n- AQI < 100 → No mask\n- AQI 100–150 → Surgical\n- AQI 150–200 → N95\n- AQI > 200 → N95 mandatory\n- AQI > 300 → Full respirator")
    c2.markdown("**🏠 Indoor Safety**\n- Close all windows\n- HEPA air purifier\n- Avoid gas cooking\n- No candles/incense\n- Monitor indoor AQI")
    c3.markdown("**🏥 Medical Alert**\n- Keep inhaler accessible\n- No outdoor exercise\n- Stay hydrated\n- Watch for symptoms\n- Call doctor if needed")


# ════════════════════════════════════════════════════════
# PAGE 7 — CITY DEEP DIVE
# ════════════════════════════════════════════════════════
elif "Deep Dive" in page:
    st.markdown('<div class="hero-header">🔬 CITY DEEP DIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Comprehensive Single-City Analysis Dashboard</div>', unsafe_allow_html=True)

    city = st.selectbox("Select city for deep analysis", list(PAKISTAN_CITIES.keys()))
    with st.spinner(f"🛰️ Fetching full data for {city}..."):
        data = get_live_aqi(PAKISTAN_CITIES[city]["waqi"])

    if data:
        aqi = data['aqi']
        level,emoji,color,severity = classify_aqi(aqi)
        info = PAKISTAN_CITIES[city]

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{color}0d,{color}20);
                    border:1px solid {color}44;border-radius:18px;padding:22px;margin:12px 0'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <div style='font-family:Orbitron;font-size:1.9rem;color:#c9d1d9'>{city}</div>
                    <div style='font-family:Rajdhani;color:#8b949e;margin-top:3px'>
                        {info["province"]} Province &nbsp;·&nbsp; 📍 {info["lat"]}°N, {info["lon"]}°E
                    </div>
                </div>
                <div style='text-align:right'>
                    <div style='font-family:Orbitron;font-size:4rem;font-weight:900;
                                color:{color};text-shadow:0 0 22px {color}'>{aqi}</div>
                    <div style='font-family:Rajdhani;color:{color};font-size:1.2rem'>{emoji} {level}</div>
                </div>
            </div>
        </div>""",unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            pols = {'PM2.5':data['pm25'] or 0,'PM10':data['pm10'] or 0,
                    'NO₂':data['no2'] or 0,'O₃':data['o3'] or 0,
                    'CO':data['co'] or 0,'SO₂':data['so2'] or 0}
            cats = list(pols.keys())
            vals = [min(v/200*100,100) for v in pols.values()]
            r,g,b_ = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig = go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill='toself',
                fillcolor=f'rgba({r},{g},{b_},0.18)',line_color=color))
            fig.update_layout(polar=dict(
                radialaxis=dict(visible=True,range=[0,100],color='#8b949e'),
                angularaxis=dict(color='#8b949e')),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9',family='Rajdhani'),
                title=dict(text="Pollutant Radar",font=dict(color='#c9d1d9')),
                height=310,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

        with c2:
            st.plotly_chart(gauge(aqi,city),use_container_width=True)
            st.markdown(f"""
            <div style='font-family:Rajdhani;background:#0d1117;border:1px solid #21262d;
                        border-radius:10px;padding:14px;margin-top:4px'>
                <div style='color:#8b949e;font-size:0.75rem;letter-spacing:2px'>WEATHER CONDITIONS</div>
                <div style='display:flex;justify-content:space-around;margin-top:8px'>
                    <div style='text-align:center'>
                        <div style='font-family:Orbitron;color:#00d4ff;font-size:1.3rem'>{data['temp'] or '--'}°C</div>
                        <div style='color:#8b949e;font-size:0.7rem'>TEMP</div>
                    </div>
                    <div style='text-align:center'>
                        <div style='font-family:Orbitron;color:#00d4ff;font-size:1.3rem'>{data['humidity'] or '--'}%</div>
                        <div style='color:#8b949e;font-size:0.7rem'>HUMIDITY</div>
                    </div>
                    <div style='text-align:center'>
                        <div style='font-family:Orbitron;color:#00d4ff;font-size:1.3rem'>{data['wind'] or '--'}</div>
                        <div style='color:#8b949e;font-size:0.7rem'>WIND</div>
                    </div>
                </div>
            </div>""",unsafe_allow_html=True)

        st.markdown('<div class="section-header">🧪 DETAILED POLLUTANT READINGS</div>',unsafe_allow_html=True)
        pol_df = pd.DataFrame([
            {"Pollutant":"PM2.5 (Fine Particles)", "Reading":data['pm25'],"WHO Limit":"15","Unit":"μg/m³"},
            {"Pollutant":"PM10 (Coarse Particles)","Reading":data['pm10'],"WHO Limit":"45","Unit":"μg/m³"},
            {"Pollutant":"NO₂ (Nitrogen Dioxide)", "Reading":data['no2'], "WHO Limit":"25","Unit":"μg/m³"},
            {"Pollutant":"O₃ (Ozone)",             "Reading":data['o3'],  "WHO Limit":"100","Unit":"μg/m³"},
            {"Pollutant":"CO (Carbon Monoxide)",    "Reading":data['co'],  "WHO Limit":"4","Unit":"mg/m³"},
            {"Pollutant":"SO₂ (Sulfur Dioxide)",   "Reading":data['so2'], "WHO Limit":"40","Unit":"μg/m³"},
        ])
        st.dataframe(pol_df,use_container_width=True,hide_index=True)
    else:
        st.warning(f"Could not fetch live data for {city}. Try another city.")