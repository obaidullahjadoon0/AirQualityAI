import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# Try multiple ways to get token
WAQI_TOKEN = (
    os.getenv("WAQI_TOKEN") or
    "facf272642c8f4364de71f9b6da659a571d340fd"
)

# Fallback — hardcode temporarily to test
if not WAQI_TOKEN:
    WAQI_TOKEN = "facf272642c8f4364de71f9b6da659a571d340fd"

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AirGuard Pakistan 🇵🇰",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .good     { background: linear-gradient(135deg,#43e97b,#38f9d7); color:#1a1a1a; }
    .moderate { background: linear-gradient(135deg,#f9d423,#ff4e50); color:#1a1a1a; }
    .unhealthy{ background: linear-gradient(135deg,#f12711,#f5af19); }
    .hazardous{ background: linear-gradient(135deg,#200122,#6f0000); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def get_live_aqi(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data["status"] == "ok":
            d = data["data"]
            return {
                "city": city.title(),
                "aqi": d["aqi"],
                "pm25": d["iaqi"].get("pm25",{}).get("v"),
                "pm10": d["iaqi"].get("pm10",{}).get("v"),
                "co":   d["iaqi"].get("co",  {}).get("v"),
                "no2":  d["iaqi"].get("no2", {}).get("v"),
                "o3":   d["iaqi"].get("o3",  {}).get("v"),
                "temp": d["iaqi"].get("t",   {}).get("v"),
                "humidity": d["iaqi"].get("h",{}).get("v"),
                "time": datetime.now().strftime("%d %b %Y, %H:%M")
            }
    except:
        return None

def classify_aqi(aqi):
    if aqi is None: return "Unknown", "⚪", "#888888"
    elif aqi <= 50:  return "Good", "🟢", "#00e400"
    elif aqi <= 100: return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive", "🟠", "#ff7e00"
    elif aqi <= 200: return "Unhealthy", "🔴", "#ff0000"
    elif aqi <= 300: return "Very Unhealthy", "🟣", "#8f3f97"
    else:            return "Hazardous", "⚫", "#7e0023"

def get_aqi_gauge(aqi, city):
    level, emoji, color = classify_aqi(aqi)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        title={'text': f"{city} AQI", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': color},
            'steps': [
                {'range': [0,   50],  'color': '#00e400'},
                {'range': [50,  100], 'color': '#ffff00'},
                {'range': [100, 150], 'color': '#ff7e00'},
                {'range': [150, 200], 'color': '#ff0000'},
                {'range': [200, 300], 'color': '#8f3f97'},
                {'range': [300, 500], 'color': '#7e0023'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=20))
    return fig

# ─────────────────────────────────────────
# Pakistan City Coordinates
# ─────────────────────────────────────────
CITY_COORDS = {
    "Lahore":    {"lat": 31.5497, "lon": 74.3436},
    "Karachi":   {"lat": 24.8607, "lon": 67.0011},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Peshawar":  {"lat": 34.0150, "lon": 71.5249},
    "Quetta":    {"lat": 30.1798, "lon": 66.9750},
    "Multan":    {"lat": 30.1575, "lon": 71.5249},
}

# ─────────────────────────────────────────
# Pakistan Map Function
# ─────────────────────────────────────────
def show_pakistan_map(city_data):
    st.subheader("🗺️ Pakistan AQI Map")

    map_data = []
    for city, data in city_data.items():
        coords = CITY_COORDS.get(city, {})
        if coords:
            level, emoji, color = classify_aqi(data['aqi'])
            map_data.append({
                "City": city,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "AQI": data['aqi'],
                "Status": level,
                "Size": max(data['aqi'] / 5, 10)
            })

    if map_data:
        df_map = pd.DataFrame(map_data)
        fig = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            size="Size",
            color="AQI",
            hover_name="City",
            hover_data={
                "AQI": True,
                "Status": True,
                "lat": False,
                "lon": False,
                "Size": False
            },
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 300],
            mapbox_style="carto-positron",
            zoom=4.5,
            center={"lat": 30.3753, "lon": 69.3451},
            title="Live AQI Map — Pakistan"
        )
        fig.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select cities from the sidebar to see them on the map!")

# ─────────────────────────────────────────
# Health Score Calculator
# ─────────────────────────────────────────
def calculate_health_score(aqi, age_group, conditions):
    if aqi <= 50:    base = 100
    elif aqi <= 100: base = 80
    elif aqi <= 150: base = 60
    elif aqi <= 200: base = 40
    elif aqi <= 300: base = 20
    else:            base = 5

    deduction = 0
    if "Child" in age_group:         deduction += 15
    if "Elderly" in age_group:       deduction += 15
    if "Asthma" in conditions:       deduction += 20
    if "Heart Disease" in conditions: deduction += 20
    if "Pregnant" in conditions:     deduction += 15

    return max(0, base - deduction)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://flagcdn.com/w80/pk.png", width=60)
    st.title("🌫️ AirGuard")
    st.markdown("**AI-Powered Air Quality Monitor**")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Live Dashboard",
        "📈 AQI Forecast",
        "🏥 Health Risk Profiler",
        "📊 Historical Trends"
    ])

    st.divider()
    st.markdown("**Select Cities:**")
    cities = st.multiselect(
        "Cities to monitor",
        ["Lahore", "Karachi", "Islamabad",
         "Peshawar", "Quetta", "Multan"],
        default=["Lahore", "Karachi", "Islamabad"]
    )

    st.divider()
    st.markdown("**Your Health Profile:**")
    age_group = st.selectbox("Age Group", [
        "Adult (18-59)", "Child (under 12)",
        "Elderly (60+)"
    ])
    conditions = st.multiselect("Health Conditions", [
        "Asthma", "Heart Disease",
        "Pregnant", "Diabetes", "None"
    ], default=["None"])

# ─────────────────────────────────────────
# PAGE 1: LIVE DASHBOARD
# ─────────────────────────────────────────
if "Live Dashboard" in page:
    st.markdown('<p class="main-header">🌫️ AirGuard Pakistan</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time AI-powered air quality monitoring</p>',
                unsafe_allow_html=True)

    if not cities:
        st.warning("Please select at least one city from the sidebar!")
    else:
        # Fetch data for all cities
        with st.spinner("Fetching live AQI data..."):
            city_data = {}
            for city in cities:
                data = get_live_aqi(city.lower())
                if data:
                    city_data[city] = data

        if not city_data:
            st.error("Could not fetch data. Check your WAQI token in .env file.")
        else:
            # Show last updated time
            st.caption(f"🕐 Last updated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")

            # AQI Gauges
            st.subheader("📊 Live AQI Gauges")
            cols = st.columns(len(city_data))
            for i, (city, data) in enumerate(city_data.items()):
                with cols[i]:
                    level, emoji, color = classify_aqi(data['aqi'])
                    st.plotly_chart(
                        get_aqi_gauge(data['aqi'], city),
                        use_container_width=True
                    )
                    st.markdown(f"**{emoji} {level}**")

            st.divider()

            # Detailed metrics
            st.subheader("🔬 Pollutant Breakdown")
            for city, data in city_data.items():
                level, emoji, color = classify_aqi(data['aqi'])
                with st.expander(
                    f"{emoji} {city} — AQI {data['aqi']} ({level})",
                    expanded=True
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("PM2.5", f"{data['pm25'] or 'N/A'}")
                    c2.metric("PM10",  f"{data['pm10'] or 'N/A'}")
                    c3.metric("NO2",   f"{data['no2']  or 'N/A'}")
                    c4.metric("Temp",  f"{data['temp'] or 'N/A'}°C")

                    # Health advice
                    st.markdown("**💊 Health Advice:**")
                    aqi = data['aqi']
                    if aqi <= 50:
                        st.success("Air quality is great! Safe for everyone.")
                    elif aqi <= 100:
                        st.info("Moderate air quality. Sensitive groups should take care.")
                    elif aqi <= 150:
                        st.warning("Unhealthy for sensitive groups. Elderly and kids should limit outdoor time.")
                    elif aqi <= 200:
                        st.error("Unhealthy for everyone. Wear N95 mask outdoors.")
                    else:
                        st.error("🚨 HAZARDOUS! Stay indoors immediately.")

# ─────────────────────────────────────────
# PAGE 2: AQI FORECAST
# ─────────────────────────────────────────
elif "Forecast" in page:
    st.title("📈 24-Hour AQI Forecast")
    st.markdown("AI prediction using LSTM deep learning model")

    city = st.selectbox("Select city to forecast", cities if cities else ["Lahore"])

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Running LSTM model..."):
            # Load model if exists
            model_path = "models/best_lstm.keras"
            scaler_path = "models/aqi_scaler.pkl"

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)

                # Get live data as seed
                live = get_live_aqi(city.lower())
                if live and live['aqi']:
                    current_aqi = live['aqi']

                    # Generate 24-hour forecast
                    seed = np.array([current_aqi] * 12).reshape(-1, 1)
                    seed_scaled = scaler.transform(seed)

                    forecasts = []
                    input_seq = seed_scaled.copy()

                    for _ in range(24):
                        pred = model.predict(
                            input_seq.reshape(1, 12, 1), verbose=0
                        )
                        forecasts.append(pred[0][0])
                        input_seq = np.append(
                            input_seq[1:], pred
                        ).reshape(-1, 1)

                    forecast_aqi = scaler.inverse_transform(
                        np.array(forecasts).reshape(-1, 1)
                    ).flatten()
                    forecast_aqi = np.clip(forecast_aqi, 0, 500)

                    # Plot forecast
                    hours = [f"+{i}h" for i in range(24)]
                    colors = [classify_aqi(v)[2] for v in forecast_aqi]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hours, y=forecast_aqi,
                        mode='lines+markers',
                        line=dict(color='royalblue', width=3),
                        marker=dict(size=8, color=colors),
                        name='Predicted AQI'
                    ))
                    fig.add_hline(y=100, line_dash="dash",
                                  line_color="orange",
                                  annotation_text="Moderate threshold")
                    fig.add_hline(y=150, line_dash="dash",
                                  line_color="red",
                                  annotation_text="Unhealthy threshold")
                    fig.update_layout(
                        title=f"24-Hour AQI Forecast — {city}",
                        xaxis_title="Hours from now",
                        yaxis_title="Predicted AQI",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary
                    max_aqi = forecast_aqi.max()
                    avg_aqi = forecast_aqi.mean()
                    level, emoji, _ = classify_aqi(int(avg_aqi))

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current AQI", current_aqi)
                    col2.metric("Avg Forecast", f"{avg_aqi:.0f}")
                    col3.metric("Peak AQI", f"{max_aqi:.0f}")

                    st.info(f"**Forecast Summary:** Over the next 24 hours, {city} air quality is expected to be **{level}** on average {emoji}")
            else:
                st.warning("LSTM model not found! Run train_lstm.py first.")
                st.info("Showing demo forecast instead...")

                # Demo forecast
                demo = np.random.randint(80, 200, 24)
                hours = [f"+{i}h" for i in range(24)]
                fig = px.line(x=hours, y=demo,
                              title=f"Demo Forecast — {city}",
                              labels={'x':'Hours','y':'AQI'})
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3: HEALTH RISK PROFILER
# ─────────────────────────────────────────
elif "Health Risk" in page:
    st.title("🏥 Personal Health Risk Profiler")
    st.markdown("Get personalized health advice based on your profile")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Your city", cities if cities else ["Lahore"])
    with col2:
        manual_aqi = st.number_input(
            "Or enter AQI manually", 0, 500, 0
        )

    if st.button("🔍 Assess My Risk", type="primary"):
        with st.spinner("Analyzing..."):
            if manual_aqi > 0:
                aqi = manual_aqi
            else:
                data = get_live_aqi(city.lower())
                aqi = data['aqi'] if data else 150

            level, emoji, color = classify_aqi(aqi)

            # Big AQI display
            st.markdown(f"""
            <div style='background:{color};padding:30px;
                        border-radius:20px;text-align:center;
                        color:white;margin:20px 0'>
                <h1>{emoji} AQI: {aqi}</h1>
                <h2>{level}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Personalized risk
            st.subheader("👤 Your Personal Risk Assessment")

            profile_risks = []

            # Age group risk
            if "Child" in age_group:
                threshold = 100
                profile_risks.append(("Child", threshold))
            elif "Elderly" in age_group:
                threshold = 100
                profile_risks.append(("Elderly", threshold))
            else:
                threshold = 150
                profile_risks.append(("Adult", threshold))

            # Health conditions
            for condition in conditions:
                if condition == "Asthma":
                    profile_risks.append(("Asthma Patient", 50))
                elif condition == "Heart Disease":
                    profile_risks.append(("Heart Patient", 50))
                elif condition == "Pregnant":
                    profile_risks.append(("Pregnant", 50))

            for profile, thresh in profile_risks:
                if aqi >= thresh * 2:
                    st.error(f"🆘 **{profile}**: HIGH DANGER — Stay indoors!")
                elif aqi >= thresh:
                    st.warning(f"⚠️ **{profile}**: AT RISK — Take precautions")
                else:
                    st.success(f"✅ **{profile}**: SAFE — Air quality acceptable")

            score = calculate_health_score(aqi, age_group, conditions)
            st.subheader("🎯 Your Personal Safety Score")
            st.metric("Safety Score", f"{score}/100")
            st.progress(score / 100)
            if score > 70:
                st.success("You are relatively safe in current conditions")
            elif score > 40:
                st.warning("Take precautions — moderate risk for you personally")
            else:
                st.error("High personal risk — strongly advised to stay indoors")
            st.divider()

            # Detailed advice
            st.subheader("📋 What You Should Do Right Now")
            if aqi <= 50:
                st.success("🌿 Enjoy outdoor activities freely!")
            elif aqi <= 100:
                st.info("😷 Sensitive individuals should consider a mask outdoors")
            elif aqi <= 150:
                st.warning("🏠 Limit outdoor time. Close windows. Consider air purifier.")
            elif aqi <= 200:
                st.error("😷 Wear N95 mask. Avoid outdoor exercise. Keep windows shut.")
            else:
                st.error("🆘 STAY INDOORS! Seal windows. Use air purifier immediately.")

# ─────────────────────────────────────────
# PAGE 4: HISTORICAL TRENDS
# ─────────────────────────────────────────
elif "Historical" in page:
    st.title("📊 Historical AQI Trends")

    if os.path.exists("data/clean_aqi.csv"):
        df = pd.read_csv("data/clean_aqi.csv")

        # Auto detect columns
        aqi_col = None
        for c in ['city_avg_pm25','aqi','pm25','AQI']:
            if c in df.columns:
                aqi_col = c
                break

        if aqi_col and 'year' in df.columns:
            # Yearly trend
            yearly = df.groupby('year')[aqi_col].mean().reset_index()
            fig1 = px.bar(yearly, x='year', y=aqi_col,
                          title='Average Annual Air Quality',
                          color=aqi_col,
                          color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig1, use_container_width=True)

            # Monthly pattern
            if 'month' in df.columns:
                monthly = df.groupby('month')[aqi_col].mean().reset_index()
                month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                               'Jul','Aug','Sep','Oct','Nov','Dec']
                monthly['month_name'] = monthly['month'].apply(
                    lambda x: month_names[x-1]
                )
                fig2 = px.line(monthly, x='month_name', y=aqi_col,
                               title='Average Monthly Air Quality Pattern',
                               markers=True)
                st.plotly_chart(fig2, use_container_width=True)

            # City comparison if available
            if 'city' in df.columns:
                city_avg = df.groupby('city')[aqi_col].mean().reset_index()
                fig3 = px.bar(city_avg.sort_values(aqi_col, ascending=False),
                              x='city', y=aqi_col,
                              title='Average Air Quality by City',
                              color=aqi_col,
                              color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Limited columns available for trend analysis")
            st.dataframe(df.head(20))
    else:
        st.warning("No historical data found. Run data_cleaning.py first.")