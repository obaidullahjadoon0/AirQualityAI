import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OWM_API_KEY = os.getenv("OWM_API_KEY")


# ─────────────────────────────────────────
# FUNCTION 1: Get Live AQI Data (WAQI API)
# ─────────────────────────────────────────
def get_live_aqi(city="lahore"):
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        print(f"Error fetching AQI for {city}")
        return None

    aqi_data = {
        "city": city,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": data["data"]["aqi"],
        "pm25": data["data"]["iaqi"].get("pm25", {}).get("v", None),
        "pm10": data["data"]["iaqi"].get("pm10", {}).get("v", None),
        "co":   data["data"]["iaqi"].get("co",   {}).get("v", None),
        "no2":  data["data"]["iaqi"].get("no2",  {}).get("v", None),
        "o3":   data["data"]["iaqi"].get("o3",   {}).get("v", None),
        "so2":  data["data"]["iaqi"].get("so2",  {}).get("v", None),
        "temperature": data["data"]["iaqi"].get("t", {}).get("v", None),
        "humidity":    data["data"]["iaqi"].get("h", {}).get("v", None),
        "wind":        data["data"]["iaqi"].get("w", {}).get("v", None),
    }

    return aqi_data


# ─────────────────────────────────────────
# FUNCTION 2: Classify AQI Risk Level
# ─────────────────────────────────────────
def classify_aqi(aqi):
    if aqi is None:
        return "Unknown", "⚪"
    elif aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Moderate", "🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "⚫"


# ─────────────────────────────────────────
# FUNCTION 3: Get Health Recommendations
# ─────────────────────────────────────────
def get_health_advice(aqi):
    level, emoji = classify_aqi(aqi)

    advice = {
        "Good": {
            "general": "Air quality is great! Perfect for outdoor activities.",
            "elderly": "Safe for all activities.",
            "kids": "Safe for all activities.",
            "asthma": "Safe for all activities."
        },
        "Moderate": {
            "general": "Air quality is acceptable. Unusually sensitive people should limit prolonged outdoor exertion.",
            "elderly": "Consider reducing long outdoor activities.",
            "kids": "Fine for most activities.",
            "asthma": "Watch for symptoms if spending time outdoors."
        },
        "Unhealthy for Sensitive Groups": {
            "general": "Members of sensitive groups may experience health effects.",
            "elderly": "⚠️ Limit outdoor activities. Stay indoors if possible.",
            "kids": "⚠️ Reduce outdoor playtime. Keep windows closed.",
            "asthma": "⚠️ Carry inhaler. Avoid outdoor exercise."
        },
        "Unhealthy": {
            "general": "Everyone may begin to experience health effects.",
            "elderly": "🚨 Stay indoors. Wear N95 mask if going out.",
            "kids": "🚨 No outdoor activities. Keep windows shut.",
            "asthma": "🚨 Stay indoors. Have emergency medication ready."
        },
        "Very Unhealthy": {
            "general": "Health alert — everyone may experience serious effects.",
            "elderly": "🚨 Do NOT go outside. Air purifier recommended.",
            "kids": "🚨 School should consider indoor-only activities.",
            "asthma": "🚨 Emergency risk. Avoid any outdoor exposure."
        },
        "Hazardous": {
            "general": "EMERGENCY CONDITIONS. Entire population is affected.",
            "elderly": "🆘 Stay inside, seal windows, use air purifier.",
            "kids": "🆘 Absolutely no outdoor exposure.",
            "asthma": "🆘 Seek medical attention if experiencing symptoms."
        }
    }

    return level, emoji, advice.get(level, {})


# ─────────────────────────────────────────
# MAIN — Run & Print Results
# ─────────────────────────────────────────
if __name__ == "__main__":
    cities = ["lahore", "karachi", "islamabad"]

    for city in cities:
        print(f"\n{'='*50}")
        print(f"  Fetching AQI data for: {city.upper()}")
        print(f"{'='*50}")

        data = get_live_aqi(city)

        if data:
            level, emoji, advice = get_health_advice(data["aqi"])

            print(f"  Timestamp  : {data['timestamp']}")
            print(f"  AQI        : {data['aqi']}  {emoji} {level}")
            print(f"  PM2.5      : {data['pm25']}")
            print(f"  PM10       : {data['pm10']}")
            print(f"  CO         : {data['co']}")
            print(f"  NO2        : {data['no2']}")
            print(f"  O3         : {data['o3']}")
            print(f"  Temperature: {data['temperature']}°C")
            print(f"  Humidity   : {data['humidity']}%")
            print(f"\n  📋 General Advice : {advice.get('general', 'N/A')}")
            print(f"  👴 Elderly        : {advice.get('elderly', 'N/A')}")
            print(f"  👶 Kids           : {advice.get('kids', 'N/A')}")
            print(f"  🫁 Asthma Patients: {advice.get('asthma', 'N/A')}")

    # Save data to CSV for later use in ML training
    print(f"\n{'='*50}")
    print("  Saving data to data/live_aqi.csv ...")
    records = []
    for city in cities:
        d = get_live_aqi(city)
        if d:
            records.append(d)

    df = pd.DataFrame(records)
    df.to_csv("data/live_aqi.csv", index=False)
    print("  ✅ Saved successfully!")
    print(f"{'='*50}\n")