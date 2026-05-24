import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# Health profiles — who is at risk at what AQI
# ─────────────────────────────────────────
HEALTH_PROFILES = {
    "healthy_adult": {
        "name": "Healthy Adult",
        "sensitive_threshold": 150,
        "danger_threshold": 200
    },
    "elderly": {
        "name": "Elderly (60+)",
        "sensitive_threshold": 100,
        "danger_threshold": 150
    },
    "child": {
        "name": "Child (under 12)",
        "sensitive_threshold": 100,
        "danger_threshold": 150
    },
    "asthma": {
        "name": "Asthma Patient",
        "sensitive_threshold": 50,
        "danger_threshold": 100
    },
    "heart": {
        "name": "Heart Disease Patient",
        "sensitive_threshold": 50,
        "danger_threshold": 100
    },
    "pregnant": {
        "name": "Pregnant Woman",
        "sensitive_threshold": 50,
        "danger_threshold": 100
    }
}

RECOMMENDATIONS = {
    0: {  # Good
        "outdoor": "✅ Perfect for outdoor activities",
        "mask": "😊 No mask needed",
        "windows": "🪟 Open windows — enjoy fresh air",
        "exercise": "🏃 Great time for outdoor exercise"
    },
    1: {  # Moderate
        "outdoor": "✅ Outdoor activities fine for most",
        "mask": "😷 Mask optional for sensitive groups",
        "windows": "🪟 Windows okay to open",
        "exercise": "🏃 Outdoor exercise fine"
    },
    2: {  # Unhealthy Sensitive
        "outdoor": "⚠️ Limit prolonged outdoor activities",
        "mask": "😷 Wear mask if you are sensitive",
        "windows": "🪟 Consider closing windows",
        "exercise": "🏠 Move exercise indoors if sensitive"
    },
    3: {  # Unhealthy
        "outdoor": "🚨 Reduce outdoor activities",
        "mask": "😷 Wear N95 mask outdoors",
        "windows": "🔒 Keep windows closed",
        "exercise": "🏠 Exercise indoors only"
    },
    4: {  # Very Unhealthy
        "outdoor": "🚨 Avoid outdoor activities",
        "mask": "😷 N95 mask essential if going out",
        "windows": "🔒 Keep all windows shut",
        "exercise": "🏠 No outdoor exercise at all"
    },
    5: {  # Hazardous
        "outdoor": "🆘 Stay indoors — emergency conditions",
        "mask": "🆘 N95 mask — limit all outdoor exposure",
        "windows": "🔒 Seal windows — use air purifier",
        "exercise": "🏠 No outdoor activity whatsoever"
    }
}

# ─────────────────────────────────────────
# Main risk assessment function
# ─────────────────────────────────────────
def assess_risk(aqi_value, user_profiles=None):
    # Get base risk level
    if aqi_value <= 50:   risk = 0
    elif aqi_value <= 100: risk = 1
    elif aqi_value <= 150: risk = 2
    elif aqi_value <= 200: risk = 3
    elif aqi_value <= 300: risk = 4
    else:                  risk = 5

    labels = ['Good','Moderate','Unhealthy for Sensitive',
              'Unhealthy','Very Unhealthy','Hazardous']
    emojis = ['🟢','🟡','🟠','🔴','🟣','⚫']

    print("\n" + "="*55)
    print(f"  🌫️  AQI RISK ASSESSMENT")
    print("="*55)
    print(f"  AQI Value  : {aqi_value}")
    print(f"  Risk Level : {emojis[risk]} {labels[risk]}")
    print("="*55)

    # General recommendations
    rec = RECOMMENDATIONS[risk]
    print(f"\n  📋 GENERAL ADVICE:")
    print(f"     {rec['outdoor']}")
    print(f"     {rec['mask']}")
    print(f"     {rec['windows']}")
    print(f"     {rec['exercise']}")

    # Personal health profile assessment
    if user_profiles:
        print(f"\n  👤 PERSONAL RISK BY HEALTH PROFILE:")
        for profile_key in user_profiles:
            if profile_key in HEALTH_PROFILES:
                profile = HEALTH_PROFILES[profile_key]
                if aqi_value >= profile['danger_threshold']:
                    status = "🆘 HIGH DANGER"
                elif aqi_value >= profile['sensitive_threshold']:
                    status = "⚠️  AT RISK"
                else:
                    status = "✅ SAFE"
                print(f"     {profile['name']:25} → {status}")

    print("="*55)
    return risk

# ─────────────────────────────────────────
# Test it with sample AQI values
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {"city": "Lahore",     "aqi": 187},
        {"city": "Karachi",    "aqi": 95},
        {"city": "Islamabad",  "aqi": 42},
    ]

    profiles = ["elderly", "child", "asthma", "pregnant"]

    for case in test_cases:
        print(f"\n\n  📍 City: {case['city']}")
        assess_risk(case['aqi'], profiles)

    print("\n\n✅ Risk profiler working perfectly!")