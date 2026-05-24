import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ─────────────────────────────────────────
# STEP 1: Load and prepare data
# ─────────────────────────────────────────
def load_and_prepare():
    print("\n" + "="*50)
    print("  LOADING DATA FOR CLASSIFICATION")
    print("="*50)

    df = pd.read_csv("data/clean_aqi.csv")
    print(f"  ✅ Loaded {len(df)} rows")

    # Auto-detect AQI column
    aqi_candidates = ['aqi', 'AQI', 'pm25', 'PM25',
                      'city_avg_pm25', 'avg_aqi', 'mean_aqi',
                      'city_avg_pm25']
    aqi_col = None
    for col in aqi_candidates:
        if col in df.columns:
            aqi_col = col
            break
    if aqi_col is None:
        numeric = df.select_dtypes(include='number').columns
        aqi_col = [c for c in numeric if c not in
                   ['year', 'month', 'day', 'hour', 'risk_level',
                    'latitude', 'longitude', 'season']][0]

    print(f"  Using column: '{aqi_col}'")

    # Create risk level labels
    def get_risk(val):
        if val <= 50:   return 0  # Good
        elif val <= 100: return 1  # Moderate
        elif val <= 150: return 2  # Unhealthy Sensitive
        elif val <= 200: return 3  # Unhealthy
        elif val <= 300: return 4  # Very Unhealthy
        else:            return 5  # Hazardous

    def get_label(val):
        if val <= 50:   return 'Good'
        elif val <= 100: return 'Moderate'
        elif val <= 150: return 'Unhealthy-Sensitive'
        elif val <= 200: return 'Unhealthy'
        elif val <= 300: return 'Very Unhealthy'
        else:            return 'Hazardous'

    df['risk_level'] = df[aqi_col].apply(get_risk)
    df['risk_label'] = df[aqi_col].apply(get_label)

    print(f"\n  Risk level distribution:")
    print(df['risk_label'].value_counts())

    return df, aqi_col

# ─────────────────────────────────────────
# STEP 2: Build features for ML
# ─────────────────────────────────────────
def build_features(df, aqi_col):
    print("\n" + "="*50)
    print("  BUILDING FEATURES")
    print("="*50)

    feature_candidates = [
        aqi_col, 'year', 'month', 'hour',
        'day_of_week', 'is_weekend', 'temperature',
        'humidity', 'wind', 'pm25', 'pm10',
        'co', 'no2', 'o3', 'so2',
        'city_avg_pm25', 'pakistan_national_avg_pm25'
    ]

    # Only use columns that exist
    features = [f for f in feature_candidates if f in df.columns]
    print(f"  Features used: {features}")

    X = df[features].fillna(df[features].median())
    y = df['risk_level']

    return X, y, features

# ─────────────────────────────────────────
# STEP 3: Train Random Forest
# ─────────────────────────────────────────
def train_random_forest(X, y):
    print("\n" + "="*50)
    print("  TRAINING RANDOM FOREST")
    print("="*50)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        verbose=1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean() * 100

    print(f"\n  ✅ Accuracy: {accuracy:.2f}%")
    print("\n  Detailed Report:")
    present_labels = sorted(y_test.unique())
    label_names = ['Good', 'Moderate', 'Unhealthy-Sen',
                   'Unhealthy', 'Very Unhealthy', 'Hazardous']
    present_names = [label_names[i] for i in present_labels]
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    return model, X_test, y_test, y_pred

# ─────────────────────────────────────────
# STEP 4: Plot confusion matrix
# ─────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
    labels = ['Good','Moderate','Unhlthy-Sen',
              'Unhealthy','Very Unhlthy','Hazardous']
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels)
    plt.title('Risk Classification — Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png', dpi=150)
    plt.show()
    print("  ✅ Confusion matrix saved!")

# ─────────────────────────────────────────
# STEP 5: Feature importance chart
# ─────────────────────────────────────────
def plot_feature_importance(model, features):
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importance['feature'], importance['importance'],
             color='steelblue')
    plt.title('Feature Importance — What Drives AQI Risk?')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png', dpi=150)
    plt.show()
    print("  ✅ Feature importance chart saved!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    df, aqi_col = load_and_prepare()

    # Build features
    X, y, features = build_features(df, aqi_col)

    # Train model
    model, X_test, y_test, y_pred = train_random_forest(X, y)

    # Save feature list
    joblib.dump(features, "models/rf_features.pkl")

    # Plot results
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, features)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/risk_classifier.pkl")

    print("\n" + "="*50)
    print("  🎉 Random Forest trained & saved!")
    print("  models/risk_classifier.pkl")
    print("="*50)