import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# ─────────────────────────────────────────
# STEP 1: Load clean data
# ─────────────────────────────────────────
def load_clean_data():
    print("\n" + "=" * 50)
    print("  LOADING CLEAN DATA")
    print("=" * 50)

    df = pd.read_csv("data/clean_aqi.csv")
    print(f"  ✅ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────
# STEP 2: Find the AQI column
# ─────────────────────────────────────────
def find_aqi_column(df):
    possible = ['aqi', 'AQI', 'pm25', 'PM25', 'avg_aqi', 'mean_aqi', 'aqi_value']
    for col in possible:
        if col in df.columns:
            print(f"  ✅ Using '{col}' as target column")
            return col
    # If none found, show all columns and pick first numeric
    print(f"  Columns available: {list(df.columns)}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Using '{numeric_cols[0]}' as target column")
    return numeric_cols[0]


# ─────────────────────────────────────────
# STEP 3: Create sequences for LSTM
# ─────────────────────────────────────────
def create_sequences(data, sequence_length=12):
    """
    LSTM needs sequences — it looks at past N steps to predict next step.
    sequence_length=12 means: look at 12 months to predict next month
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# STEP 4: Prepare everything
# ─────────────────────────────────────────
def prepare_data(sequence_length=12):
    df = load_clean_data()

    # Find AQI column
    aqi_col = find_aqi_column(df)

    # Extract AQI values
    aqi_values = df[aqi_col].dropna().values.reshape(-1, 1)
    print(f"\n  AQI range: {aqi_values.min():.1f} - {aqi_values.max():.1f}")
    print(f"  Total data points: {len(aqi_values)}")

    # Scale values between 0 and 1 (LSTM works best this way)
    scaler = MinMaxScaler(feature_range=(0, 1))
    aqi_scaled = scaler.fit_transform(aqi_values)

    # Create sequences
    X, y = create_sequences(aqi_scaled, sequence_length)
    print(f"\n  Sequence length: {sequence_length}")
    print(f"  Total sequences: {len(X)}")

    # Split into train (80%) and test (20%)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")

    # Save scaler for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/aqi_scaler.pkl")
    print("\n  ✅ Scaler saved to models/aqi_scaler.pkl")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Save arrays for training
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)

    print("\n" + "=" * 50)
    print("  ✅ Data preparation complete!")
    print("  Saved: X_train, X_test, y_train, y_test")
    print("=" * 50)