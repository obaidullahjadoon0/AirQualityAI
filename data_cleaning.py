import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────
def load_data(filepath):
    print(f"\n{'=' * 50}")
    print(f"  Loading data from: {filepath}")
    print(f"{'=' * 50}")

    df = pd.read_csv(filepath)

    print(f"  ✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────
# STEP 2: Inspect the data
# ─────────────────────────────────────────
def inspect_data(df):
    print(f"\n{'=' * 50}")
    print("  DATA INSPECTION")
    print(f"{'=' * 50}")

    print(f"\n  Shape: {df.shape}")
    print(f"\n  First 5 rows:")
    print(df.head())

    print(f"\n  Data types:")
    print(df.dtypes)

    print(f"\n  Missing values per column:")
    print(df.isnull().sum())

    print(f"\n  Basic statistics:")
    print(df.describe())


# ─────────────────────────────────────────
# STEP 3: Clean the data
# ─────────────────────────────────────────
def clean_data(df):
    print(f"\n{'=' * 50}")
    print("  CLEANING DATA")
    print(f"{'=' * 50}")

    original_rows = len(df)

    # 1. Drop duplicate rows
    df = df.drop_duplicates()
    print(f"  Removed duplicates: {original_rows - len(df)} rows dropped")

    # 2. Handle missing values
    # Fill numeric columns with median (safer than mean for AQI)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {missing} missing values in '{col}' with median ({median_val:.2f})")

    # 3. Convert date column to datetime
    date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'time', 'Time']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.dropna(subset=[col])
            df = df.sort_values(by=col)
            df = df.rename(columns={col: 'datetime'})
            print(f"  ✅ Converted '{col}' to datetime format")
            break

    # 4. Remove extreme outliers in AQI (AQI can't be negative or above 1000)
    aqi_columns = ['aqi', 'AQI', 'pm25', 'PM25', 'pm2.5', 'PM2.5']
    for col in aqi_columns:
        if col in df.columns:
            before = len(df)
            df = df[(df[col] >= 0) & (df[col] <= 1000)]
            removed = before - len(df)
            if removed > 0:
                print(f"  Removed {removed} outlier rows in '{col}'")

    # 5. Add AQI category column
    if 'aqi' in df.columns or 'AQI' in df.columns:
        aqi_col = 'aqi' if 'aqi' in df.columns else 'AQI'
        df['aqi_category'] = df[aqi_col].apply(categorize_aqi)
        df['risk_level'] = df[aqi_col].apply(risk_level)
        print(f"  ✅ Added 'aqi_category' and 'risk_level' columns")

    print(f"\n  ✅ Cleaning complete! {len(df)} rows remaining")
    return df


# ─────────────────────────────────────────
# Helper: Categorize AQI
# ─────────────────────────────────────────
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


def risk_level(aqi):
    if aqi <= 50:
        return 0  # Low
    elif aqi <= 100:
        return 1  # Moderate
    elif aqi <= 150:
        return 2  # High for sensitive
    elif aqi <= 200:
        return 3  # High
    elif aqi <= 300:
        return 4  # Very High
    else:
        return 5  # Extreme


# ─────────────────────────────────────────
# STEP 4: Add time features for ML
# ─────────────────────────────────────────
def add_time_features(df):
    print(f"\n{'=' * 50}")
    print("  ADDING TIME FEATURES FOR ML")
    print(f"{'=' * 50}")

    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Season (for Pakistan climate)
        df['season'] = df['month'].apply(lambda m:
                                         'Winter' if m in [12, 1, 2] else
                                         'Spring' if m in [3, 4, 5] else
                                         'Summer' if m in [6, 7, 8] else 'Autumn'
                                         )

        print("  ✅ Added: hour, day, month, year, day_of_week, is_weekend, season")

    return df


# ─────────────────────────────────────────
# STEP 5: Save clean data
# ─────────────────────────────────────────
def save_clean_data(df, output_path):
    print(f"\n{'=' * 50}")
    print("  SAVING CLEAN DATA")
    print(f"{'=' * 50}")

    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved to: {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # ── Option A: Use Kaggle dataset
    # Change this to your actual filename inside data/ folder
    input_file = "data/pakistan_air_quality_monthly_2015_2025.csv"

    # ── Option B: Use our own collected live data
    # input_file = "data/live_aqi.csv"

    output_file = "data/clean_aqi.csv"

    if not os.path.exists(input_file):
        print(f"\n❌ File not found: {input_file}")
        print("Using live_aqi.csv instead (collected yesterday)...")
        input_file = "data/live_aqi.csv"

    df = load_data(input_file)
    inspect_data(df)
    df = clean_data(df)
    df = add_time_features(df)
    save_clean_data(df, output_file)

    print(f"\n{'=' * 50}")
    print("  🎉 Day 2 Complete! Clean data ready for ML!")
    print(f"{'=' * 50}\n")