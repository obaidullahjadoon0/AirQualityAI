import pandas as pd
import matplotlib.pyplot as plt

# Load clean data
df = pd.read_csv("data/clean_aqi.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# Auto-detect AQI column
aqi_candidates = ['aqi', 'AQI', 'pm25', 'PM25', 'avg_aqi',
                  'mean_aqi', 'city_avg_pm25', 'pm2_5', 'value']
aqi_col = None
for col in aqi_candidates:
    if col in df.columns:
        aqi_col = col
        break

# If still not found, use first numeric column
if aqi_col is None:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    # Skip year and month columns
    for col in numeric_cols:
        if col not in ['year', 'month', 'day', 'hour']:
            aqi_col = col
            break

print(f"\n✅ Using column: '{aqi_col}' for AQI analysis")

# ── Plot 1: AQI Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df[aqi_col].hist(bins=30, color='steelblue', edgecolor='black')
plt.title(f'{aqi_col} Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# ── Plot 2: AQI Category if exists, else value counts
plt.subplot(1, 2, 2)
if 'aqi_category' in df.columns:
    df['aqi_category'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=['green', 'yellow', 'orange', 'red', 'purple', 'black']
    )
    plt.title('AQI Category Breakdown')
else:
    df[aqi_col].plot.hist(bins=20, color='orange')
    plt.title(f'{aqi_col} Histogram')
plt.ylabel('')

plt.tight_layout()
plt.savefig('data/aqi_distribution.png')
plt.show()
print("✅ Chart saved to data/aqi_distribution.png")

# ── Plot 3: Trend over time
if 'year' in df.columns and 'month' in df.columns:
    plt.figure(figsize=(14, 5))
    # Group by year and month
    if 'city' in df.columns:
        # Plot for Lahore specifically
        lahore = df[df['city'].str.lower().str.contains('lahore', na=False)]
        if len(lahore) > 0:
            plt.plot(lahore[aqi_col].values, color='red', alpha=0.7)
            plt.title('Lahore AQI Trend Over Time')
        else:
            monthly_avg = df.groupby(['year', 'month'])[aqi_col].mean()
            plt.plot(monthly_avg.values, color='red', alpha=0.7)
            plt.title('Average AQI Trend Over Time')
    else:
        plt.plot(df[aqi_col].values, color='red', alpha=0.7)
        plt.title('AQI Trend Over Time')

    plt.xlabel('Time Steps')
    plt.ylabel('AQI / PM2.5 Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/aqi_timeline.png')
    plt.show()
    print("✅ Timeline chart saved!")

print("\n🎉 Data exploration complete!")