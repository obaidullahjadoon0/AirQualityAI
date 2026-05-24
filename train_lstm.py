import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os


# ─────────────────────────────────────────
# STEP 1: Load prepared data
# ─────────────────────────────────────────
def load_prepared_data():
    print("\n" + "=" * 50)
    print("  LOADING PREPARED DATA")
    print("=" * 50)

    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape : {X_test.shape}")
    print(f"  ✅ Data loaded successfully")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# STEP 2: Build LSTM model
# ─────────────────────────────────────────
def build_lstm_model(sequence_length=12):
    print("\n" + "=" * 50)
    print("  BUILDING LSTM MODEL")
    print("=" * 50)

    model = Sequential([
        # First LSTM layer — learns basic patterns
        LSTM(64,
             return_sequences=True,
             input_shape=(sequence_length, 1)),
        Dropout(0.2),  # prevents overfitting

        # Second LSTM layer — learns complex patterns
        LSTM(32,
             return_sequences=False),
        Dropout(0.2),

        # Dense layers — makes final prediction
        Dense(16, activation='relu'),
        Dense(1)  # output: single AQI value
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    model.summary()
    return model


# ─────────────────────────────────────────
# STEP 3: Train the model
# ─────────────────────────────────────────
def train_model(model, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 50)
    print("  TRAINING LSTM MODEL")
    print("  This may take a few minutes...")
    print("=" * 50)

    # Early stopping — stops if model stops improving
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Save best model automatically
    checkpoint = ModelCheckpoint(
        'models/best_lstm.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    return history


# ─────────────────────────────────────────
# STEP 4: Evaluate the model
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    print("\n" + "=" * 50)
    print("  EVALUATING MODEL")
    print("=" * 50)

    # Make predictions
    y_pred_scaled = model.predict(X_test)

    # Convert back to real AQI values
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate errors
    mae = np.mean(np.abs(y_pred - y_actual))
    rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    print(f"\n  📊 Model Performance:")
    print(f"  MAE  (avg error)     : {mae:.2f} AQI points")
    print(f"  RMSE (root mean sq)  : {rmse:.2f} AQI points")
    print(f"  MAPE (% error)       : {mape:.2f}%")

    if mae < 20:
        print("\n  🎉 Excellent model! MAE under 20 AQI points")
    elif mae < 40:
        print("\n  ✅ Good model! Acceptable accuracy")
    else:
        print("\n  ⚠️ Model needs more data or tuning")

    return y_pred, y_actual


# ─────────────────────────────────────────
# STEP 5: Plot results
# ─────────────────────────────────────────
def plot_results(history, y_pred, y_actual):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1 — Training loss curve
    axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Model Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2 — Predicted vs Actual AQI
    axes[1].plot(y_actual[:50], label='Actual AQI', color='blue', marker='o')
    axes[1].plot(y_pred[:50], label='Predicted AQI', color='red',
                 linestyle='--', marker='x')
    axes[1].set_title('Predicted vs Actual AQI')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('AQI Value')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('data/lstm_results.png', dpi=150)
    plt.show()
    print("\n  ✅ Results chart saved to data/lstm_results.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_prepared_data()

    # Build model
    model = build_lstm_model(sequence_length=X_train.shape[1])

    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Load scaler
    scaler = joblib.load("models/aqi_scaler.pkl")

    # Evaluate
    y_pred, y_actual = evaluate_model(model, X_test, y_test, scaler)

    # Plot
    plot_results(history, y_pred, y_actual)

    print("\n" + "=" * 50)
    print("  🎉 Day 3 Complete!")
    print("  Model saved to: models/best_lstm.keras")
    print("  Results saved to: data/lstm_results.png")
    print("=" * 50)