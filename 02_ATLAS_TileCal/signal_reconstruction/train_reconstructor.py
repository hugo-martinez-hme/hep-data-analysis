import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


def train_atlas_model():
    data_path = "../data/tilecal_pulses.csv"
    if not os.path.exists(data_path):
        print("Error: No se encuentra el dataset. Ejecuta primero el generador.")
        return

    df = pd.read_csv(data_path)

    X = df[["s0", "s1", "s2", "s3", "s4", "s5", "s6"]]
    y = df["true_amplitude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("--- Entrenando Reconstructor de Señales ATLAS ---")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Modelo entrenado.")
    print(f"Error Medio Absoluto (MAE): {mae:.4f} GeV")
    print(f"Precisión (R2 Score): {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/tilecal_mlp_model.pkl")
    print("✅ Modelo guardado en: models/tilecal_mlp_model.pkl")


if __name__ == "__main__":
    train_atlas_model()
