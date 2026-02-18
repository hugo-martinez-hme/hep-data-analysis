import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os


def train_hardened():
    data_path = "../data/tilecal_hardened_data.csv"
    df = pd.read_csv(data_path)

    X = df[[f"s{i}" for i in range(7)]]
    y = df["true_amplitude"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Mantenemos la misma arquitectura
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

    print("--- ENTRENANDO MODELO HARDENED (RESISTENTE AL PILE-UP) ---")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f} GeV")
    print(f"R2 Score: {r2_score(y_test, preds):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/tilecal_hardened_model.pkl")
    print("Modelo resiliente guardado.")


if __name__ == "__main__":
    train_hardened()
