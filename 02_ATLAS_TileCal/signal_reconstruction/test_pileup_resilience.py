import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score


def check_resilience():
    model = joblib.load("models/tilecal_hardened_model.pkl")
    data = pd.read_csv("../data/tilecal_pileup.csv")

    X = data[[f"s{i}" for i in range(7)]]
    y_true = data["true_amplitude"]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("--- RESULTADOS BAJO CONDICIONES DE PILE-UP ---")
    print(f"MAE: {mae:.4f} GeV")
    print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    check_resilience()
