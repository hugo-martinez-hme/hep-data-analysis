import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


# --- 1. Definición de la Física del Detector ---
def pulse_shape_physics(t, amplitude, tau=25.0):
    """
    Ecuación teórica del pulso de ATLAS TileCal.
    Representa la respuesta ideal del calorímetro.
    """
    amp_clipped = torch.clamp(amplitude, min=0.0)
    return amp_clipped * (t / tau) * torch.exp(1 - (t / tau))


# --- 2. Función de Pérdida "Physics-Informed" (PINN) ---
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambd=0.05):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambd = lambd
        self.t_values = torch.tensor(
            [0, 25, 50, 75, 100, 125, 150], dtype=torch.float32
        )

    def forward(self, pred_amplitude, target_amplitude, input_samples):
        # A. Pérdida de Datos
        data_loss = self.mse(pred_amplitude, target_amplitude)

        # B. Pérdida Física
        t = self.t_values.to(pred_amplitude.device)
        physics_samples = pulse_shape_physics(t, pred_amplitude)

        # Comparamos la forma del pulso teórico con el input original
        physics_loss = self.mse(physics_samples, input_samples)

        return data_loss + self.lambd * physics_loss


# --- 3. Arquitectura del Modelo (LSTM) ---
class TileCalPINN(nn.Module):
    def __init__(self):
        super(TileCalPINN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=32, num_layers=1, batch_first=True
        )
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


# --- 4. Pipeline de Entrenamiento ---
def train_pinn():
    data_path = "../data/tilecal_hardened_data.csv"
    if not os.path.exists(data_path):
        print("Error: Ejecuta primero los scripts de generación de datos.")
        return

    df = pd.read_csv(data_path)
    X = df[[f"s{i}" for i in range(7)]].values
    y = df["true_amplitude"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Escalado de datos
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    # Preparar Tensores
    X_train_t = torch.FloatTensor(X_train_scaled).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(-1)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
    )

    model = TileCalPINN()
    criterion = PhysicsInformedLoss(lambd=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print("--- ENTRENANDO PINN (PHYSICS-INFORMED NEURAL NETWORK) ---")
    model.train()
    for epoch in range(51):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)

            # Re-escalamos batch_x para la comparación física
            original_samples = batch_x.squeeze() * X_std + X_mean
            loss = criterion(outputs, batch_y, original_samples)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)  # El scheduler ajusta el LR si la pérdida se estanca

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d} | Total Loss: {avg_loss:.4f} | LR: {current_lr:.6f}"
            )

    # Evaluación
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        mae = torch.mean(torch.abs(preds - y_test_t))
        print(f"\n PINN Finalizada.")
        print(f"MAE Final en Test: {mae.item():.4f} GeV")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tilecal_pinn_model.pth")
    joblib.dump({"mean": X_mean, "std": X_std}, "models/scaling_params.pkl")
    print("Modelo y parámetros de escalado guardados.")


if __name__ == "__main__":
    train_pinn()
