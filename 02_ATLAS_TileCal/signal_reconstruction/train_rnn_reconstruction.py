import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class TileCalRNN(nn.Module):
    def __init__(self):
        super(TileCalRNN, self).__init__()
        # Capa LSTM que analiza la secuencia de 7 muestras
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=32, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


def train_rnn():
    df = pd.read_csv("../data/tilecal_hardened_data.csv")
    X = df[[f"s{i}" for i in range(7)]].values
    y = df["true_amplitude"].values

    X = X.reshape(-1, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Tensores de PyTorch
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )

    model = TileCalRNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("--- ENTRENANDO RNN (LSTM) PARA ATLAS TILECAL ---")
    for epoch in range(20):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        preds = model(X_test_t).numpy()
        mae = np.mean(np.abs(preds.flatten() - y_test))
        print(f"\n RNN Finalizada.")
        print(f"MAE en test: {mae:.4f} GeV")

    torch.save(model.state_dict(), "models/tilecal_rnn_model.pth")


if __name__ == "__main__":
    train_rnn()
