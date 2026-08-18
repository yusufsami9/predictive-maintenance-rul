import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    """Reads a fixed-length window and predicts a single RUL value."""

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # sequence-to-one: keep only the final time step
        return self.fc(last_hidden).squeeze(-1)


def train_lstm(
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq,
    epochs=40,
    seed=42,
    batch_size=128,
    lr=0.001,
    device_name="cpu",
):
    """Train an LSTMRegressor with best-epoch checkpointing on validation RMSE.

    Validation error typically starts climbing once the network begins
    overfitting the training set, so the weights from the best epoch are
    restored before returning rather than keeping whatever epoch ran last.
    """
    torch.manual_seed(seed)
    device = torch.device(device_name)

    X_train_t = torch.tensor(X_train_seq)
    y_train_t = torch.tensor(y_train_seq)
    X_val_t = torch.tensor(X_val_seq)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = LSTMRegressor(n_features=X_train_seq.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse, best_state, best_epoch = float("inf"), None, 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)
        train_mse = epoch_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device)).cpu().numpy()
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val_seq) ** 2)))
        history.append((train_mse, val_rmse))

        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t.to(device)).cpu().numpy()

    info = {"history": history, "best_epoch": best_epoch, "best_rmse": best_rmse}
    return model, val_pred, info


def fit_lstm_fixed_epochs(
    X_train_seq,
    y_train_seq,
    epochs,
    seed=42,
    batch_size=128,
    lr=0.001,
    device_name="cpu",
):
    """Fit an LSTM for a preselected epoch count without inspecting holdout labels.

    This is used after model selection: for cross-fitted calibration models and
    for the final refit on every available training engine.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    device = torch.device(device_name)

    X_train_t = torch.tensor(X_train_seq)
    y_train_t = torch.tensor(y_train_seq)
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model = LSTMRegressor(n_features=X_train_seq.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)
        history.append(epoch_loss / len(X_train_t))

    model.eval()
    return model, history
