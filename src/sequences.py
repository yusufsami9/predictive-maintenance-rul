import numpy as np


def create_sequences(df, feature_cols, window=30, target_col="RUL", unit_col="unit"):
    """Turn per-cycle rows into fixed-length windows for an LSTM.

    Builds windows per engine so a window never spans two engines. Returns X of
    shape (n_windows, window, n_features) and y of shape (n_windows,), where each
    label is the target column's value at the window's final cycle.
    """
    X_list, y_list = [], []

    for unit in df[unit_col].unique():
        engine = df[df[unit_col] == unit]
        features = engine[feature_cols].values
        targets = engine[target_col].values
        n_windows = len(features) - window + 1

        for i in range(n_windows):
            X_list.append(features[i : i + window])
            y_list.append(targets[i + window - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
