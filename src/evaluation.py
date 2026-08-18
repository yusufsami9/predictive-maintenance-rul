import numpy as np


def nasa_score(y_true, y_pred):
    """Asymmetric scoring from the PHM08 challenge (Saxena et al., 2008, eq. 11).

    Late predictions (d >= 0, model overestimates remaining life) use the smaller
    denominator (10), producing steeper penalty growth -- since flying an engine
    longer than it can safely operate is more dangerous than retiring it early.
    Early predictions (d < 0) use the larger denominator (13), a gentler penalty.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)
