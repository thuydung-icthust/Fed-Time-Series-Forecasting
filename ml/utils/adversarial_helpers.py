
import numpy as np


def poison_batch(batch, ratio=1.0):
    x, exogenous, y_hist, y = batch
    size = x.shape[0]
    poisct = int(ratio * size)
    samplepois = np.random.choice(size, poisct)
    for idx in samplepois:
        # Flip the target values at the selected time steps
        y[idx, 0] = 1.0 - y[idx, 0]
    return x, exogenous, y_hist, y
    