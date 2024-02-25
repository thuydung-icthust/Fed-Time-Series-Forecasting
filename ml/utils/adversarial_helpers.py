
import numpy as np
import torch


def poison_batch(batch, ratio=1.0):
    x, exogenous, y_hist, y = batch
    size = x.shape[0]
    poisct = int(ratio * size)
    samplepois = np.random.choice(size, poisct)
    for idx in samplepois:
        # Flip the target values at the selected time steps
        y[idx, 0] = 1.0 - y[idx, 0]
    return x, exogenous, y_hist, y

def poison_flip(batch, ratio=1.0, regressor=None, best_relax=1, poison_dim=0):
    # regressor is just here to have unified api with statP
    x, exogenous, y_hist, y = batch
    # print(f"y: {y}")
    size = x.shape[0]
    n_poisoned = int(ratio * size)
    # samplepois = np.random.choice(size, n_poisoned)
    del regressor
    if n_poisoned < 1:
        return np.zeros(shape=(0, x.shape[1])), np.zeros(shape=(0,))
    y = np.asarray(y)
    tmax = np.max(y, axis=0)
    tmin = np.min(y, axis=0)

    # find out if there is more potential shifting the decision surface uniformly towards the max or min
    upper_max_abs_error = np.abs(y - tmax)
    lower_max_abs_error = np.abs(y - tmin)

    for j in range(y.shape[-1]):
        direction = ['up' if upper_max_abs_error[i][j] > lower_max_abs_error[i][j] else 'down' for i in range(y.shape[0])]

        # get x_p (subset of true x)
        delta = [upper_max_abs_error[i][j] if direction[i] == 'up' else lower_max_abs_error[i][j] for i in range(y.shape[0])]

        # select those who are 1) farthest
        assert best_relax >= 1
        best = np.argsort(delta)[-n_poisoned * best_relax:]
        best = np.random.choice(best, size=n_poisoned, replace=False) # => idxes
        for idx in best:
            y[idx, j] = tmin[j] if direction[idx] == 'down' else tmax[j]
            
    # x_p = x[best]
    # get y_p
    # target = [tmin if (numpy.max(y) - numpy.min(y)) / 2 < yyy else tmax for yyy in y[best]]
    # target = [tmin if direction[i] == 'down' else tmax for i in best]
    # y_p = np.asarray(target)
    return x, exogenous, y_hist, torch.tensor(y)
    