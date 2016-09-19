from dagbldr.utils import load_checkpoint, make_masked_minibatch
from dagbldr.datasets import fetch_iamondb, list_iterator
import numpy as np
import theano
import sys
import time


def plot_scatter_iamondb_example(X, title=None, index=None, ax=None, equal=True,
                                 save=True):
    if save:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if ax is None:
        f, ax = plt.subplots()
    down = np.where(X[:, 0] == 0)[0]
    up = np.where(X[:, 0] == 1)[0]
    ax.scatter(X[down, 1], X[down, 2], color="steelblue")
    ax.scatter(X[up, 1], X[up, 2], color="darkred")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if equal:
        ax.set_aspect('equal')
    if title is not None:
        plt.title(title)
    if ax is None:
        if not save:
            plt.show()
        else:
            if index is None:
                t = time.time()
            else:
                t = index
            plt.savefig("scatter_%i.png" % t)


def plot_lines_iamondb_example(X, title=None, index=None, ax=None, equal=True,
                               save=True):
    if save:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if ax is None:
        f, ax = plt.subplots()
    points = list(np.where(X[:, 0] > 0)[0])
    start_points = [0] + points
    stop_points = points + [len(X)]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for start, stop in zip(start_points, stop_points):
        # Hack to actually separate lines...
        ax.plot(X[start + 2:stop, 1], X[start + 2:stop, 2], color="black")
    if equal:
        ax.set_aspect('equal')
    if title is not None:
        plt.title(title)
    if ax is None:
        if not save:
            plt.show()
        else:
            if index is None:
                t = time.time()
            else:
                t = index
            plt.savefig("lines_%i.png" % t)


def delta(x):
    return np.hstack((x[1:, 0][:, None], x[1:, 1:] - x[:-1, 1:]))


def undelta(x):
    agg = np.cumsum(x[:, 1:], axis=0)
    return np.hstack((x[:, 0][:, None], agg))

model_path = sys.argv[1]
checkpoint = load_checkpoint(model_path)
predict_function = checkpoint.checkpoint_dict["predict_function"]
cost_function = checkpoint.checkpoint_dict["cost_function"]

iamondb = fetch_iamondb()
X = iamondb["data"]
X_offset = [delta(x) for x in X]
X = X_offset
Xt = [x[:, 1:] for x in X]
X_len = np.array([len(x) for x in Xt]).sum()
X_mean = np.array([x.sum() for x in Xt]).sum() / X_len
X_sqr = np.array([(x**2).sum() for x in Xt]).sum() / X_len
X_std = np.sqrt(X_sqr - X_mean ** 2)


def normalize(x):
    return np.hstack((x[:, 0][:, None], (x[:, 1:] - X_mean) / (X_std)))


def unnormalize(x):
    return np.hstack((x[:, 0][:, None], (x[:, 1:] * X_std) + X_mean))

X = np.array([normalize(x).astype(theano.config.floatX) for x in X])
y = np.array([x[1:] for x in X])
X = np.array([x[:-1] for x in X])

minibatch_size = 20  # Size must match size in training, same for above preproc
train_itr = list_iterator([X, y], minibatch_size, axis=1, make_mask=True,
                          stop_index=1000)
X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)

running_mb = X_mb[:1] * 0
running_mask = X_mb_mask[:1] * 0


def gen_sample(rval, random_state, idx=-1):
    # binary
    # coeffs
    # mus
    # log_sigmas
    # corr
    binary, coeffs, mus, sigmas, corr = rval
    binary = binary[idx, :, 0]
    coeffs = coeffs[idx, :, :]
    mu_x = mus[idx, :, 0, :]
    mu_y = mus[idx, :, 1, :]
    sigma_x = np.exp(sigmas[idx, :, 0, :])
    sigma_y = np.exp(sigmas[idx, :, 1, :])
    corr = corr[idx, :, 0, :]
    z_x = random_state.randn(*mu_x.shape)
    z_y = random_state.randn(*mu_y.shape)
    chosen = np.array([np.argmax(random_state.multinomial(1, coeffs[i]))
                       for i in range(len(coeffs))])
    s_x = mu_x + sigma_x * z_x
    s_y = mu_y + sigma_y * ((z_x * corr) + z_y * np.sqrt(1. - corr ** 2))
    binarized = random_state.binomial(1, binary).ravel()[:, None]
    s_x = s_x[np.arange(len(s_x)), chosen.ravel()][:, None]
    s_y = s_y[np.arange(len(s_x)), chosen.ravel()][:, None]
    return binarized, s_x, s_y, chosen


random_state = np.random.RandomState(1999)
n_samples = 700
for i in range(n_samples):
    print("Generating sample %i" % i)
    rval = predict_function(running_mb, running_mask)
    b, s_x, s_y, c = gen_sample(rval, random_state)
    mb_stack = np.hstack((b, s_x, s_y))[None]
    running_mb = np.concatenate((running_mb, mb_stack), axis=0).astype(
        theano.config.floatX)
    running_mask = np.ones_like(running_mb[:, :, 0])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
n_plot_samples = 3
assert n_plot_samples <= minibatch_size
fline, axline = plt.subplots(n_plot_samples, 1)
for i in range(n_plot_samples):
    r = unnormalize(running_mb[:, i])
    r = undelta(r)
    plot_lines_iamondb_example(r, ax=axline[i])
plt.savefig('line.png')
plt.close()

fscatter, axscatter = plt.subplots(n_plot_samples, 1)
for i in range(n_plot_samples):
    r = unnormalize(running_mb[:, i])
    r = undelta(r)
    plot_scatter_iamondb_example(r, ax=axscatter[i])
plt.savefig('scatter.png')
plt.close()
