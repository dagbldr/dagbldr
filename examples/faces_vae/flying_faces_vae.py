import argparse
import numpy as np
import os

from dagbldr.datasets import fetch_fer
from dagbldr.utils import load_checkpoint, interpolate_between_points, make_gif

parser = argparse.ArgumentParser()
parser.add_argument("saved_functions_file",
                    help="Saved pickle file from vae training")
parser.add_argument("--seed", "-s",
                    help="random seed for path calculation",
                    action="store", default=1979, type=int)

args = parser.parse_args()
if not os.path.exists(args.saved_functions_file):
    raise ValueError("Please provide a valid path for saved pickle file!")

checkpoint_dict = load_checkpoint(args.saved_functions_file)
encode_function = checkpoint_dict["encode_function"]
decode_function = checkpoint_dict["decode_function"]

fer = fetch_fer()
data = fer["data"]
valid_indices = fer["valid_indices"]
valid_data = data[valid_indices]
mean_norm = fer["mean0"]
pca_tf = fer["pca_matrix"]
X = valid_data - mean_norm
X = np.dot(X, pca_tf.T)
random_state = np.random.RandomState(args.seed)

# number of samples
n_plot_samples = 5
# tfd dimensions
width = 48
height = 48
# Get random data samples
ind = np.arange(len(X))
random_state.shuffle(ind)
sample_X = X[ind[:n_plot_samples]]


def gen_samples(arr):
    mu, log_sig = encode_function(arr)
    # No noise at test time
    out, = decode_function(mu + np.exp(log_sig))
    return out

# VAE specific plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
samples = gen_samples(sample_X)
samples = np.dot(samples, pca_tf) + mean_norm
f, axarr = plt.subplots(n_plot_samples, 2)
for n, (X_i, s_i) in enumerate(zip(np.dot(sample_X, pca_tf) + mean_norm,
                                   samples)):
    axarr[n, 0].matshow(X_i.reshape(width, height), cmap="gray")
    axarr[n, 1].matshow(s_i.reshape(width, height), cmap="gray")
    axarr[n, 0].axis('off')
    axarr[n, 1].axis('off')
plt.savefig('vae_reconstruction.png')
plt.close()

# Calculate linear path between points in space
mus, log_sigmas = encode_function(sample_X)
mu_path = interpolate_between_points(mus)
log_sigma_path = interpolate_between_points(log_sigmas)

# Path across space from one point to another
path = mu_path + np.exp(log_sigma_path)
out, = decode_function(path)
out = np.dot(out, pca_tf) + mean_norm
make_gif(out, "vae_code.gif", width, height, delay=1, grayscale=True)
