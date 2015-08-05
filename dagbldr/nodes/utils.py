import numpy as np


def generate_autoregressive_mask(input_size, output_size, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    input_reordered = np.arange(1, input_size + 1)
    random_state.shuffle(input_reordered)
    output_sampled = random_state.randint(1, input_size, output_size)
    connections_mask = np.ones((input_size, output_size))
    for i in range(input_size):
        connections_mask[i, :] *= (
            output_sampled > input_reordered[i]).astype("int32")
    return connections_mask, input_reordered
