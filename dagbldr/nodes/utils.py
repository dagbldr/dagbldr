import numpy as np


def _generate_autoregressive_mask(input_ordering, output_ordering):
    input_size = len(input_ordering)
    output_size = len(output_ordering)
    connections_mask = np.ones((input_size, output_size))
    for i in range(input_size):
        connections_mask[i, :] *= (
            input_ordering[i] <= output_ordering).astype("int32")
    return connections_mask


def generate_autoregressive_masks(sizes, forced_input_ordering=None,
                                  forced_samplings=None,
                                  random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    masks = []
    orderings_and_samplings = []
    if forced_input_ordering is not None:
        assert len(forced_input_ordering) == sizes[0]
    if forced_samplings is not None:
        # -2 to discount input, output
        assert len(forced_samplings) == len(sizes) - 2
    for n, (i, j) in enumerate(list(zip(sizes[:-1], sizes[1:]))):
        if n == 0:
            if forced_input_ordering is not None:
                input_ordering = forced_input_ordering
            else:
                input_ordering = np.arange(1, sizes[0] + 1)
                random_state.shuffle(input_ordering)
            if forced_samplings is not None:
                output_ordering = forced_samplings[0]
            else:
                output_ordering = random_state.randint(1, sizes[0], j)
            assert min(input_ordering) == 1
            assert max(input_ordering) == sizes[0]
            assert len(np.unique(input_ordering)) == sizes[0]
            assert min(output_ordering) > 0
            assert max(output_ordering) < sizes[0]
            l_mask = _generate_autoregressive_mask(
                input_ordering, output_ordering)
            orderings_and_samplings.extend([input_ordering, output_ordering])
            masks.append(l_mask)
        elif j == sizes[-1]:
            input_ordering = orderings_and_samplings[-1]
            output_ordering = orderings_and_samplings[0]
            # invert mask generation function for last layer!
            # in order to get the correct output will need to do some work
            l_mask = _generate_autoregressive_mask(
                output_ordering, input_ordering)
            # Turn 0 to 1 and 1 to 0
            l_mask = l_mask.T
            l_mask[l_mask < 0.5] = 2.
            l_mask[l_mask < 1.5] = 1.
            l_mask -= 1.
            masks.append(l_mask)
            orderings_and_samplings.append(output_ordering)
        else:
            if forced_samplings is not None:
                output_ordering = forced_samplings[n]
            else:
                output_ordering = random_state.randint(1, sizes[0], j)
            input_ordering = orderings_and_samplings[-1]
            assert min(input_ordering) > 0
            assert max(input_ordering) < sizes[0]
            assert min(output_ordering) > 0
            assert min(output_ordering) < sizes[0]
            l_mask = _generate_autoregressive_mask(
                input_ordering, output_ordering)
            masks.append(l_mask)
            orderings_and_samplings.append(output_ordering)
    return masks, orderings_and_samplings
