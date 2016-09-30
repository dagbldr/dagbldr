# Author: Kyle Kastner
# License: BSD 3-clause
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor
from ..utils import concatenate
from ..core import get_logger, get_type

logger = get_logger()
_type = get_type()


'''
def embedding(list_of_index_inputs, max_index, proj_dim, graph, name,
              random_state=None, strict=True, init_func=np_uniform):
    check_type = any([index_input.dtype != "int32"
                      for index_input in list_of_index_inputs])
    check_dim = any([index_input.ndim != 1
                     for index_input in list_of_index_inputs])
    if check_type or check_dim:
        raise ValueError("index_input must be an ivector!")
    embedding_W_name = name + "_embedding_W"
    list_of_names = [embedding_W_name]
    if not names_in_graph(list_of_names, graph):
        assert random_state is not None
        np_embedding_W = init_func((max_index, proj_dim), random_state)
        add_arrays_to_graph([np_embedding_W], list_of_names, graph,
                            strict=strict)
    else:
        if strict:
            raise AttributeError(
                "Name %s already found in graph with strict mode!" % name)
    embedding_W, = fetch_from_graph(list_of_names, graph)
    embeddings = [embedding_W[index_input]
                  for index_input in list_of_index_inputs]
    # could sum instead?
    output = concatenate(embeddings, graph, name, axis=embedding_W.ndim - 1)
    n_lists = len(list_of_index_inputs)
    return output.reshape((-1, n_lists, proj_dim))


def softmax_sample(list_of_multinomial_inputs, graph, name,
                         random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_multinomial = concatenate(list_of_multinomial_inputs, graph,
                                   name,
                                   axis=list_of_multinomial_inputs[0].ndim - 1)
    conc_multinomial /= len(list_of_multinomial_inputs)
    samp = theano_rng.multinomial(pvals=conc_multinomial,
                                  dtype="int32")
    # We know shape of conc_multinomial == shape of random sample
    shape = calc_expected_dims(graph, conc_multinomial)
    list_of_random = [samp, ]
    list_of_names = [name + "_random", ]
    list_of_shapes = [shape, ]
    add_random_to_graph(list_of_random, list_of_shapes, list_of_names, graph)
    return samp
'''


def gaussian_sample(list_of_mu_inputs, list_of_sigma_inputs,
                    name=None, random_state=None):
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        print("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs,
                          axis=list_of_mu_inputs[0].ndim - 1)
    conc_sigma = concatenate(list_of_sigma_inputs,
                             axis=list_of_sigma_inputs[0].ndim - 1)
    e = theano_rng.normal(size=(conc_sigma.shape[0],
                                conc_sigma.shape[1]),
                          dtype=conc_sigma.dtype)
    samp = conc_mu + conc_sigma * e
    return samp


def gaussian_log_sample(list_of_mu_inputs, list_of_log_sigma_inputs,
                        name, random_state=None):
    """ log_sigma_inputs should be from a linear """
    theano_seed = random_state.randint(-2147462579, 2147462579)
    # Super edge case...
    if theano_seed == 0:
        logger.info("WARNING: prior layer got 0 seed. Reseeding...")
        theano_seed = random_state.randint(-2**32, 2**32)
    theano_rng = MRG_RandomStreams(seed=theano_seed)
    conc_mu = concatenate(list_of_mu_inputs,
                          axis=list_of_mu_inputs[0].ndim - 1)
    conc_log_sigma = concatenate(list_of_log_sigma_inputs,
                                 axis=list_of_log_sigma_inputs[0].ndim - 1)
    e = theano_rng.normal(size=(conc_log_sigma.shape[0],
                                conc_log_sigma.shape[1]),
                          dtype=conc_log_sigma.dtype)
    samp = conc_mu + tensor.exp(conc_log_sigma) * e
    return samp
