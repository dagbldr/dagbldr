import tables
import numbers
import numpy as np
import itertools

from ..core import get_logger

logger = get_logger()


class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 make_mask=False,
                 one_hot_class_size=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.slice_start_ = start_index
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)

    def reset(self):
        self.slice_start_ = self.start_index

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = np.arange(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            return self._slice_without_masks(ind)
        else:
            return self._slice_with_masks(ind)

    def _slice_without_masks(self, ind):
        try:
            if self.axis == 0:
                return [c[ind] for c in self.list_of_containers]
            elif self.axis == 1:
                return [c[:, ind] for c in self.list_of_containers]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")

    def _slice_with_masks(self, ind):
        try:
            cs = self._slice_without_masks(ind)
            if self.axis == 0:
                ms = [np.ones_like(c[:, 0]) for c in cs]
            elif self.axis == 1:
                ms = [np.ones_like(c[:, :, 0]) for c in cs]
            assert len(cs) == len(ms)
            return [i for sublist in list(zip(cs, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


class minibatch_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        try:
            if self.axis == 0:
                if len(self.list_of_containers) > 1:
                    return [c[ind] for c in self.list_of_containers]
                else:
                    return self.list_of_containers[0][ind]

            elif self.axis == 1:
                if len(self.list_of_containers) > 1:
                    return [c[:, ind] for c in self.list_of_containers]
                else:
                    return self.list_of_containers[0][:, ind]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")

    def _slice_with_masks(self, ind):
        try:
            raise ValueError("Not yet implemented")
            cs = self._slice_without_masks(ind)
            if self.axis == 0:
                ms = [np.ones_like(c[:, 0]) for c in cs]
            elif self.axis == 1:
                ms = [np.ones_like(c[:, :, 0]) for c in cs]
            assert len(cs) == len(ms)
            return [i for sublist in list(zip(cs, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind):
        try:
            sliced_c = [np.asarray(c[ind]) for c in self.list_of_containers]
            for n in range(len(sliced_c)):
                sc = sliced_c[n]
                if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                    maxlen = max([len(i) for i in sc])
                    # Assume they at least have the same internal dtype
                    if len(sc[0].shape) > 1:
                        total_shape = (maxlen, sc[0].shape[1])
                    elif len(sc[0].shape) == 1:
                        total_shape = (maxlen, 1)
                    else:
                        raise ValueError("Unhandled array size in list")
                    if self.axis == 0:
                        raise ValueError("Unsupported axis of iteration")
                        new_sc = np.zeros((len(sc), total_shape[0],
                                           total_shape[1]))
                        new_sc = new_sc.squeeze().astype(sc[0].dtype)
                    else:
                        new_sc = np.zeros((total_shape[0], len(sc),
                                           total_shape[1]))
                        new_sc = new_sc.astype(sc[0].dtype)
                        for m, sc_i in enumerate(sc):
                            new_sc[:len(sc_i), m, :] = sc_i
                    sliced_c[n] = new_sc
            return sliced_c
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")

    def _slice_with_masks(self, ind):
        try:
            cs = self._slice_without_masks(ind)
            if self.axis == 0:
                ms = [np.ones_like(c[:, 0]) for c in cs]
            elif self.axis == 1:
                ms = [np.ones_like(c[:, :, 0]) for c in cs]
            assert len(cs) == len(ms)
            return [i for sublist in list(zip(cs, ms)) for i in sublist]
        except IndexError:
            self.reset()
            raise StopIteration("End of iteration")


def add_memory_swapper(earray, mem_size):
    class _cEArray(tables.EArray):
        pass

    # Filthy hack to override getter which is a cextension...
    earray.__class__ = _cEArray

    earray._in_mem_size = int(float(mem_size))
    assert earray._in_mem_size >= 1E6  # don't use for smaller than 1MB
    earray._in_mem_slice = np.empty([1] * len(earray.shape)).astype("float32")
    earray._in_mem_limits = [np.inf, -np.inf]

    old_getter = earray.__getitem__

    def _check_in_mem(earray, start, stop):
        lower = earray._in_mem_limits[0]
        upper = earray._in_mem_limits[1]
        if start < lower or stop > upper:
            return False
        else:
            return True

    def _load_in_mem(earray, start, stop):
        # start and stop are slice indices desired - we calculate different
        # sizes to put in memory
        n_bytes_per_entry = earray._in_mem_slice.dtype.itemsize
        n_entries = earray._in_mem_size / float(n_bytes_per_entry)
        n_samples = earray.shape[0]
        n_other = earray.shape[1:]
        n_samples_that_fit = int(n_entries / np.prod(n_other))
        assert n_samples_that_fit > 0
        # handle - index case later
        assert start >= 0
        assert stop >= 0
        assert stop >= start
        slice_size = stop - start
        if slice_size > n_samples_that_fit:
            err_str = "Slice from [%i:%i] (size %i) too large! " % (
                start, stop, slice_size)
            err_str += "Max slice size %i" % n_samples_that_fit
            raise ValueError(err_str)
        slice_limit = [start, stop]
        earray._in_mem_limits = slice_limit
        if earray._in_mem_slice.shape[0] == 1:
            # allocate memory
            print("Allocating %f gigabytes of memory for EArray swap buffer" %
                  (earray._in_mem_size / float(1E9)))
            earray._in_mem_slice = np.empty((n_samples_that_fit,) + n_other,
                                            dtype=earray.dtype)
        # handle edge case when last chunk is smaller than what slice will
        # return
        limit = min([slice_limit[1] - slice_limit[0],
                     n_samples - slice_limit[0]])
        earray._in_mem_slice[:limit] = old_getter(
            slice(slice_limit[0], slice_limit[1], 1))

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            start, stop, step = self._processRange(key, key, 1)
            if key < 0:
                key = start
            if _check_in_mem(self, key, key):
                lower = self._in_mem_limits[0]
            else:
                # slice into memory...
                _load_in_mem(self, key, key)
                lower = self._in_mem_limits[0]
            return self._in_mem_slice[key - lower]
        elif isinstance(key, slice):
            start, stop, step = self._processRange(
                key.start, key.stop, key.step)
            if _check_in_mem(self, start, stop):
                lower = self._in_mem_limits[0]
            else:
                # slice into memory...
                _load_in_mem(self, start, stop)
                lower = self._in_mem_limits[0]
            return self._in_mem_slice[start - lower:stop - lower:step]
        elif len(key) == 2:
            # Slice with extra axes like [1:100, :]
            key = key[0]
            start, stop, step = self._processRange(
                key.start, key.stop, key.step)
            if _check_in_mem(self, start, stop):
                lower = self._in_mem_limits[0]
            else:
                # slice into memory...
                _load_in_mem(self, start, stop)
                lower = self._in_mem_limits[0]
            return self._in_mem_slice[start - lower:stop - lower:step]
        else:
            raise ValueError("Must index with a slice object or single value"
                             "along 0 axis!")
    # This line is critical...
    _cEArray.__getitem__ = getter
    return earray


cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lower = "abcdefghijklmnopqrstuvwxyz"
alpha = "0123456789"
punc = " \\`,.?!=-+!@#$%^&*()[]{}:;'|/"
punc += '"'
all_chars = cap + lower + alpha + punc

class character_sequence_iterator(object):
    def __init__(self, sentence_iterator, minibatch_size,
                 truncation_length,
                 iterator_length=None,
                 start_index=0,
                 stop_index=np.inf,
                 valid_items=None,
                 stop_items=None,
                 extra_preproc_options=None):
        self.sentence_iterator = sentence_iterator
        self.minibatch_size = minibatch_size
        self.truncation_length = truncation_length
        self.start_index = start_index
        self.stop_index = stop_index
        if start_index != 0 or stop_index != np.inf:
            raise AttributeError("start_index and stop_index not yet supported")
        self.slice_start_ = start_index
        self.extra_preproc_options = extra_preproc_options
        if self.extra_preproc_options is not None:
            raise AttributeError("Extra preproc options not yet supported")
        self.n_classes = len(all_chars)
        rlu = {k: v for k, v in enumerate(all_chars)}
        lu = {v: k for k, v in rlu.items()}
        self.char_to_class = lu
        self.class_to_char = rlu
        if iterator_length is None:
            logger.info("No iterator_length provided for truncation mode.")
            logger.info("Calculating...")
            l = 0
            for s in sentence_iterator:
                l += len(s)
            self.iterator_length = l

        self.iterator_length -= self.iterator_length % (minibatch_size * truncation_length)
        self.sequence_length = self.iterator_length // minibatch_size

        def rg():
            minibatch_gens = []
            sl = self.sequence_length
            for i in range(minibatch_size):
                self.flat_gen = (i for s in sentence_iterator for i in s)
                minibatch_gens.append(itertools.islice(self.flat_gen, i * sl, (i + 1) * sl))
            self.flat_gen = (i for s in sentence_iterator for i in s)
            self.minibatch_gens = minibatch_gens
        rg()
        self.reset_gens = rg

    def reset(self):
        #self.slice_start_ = self.start_index
        self.reset_gens()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def _t(self, el):
        return self.char_to_class[el]

    def __next__(self):
        try:
            arr  = [[self._t(self.minibatch_gens[i].next())
                    for n in range(self.truncation_length)]
                    for i in range(self.minibatch_size)]
            return np.asarray(arr).T.astype("float32")
        except StopIteration:
            self.reset()
            raise StopIteration("Stop index reached")
