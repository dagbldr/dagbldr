import tables
import numbers
import numpy as np


class minibatch_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 make_mask=False,
                 random_access=False,
                 list_of_formatters=None,
                 random_state=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.slice_start_ = start_index
        if list_of_formatters is None:
            self.list_of_formatters = [lambda x: x for x in list_of_containers]
        else:
            self.list_of_formatters = list_of_formatters
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % sample_axis)
        self.random_access = random_access
        if self.random_access and random_state is None:
            self.random_state = random_state
            raise ValueError("Must provide random seed for random_access!")

    def reset(self):
        self.slice_start_ = self.start_index

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
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
