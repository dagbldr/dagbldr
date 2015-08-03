import tables
import numbers
import numpy as np

def add_memory_swapper(earray, mem_size):
    class _cEArray(tables.EArray):
        pass

    # Filthy hack to override getter which is a cextension...
    earray.__class__ = _cEArray

    earray._in_mem_size = int(float(mem_size))
    assert earray._in_mem_size >= 1E6 # anything smaller than 1MB is worthless
    earray._in_mem_slice = np.empty((1, 1)).astype("float32")
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
        n_features = earray.shape[1]
        n_samples_that_fit = int(n_entries / n_features)
        assert n_samples_that_fit > 0
        # handle - index case later
        assert start >= 0
        assert stop >= 0
        assert stop >= start
        slice_size = stop - start
        if slice_size > n_samples_that_fit:
            err_str = "Slice from [%i:%i] (size %i) too large! " % (start, stop, slice_size)
            err_str += "Max slice size %i" % n_samples_that_fit
            raise ValueError(err_str)
        slice_limit = [start, stop]
        earray._in_mem_limits = slice_limit
        if earray._in_mem_slice.shape[0] == 1:
            # allocate memory
            print("Allocating %i bytes of memory for EArray swap buffer" % earray._in_mem_size)
            earray._in_mem_slice = np.empty((n_samples_that_fit, n_features), dtype="float32")
        # handle edge case when last chunk is smaller than what slice will
        # return
        limit = min([slice_limit[1] - slice_limit[0], n_samples - slice_limit[0]])
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
            start, stop, step = self._processRange(key.start, key.stop, key.step)
            if _check_in_mem(self, start, stop):
                lower = self._in_mem_limits[0]
            else:
                # slice into memory...
                _load_in_mem(self, start, stop)
                lower = self._in_mem_limits[0]
            return self._in_mem_slice[start - lower:stop - lower:step]
    # This line is critical...
    _cEArray.__getitem__ = getter
    return earray
