from dagbldr.datasets.dataset_utils import add_memory_swapper
from numpy.testing import assert_raises, assert_array_equal
import time
import tables
import numpy as np


def test_add_memory_swapper():
    n_samples = 1000
    n_features = 2500
    # 10 MB = 10E6 Bytes
    # 4 Bytes per float32, 2.5E6 items
    # This does an in-memory hdf5 so no saving to disk
    # copy time vs read time vs alloc time ratio should still apply
    # though times will be faster
    hdf5_file = tables.open_file("fake.hdf5", "w", driver="H5FD_CORE",
                                 driver_core_backing_store=0)
    data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                  tables.Float32Atom(),
                                  shape=(0, n_features),
                                  expectedrows=n_samples)
    random_state = np.random.RandomState(1999)
    r = random_state.rand(n_samples, n_features).astype("float32")
    for n in range(len(r)):
        data.append(r[n][None])

    # 9 MB storage
    data = add_memory_swapper(data, mem_size=9E6)

    old_getter = data.__getitem__

    # First access allocates memory and stripes in
    t1 = time.time()
    data[:10]
    t2 = time.time()
    # Second access should be in the already stored data
    data[10:20]
    t3 = time.time()
    # This should force a copy into memory swapper
    data[-20:-10]
    t4 = time.time()
    # This should access existing data
    data[-10:]
    t5 = time.time()

    # Make sure results from beginning and end are matched
    assert np.all(data[0] == old_getter(0))
    assert np.all(data[0:10] == old_getter(slice(0, 10, 1)))
    assert np.all(data[len(data) - 10:len(data)] == old_getter(
        slice(-10, None, 1)))
    assert np.all(data[-1] == old_getter(-1))
    assert np.all(data[-10:None] == old_getter(slice(-10, None, 1)))
    assert np.all(data[-20:-10] == old_getter(slice(-20, -10, 1)))
    assert_raises(ValueError, lambda: data[:])
    hdf5_file.close()

    # Should be fast to read things already in memory
    assert (t3 - t2) < (t2 - t1)
    assert (t5 - t4) < (t4 - t3)

    n_samples = 1000
    n_features = 2500
    # 10 MB = 10E6 Bytes
    # 4 Bytes per float32, 2.5E6 items
    # This does an in-memory hdf5 so no saving to disk
    # copy time vs read time vs alloc time ratio should still apply
    # though times will be faster
    hdf5_file = tables.open_file("fake.hdf5", "w", driver="H5FD_CORE",
                                 driver_core_backing_store=0)
    data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                  tables.Float32Atom(),
                                  shape=(0, n_features),
                                  expectedrows=n_samples)
    random_state = np.random.RandomState(1999)
    r = random_state.rand(n_samples, n_features).astype("float32")
    for n in range(len(r)):
        data.append(r[n][None])
    old_getter = data.__getitem__

    # 11 MB storage
    data = add_memory_swapper(data, mem_size=11E6)
    t1 = time.time()
    assert np.all(data[:] == old_getter(slice(0, None, 1)))
    t2 = time.time()
    assert np.all(data[:-1] == old_getter(slice(0, -1, 1)))
    t3 = time.time()
    assert (t3 - t2) < (t2 - t1)
    hdf5_file.close()

if __name__ == "__main__":
    test_add_memory_swapper()
