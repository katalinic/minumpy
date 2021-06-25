# Minumpy

Playing around with Python C extensions with a very bare-bones implementation of the [NumPy](https://github.com/numpy/numpy) `ndarray`. Purely for educational purposes.

Current API is
* `np.array(initialiser=None, shape=None, dtype=None)`
* `np.ones(shape=None, dtype=None)`
* `np.randint(low=0, high=1, shape=None, dtype=None)`
* `np.ravel(arr)`
* `np.transpose(arr, permutation=None)`
* `np.sum(arr, axis=0)`
* `np.dot(arr, other)`
