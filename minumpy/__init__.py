from minarray import array as _array

_dtypes = {"int32": 0, "int64": 1, "float": 2, "double": 3}

int32 = _dtypes["int32"]
int64 = _dtypes["int64"]
float = _dtypes["float"]
double = _dtypes["double"]

__all__ = ["array", "ones", "randint", "ravel", "transpose", "sum", "dot"]
__all__.extend(_dtypes.keys())


def _check_dtype(dtype):
    if dtype is not None and dtype not in _dtypes.values():
        raise ValueError("Unsupported dtype {}, must be one of {}".format(
            dtype, _dtypes.keys()
        ))


def array(vals=None, shape=None, dtype=None):
    kwargs = {}
    if shape is not None:
        kwargs["shape"] = shape
    if dtype is not None:
        _check_dtype(dtype)
        kwargs["dtype"] = dtype
    return _array(vals, **kwargs)


def ones(shape=None, dtype=None):
    _check_dtype(dtype)
    return array(shape=shape, dtype=dtype).ones()


def randint(low=0, high=1, shape=None, dtype=None):
    _check_dtype(dtype)
    return array(shape=shape, dtype=dtype).randint(low, high)


def ravel(a):
    return a.ravel()


def transpose(a, perm=None):
    return a.transpose(perm)


def sum(a, axis=0):
    return a.sum(axis)


def dot(a, b):
    return a.dot(b)
