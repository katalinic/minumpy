import pytest
from pytest import raises as assert_raises

import minumpy as np

EPSILON = 1e-8


def assert_sequences_equal(a, b):
    def _vals_equal(x, y): return abs(x - y) < EPSILON
    assert all(_vals_equal(va, vb) for va, vb in zip(a, b))


def assert_array_metadata(a, dtype, nd, dims, strides):
    assert (a.dtype == dtype and a.nd == nd and a.dims == dims and
            a.strides == strides)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float, np.double])
def test_initialisation(dtype):
    a = np.array([4], dtype=dtype)
    assert_array_metadata(a, dtype, 2, (1, 1), (1, 1))

    b = np.array([398, 41], dtype=dtype)
    assert_array_metadata(b, dtype, 2, (2, 1), (1, 1))

    c = np.array([[2, -81, 26],
                  [17, 102, -3]], dtype=dtype)
    assert_array_metadata(c, dtype, 2, (2, 3), (3, 1))

    d = np.array(shape=(1), dtype=dtype)
    assert_array_metadata(d, dtype, 2, (1, 1), (1, 1))

    e = np.array(shape=(3, 7), dtype=dtype)
    assert_array_metadata(e, dtype, 2, (3, 7), (7, 1))

    assert_raises(ValueError, np.array, [])
    assert_raises(ValueError, np.array, [[[]]])
    assert_raises(ValueError, np.array, ())

    assert_raises(ValueError, np.array, [1, 1.])
    assert_raises(ValueError, np.array, [[1, 1], [1, 1, 1]])

    assert_raises(ValueError, np.array, shape=(0))
    assert_raises(ValueError, np.array, shape=(1, 1, 1))


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float, np.double])
def test_ravel(dtype):
    assert_sequences_equal(
        np.array([4], dtype=dtype).ravel(), [4])
    assert_sequences_equal(
        np.array([398, 41], dtype=dtype).ravel(), [398, 41])
    assert_sequences_equal(
        np.array([[2, -81, 26],
                  [17, 102, -3]], dtype=dtype).ravel(),
        [2, -81, 26, 17, 102, -3])


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float, np.double])
def test_transpose(dtype):
    a = np.array([4], dtype=dtype)
    b = np.array([398, 41], dtype=dtype)
    c = np.array([[2, -81, 26],
                  [17, 102, -3]], dtype=dtype)
    np.transpose(a, (1, 0))
    np.transpose(b, (1, 0))
    np.transpose(c, (1, 0))

    assert_array_metadata(a, dtype, 2, (1, 1), (1, 1))
    assert_array_metadata(b, dtype, 2, (1, 2), (1, 1))
    assert_array_metadata(c, dtype, 2, (3, 2), (1, 3))

    assert_sequences_equal(a.ravel(), [4])
    assert_sequences_equal(b.ravel(), [398, 41])
    assert_sequences_equal(c.ravel(), [2, 17, -81, 102, 26, -3])

    assert_raises(ValueError, np.array([1, 1], dtype=dtype).transpose, (1, 2))
    assert_raises(ValueError, np.array([1, 1], dtype=dtype).transpose, (1))


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float, np.double])
def test_sum(dtype):
    a = np.array([4], dtype=dtype)
    b = np.array([398, 41], dtype=dtype)
    c = np.array([[2, -81, 26],
                  [17, 102, -3]], dtype=dtype)

    as0 = np.sum(a, 0)
    bs0 = np.sum(b, 0)
    bs1 = np.sum(b, 1)
    cs0 = np.sum(c, 0)
    cs1 = np.sum(c, 1)
    assert_sequences_equal(
        as0.ravel(),
        [4])
    assert_sequences_equal(
        bs0.ravel(),
        [439])
    assert_sequences_equal(
        bs1.ravel(),
        [398, 41])
    assert_sequences_equal(
        cs0.ravel(),
        [19, 21, 23])
    assert_sequences_equal(
        cs1.ravel(),
        [-53, 116])

    assert_raises(ValueError, np.array([1, 1]).sum, 2)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float, np.double])
def test_dot(dtype):
    assert_sequences_equal(
        np.dot(
            np.array([[3, 1, 5, 2, 0],
                      [4, 4, 1, 0, 3],
                      [0, 2, 7, 1, 9]], dtype=dtype),
            np.array([1, 1, 0, 3, 2], dtype=dtype)).ravel(),
        [10, 14, 23]
    )


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float, np.double])
def test_ones(dtype):
    a = np.ones(shape=(3,), dtype=dtype)
    assert_array_metadata
    assert_array_metadata(a, dtype, 2, (3, 1), (1, 1))

    b = np.ones(shape=(7, 4), dtype=dtype)
    assert_array_metadata(b, dtype, 2, (7, 4), (4, 1))

    assert np.sum(a, 0).ravel() == [3]
    assert np.sum(np.sum(b, 1), 0).ravel() == [7 * 4]


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_randint(dtype):
    a = np.randint(shape=(6, 3), dtype=dtype)

    s = np.sum(np.sum(a, 1), 0).ravel()
    assert s[0] >= 0 and s[0] <= 18

    assert_raises(ValueError, np.randint, low=2, high=1)
