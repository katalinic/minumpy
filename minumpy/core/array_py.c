#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include "array.h"
#include "array_dtypes.h"
#include "array_py.h"
#include "array_py_utils.h"
#include "array_utils.h"

static void
py_array_dealloc(pyArrayObject *pa)
{
    if (pa->arr) array_free(pa->arr);
    Py_TYPE(pa)->tp_free((PyObject *)pa);
}

static PyObject *
py_array_alloc(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pyArrayObject *pa = NULL;
    arrayObject *a = NULL;
    PyObject *b = NULL;
    static char *kwlist[] = {"initialiser", "shape", "dtype", NULL};
    arrayDims dims = {NULL, 0};
    ARRAY_DTYPE dtype = UNKNOWN;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&i", kwlist,
                                     &b,
                                     py_seq_to_intp,
                                     &dims,
                                     &dtype)) {
        goto fail;
    }

    if (check_array_object_py_initialiser(b, &dims, &dtype)) {
        goto fail;
    }

    a = array_alloc(dims.ptr, dims.len, dtype);
    if (a == NULL) {
        PyErr_SetString(PyExc_ValueError, "Array allocation failed");
        goto fail;
    }

    fill_array_object_with_py_init(a, b, &dims);
    free(dims.ptr);

    pa = (pyArrayObject *)type->tp_alloc(type, 0);
    pa->arr = a;
    return (PyObject *)pa;

fail:
    if (dims.ptr) free(dims.ptr);
    return NULL;
}

static PyObject *
py_array_str(pyArrayObject *pa)
{
    return PyUnicode_FromString(array_str(pa->arr));
}

static PyObject *
py_array_ravel(pyArrayObject *pa, PyObject *Py_UNUSED(ignored))
{
    arrayObject *a = pa->arr;
    int n = NUM_ARRAY_ELEMS(a);
    PyObject *ret = PyList_New(n);
    // TODO: avoid double allocation
    void *ra = array_ravel(a);
    fill_py_list_from_buf(ret, ra, n, a->dtype);
    free(ra);
    return ret;
}

static PyObject *
py_array_transpose(pyArrayObject *pa, PyObject *perm)
{
    arrayObject *a = pa->arr;
    arrayDims dims = {NULL, 0};

    if (!py_seq_to_intp(perm, &dims)) {
        return NULL;
    }

    if (a->nd != dims.len) {
        PyErr_SetString(PyExc_ValueError,
            "Length of permutations does not match dimensionality of array");
        goto fail;
    }

    for (int i = 0; i < dims.len; i++) {
        if (dims.ptr[i] < 0 || dims.ptr[i] >= a->nd) {
            PyErr_SetString(PyExc_ValueError,
                "Permutation values must be in [0, a->nd)");
            goto fail;
        }
    }

    array_transpose(a, dims.ptr);
    free(dims.ptr);
    Py_RETURN_NONE;

fail:
    if (dims.ptr) free(dims.ptr);
    return NULL;
}

static PyObject *
py_array_sum(pyArrayObject *pa, PyObject *pyAxis)
{

    arrayObject *a = pa->arr;
    arrayObject *ret_arr = NULL;
    pyArrayObject *ret = NULL;
    PyTypeObject *type = NULL;

    if (!PyLong_Check(pyAxis)) {
        PyErr_SetString(PyExc_TypeError,
            "Axis argument must be an integer");
        return NULL;
    }

    int axis = PyLong_AsLong(pyAxis);
    if (axis < 0 || axis >= a->nd) {
        PyErr_SetString(PyExc_ValueError,
            "Axis argument must be in [0, a->nd)");
        return NULL;
    }

    ret_arr = array_sum(a, axis);
    if (ret_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Sum failed");
        return NULL;
    }

    type = Py_TYPE(pa);
    ret = (pyArrayObject *)type->tp_alloc(type, 0);
    ret->arr = ret_arr;
    return (PyObject *)ret;
}

static PyObject *
py_array_dot(pyArrayObject *pa, PyObject *b)
{
    arrayObject *a = pa->arr;
    arrayObject *ret_arr = NULL;
    pyArrayObject *ret = NULL;
    PyTypeObject *type = NULL;

    if (Py_TYPE(b) != Py_TYPE(pa)) {
        PyErr_SetString(PyExc_TypeError, "Expected array argument");
        return NULL;
    }

    ret_arr = array_dot(a, ((pyArrayObject *)b)->arr);
    if (ret_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Dot product failed");
        return NULL;
    }

    type = Py_TYPE(pa);
    ret = (pyArrayObject *)type->tp_alloc(type, 0);
    ret->arr = ret_arr;
    return (PyObject *)ret;
}

static PyObject *
py_array_ones(pyArrayObject *pa, PyObject *Py_UNUSED(ignored))
{
    arrayObject *a = pa->arr;
    arrayObject *ret_arr = NULL;
    PyTypeObject *type = NULL;
    pyArrayObject *ret = NULL;

    ret_arr = array_copy(a);
    if (ret_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Array copy failed");
        return NULL;
    }

    double one = 1;
    buf_fill_val(ret_arr->data, one, NUM_ARRAY_ELEMS(a), a->dtype);

    type = Py_TYPE(pa);
    ret = (pyArrayObject *)type->tp_alloc(type, 0);
    ret->arr = ret_arr;
    return (PyObject *)ret;
}

static PyObject *
py_array_randint(pyArrayObject *pa, PyObject *args)
{
    arrayObject *a = pa->arr;
    arrayObject *ret_arr = NULL;
    PyTypeObject *type = NULL;
    pyArrayObject *ret = NULL;

    int low;
    int high;
    if (!PyArg_ParseTuple(args, "ii", &low, &high)) {
        return NULL;
    }

    if (high <= low) {
        PyErr_SetString(PyExc_ValueError, "High must be greater than low");
        return NULL;
    }

    ret_arr = array_copy(a);
    if (ret_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Array copy failed");
        return NULL;
    }

    buf_fill_uniform_int(ret_arr->data, low, high, NUM_ARRAY_ELEMS(a), a->dtype);

    type = Py_TYPE(pa);
    ret = (pyArrayObject *)type->tp_alloc(type, 0);
    ret->arr = ret_arr;
    return (PyObject *)ret;
}

static PyObject *
py_array_get_dtype(pyArrayObject *a)
{
    return PyLong_FromLong(a->arr->dtype);
}

static PyObject *
py_array_get_nd(pyArrayObject *a)
{
    return PyLong_FromLong(a->arr->nd);
}

static PyObject *
py_array_get_dims(pyArrayObject *a)
{
    return py_tup_from_intp(a->arr->dims, a->arr->nd);
}

static PyObject *
py_array_get_strides(pyArrayObject *a)
{
    return py_tup_from_intp(a->arr->strides, a->arr->nd);
}

static PyGetSetDef py_array_getsetters[] = {
    {"dtype", (getter)py_array_get_dtype, NULL, NULL, NULL},
    {"nd", (getter)py_array_get_nd, NULL, NULL, NULL},
    {"dims", (getter)py_array_get_dims, NULL, NULL, NULL},
    {"strides", (getter)py_array_get_strides, NULL, NULL, NULL},
    {NULL},
};

static PyMethodDef py_array_methods[] = {
    {"ravel", (PyCFunction)py_array_ravel, METH_NOARGS, NULL},
    {"transpose", (PyCFunction)py_array_transpose, METH_O, NULL},
    {"sum", (PyCFunction)py_array_sum, METH_O, NULL},
    {"dot", (PyCFunction)py_array_dot, METH_O, NULL},
    {"ones", (PyCFunction)py_array_ones, METH_NOARGS, NULL},
    {"randint", (PyCFunction)py_array_randint, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject ArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "minumpy.array",
    .tp_doc = "Minumpy ndarray",
    .tp_basicsize = sizeof(pyArrayObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = py_array_alloc,
    .tp_dealloc = (destructor) py_array_dealloc,
    .tp_getset = py_array_getsetters,
    .tp_methods = py_array_methods,
    .tp_str = (reprfunc) py_array_str
};

static struct PyModuleDef minarraydef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "minarray",
    .m_doc = "",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_minarray(void)
{
    PyObject *ret;
    if (PyType_Ready(&ArrayType) < 0) {
        return NULL;
    }

    ret = PyModule_Create(&minarraydef);
    if (ret == NULL) {
        return NULL;
    }

    Py_INCREF(&ArrayType);
    if (PyModule_AddObject(ret, "array", (PyObject *)&ArrayType) < 0) {
        Py_DECREF(&ArrayType);
        Py_DECREF(ret);
        return NULL;
    }

    return ret;
}
