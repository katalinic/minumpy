#include "array.h"
#include "array_dtypes.h"
#include "array_py_utils.h"

int
check_py_list_uniform_type(PyObject *obj, PyTypeObject *type)
{
    PyObject *v = NULL;
    for (int i = 0; i < PyList_Size(obj); i++) {
        v = PySequence_GetItem(obj, i);
        if (Py_TYPE(v) != type) {
            Py_DECREF(v);
            return 1;
        }
        Py_DECREF(v);
    }
    return 0;
}


int
check_array_object_py_initialiser(PyObject *obj, arrayDims *dims, ARRAY_DTYPE *dtype)
{
    Py_ssize_t len = 0;
    Py_ssize_t subLen = 0;
    PyObject *subObj = NULL;
    PyTypeObject *subType = NULL;

    if (PyList_Check(obj)) {
        len = PyList_Size(obj);
    }
    if (len <= 0) {
        if (dims->len == 0 || dims->ptr == NULL) {
            PyErr_SetString(PyExc_ValueError,
                "Array initialisation expects a base type of list");
            goto fail;
        }
        if (*dtype == UNKNOWN) {
            *dtype = DOUBLE;
        }
        return 0;
    }

    subObj = PySequence_GetItem(obj, 0);
    if (PyList_Check(subObj)) {
        subLen = PyList_Size(subObj);
        if (subLen <= 0) {
            PyErr_SetString(PyExc_ValueError,
                "Sub-lists cannot be empty.");
            goto fail;
        }

        PyObject *firstVal = PySequence_GetItem(subObj, 0);
        if (PyLong_Check(firstVal)) {
            subType = Py_TYPE(firstVal);
        }
        if (PyFloat_Check(firstVal)) {
            subType = Py_TYPE(firstVal);
        }
        Py_DECREF(firstVal);

        if (subType == NULL) {
            PyErr_SetString(PyExc_ValueError,
                "Sub-lists can only consist of ints and floats.");
            goto fail;
        }
        Py_DECREF(subObj);

        for (int i = 1; i < len; i++) {
            subObj = PySequence_GetItem(obj, i);
            if (!PyList_Check(subObj)) {
                PyErr_SetString(PyExc_ValueError,
                    "Non-uniform type provided in base list");
                goto fail;
            }
            if (PyList_Size(subObj) != subLen) {
                PyErr_SetString(PyExc_ValueError,
                    "Sub-lists must be of equal length.");
                goto fail;
            }
            if (check_py_list_uniform_type(subObj, subType)) {
                PyErr_SetString(PyExc_ValueError,
                    "Sub-lists must be of equal type.");
                goto fail;
            }
            Py_DECREF(subObj);
        }
    } else if (PyLong_Check(subObj)) {
        subType = Py_TYPE(subObj);
        Py_DECREF(subObj);
        if (check_py_list_uniform_type(obj, subType)) {
            PyErr_SetString(PyExc_ValueError,
                "List elements must be of same type.");
            goto fail;
        }
    } else if (PyFloat_Check(subObj)) {
        subType = Py_TYPE(subObj);
        Py_DECREF(subObj);
        if (check_py_list_uniform_type(obj, subType)) {
            PyErr_SetString(PyExc_ValueError,
                "List elements must be of same type.");
            goto fail;
        }
    } else {
        PyErr_SetString(PyExc_ValueError,
            "List elements can only be ints or floats.");
        goto fail;
    }

    if (subLen) {
        dims->len = ARRAY_NUM_DIMS;
        dims->ptr = calloc(dims->len, sizeof(int));
        dims->ptr[0] = len; dims->ptr[1] = subLen;
    } else {
        dims->len = 1;
        dims->ptr = calloc(dims->len, sizeof(int));
        dims->ptr[0] = len;
    }
    if (*dtype == UNKNOWN) {
        *dtype = subType == &PyLong_Type ? INT64 : DOUBLE;
    }
    return 0;

fail:
    if (subObj) Py_DECREF(subObj);
    if (subType) Py_DECREF(subType);
    return 1;
}

void
fill_buf_from_py_list(char *buf, int offset, PyObject *list, int n, ARRAY_DTYPE dtype) {
    PyObject *pyVal = NULL;
    for (int i = 0; i < n; i++) {
        pyVal = PySequence_GetItem(list, i);
        if (PyLong_Check(pyVal)) {
            switch (dtype) {
            case INT32: ((int32_t *)buf)[offset + i] = (int32_t)PyLong_AsLong(pyVal); break;
            case INT64: ((int64_t *)buf)[offset + i] = (int64_t)PyLong_AsLong(pyVal); break;
            case FLOAT: ((float *)buf)[offset + i] = (float)PyLong_AsLong(pyVal); break;
            case DOUBLE: ((double *)buf)[offset + i] = (double)PyLong_AsLong(pyVal); break;
            case UNKNOWN: break;
            }
        } else if (PyFloat_Check(pyVal)) {
            switch (dtype) {
            case INT32: ((int32_t *)buf)[i] = (int32_t)PyFloat_AsDouble(pyVal); break;
            case INT64: ((int64_t *)buf)[i] = (int64_t)PyFloat_AsDouble(pyVal); break;
            case FLOAT: ((float *)buf)[i] = (float)PyFloat_AsDouble(pyVal); break;
            case DOUBLE: ((double *)buf)[i] = (double)PyFloat_AsDouble(pyVal); break;
            case UNKNOWN: break;
            }
        }
        Py_DECREF(pyVal);
    }
}

void
fill_py_list_from_buf(PyObject *list, void *buf, int n, ARRAY_DTYPE dtype) {
    for (int i = 0; i < n; i++) {
        switch (dtype) {
            case INT32: PyList_SetItem(list, i, PyLong_FromLong(((int32_t *)buf)[i])); break;
            case INT64: PyList_SetItem(list, i, PyLong_FromLong(((int64_t *)buf)[i])); break;
            case FLOAT: PyList_SetItem(list, i, PyFloat_FromDouble(((float *)buf)[i])); break;
            case DOUBLE: PyList_SetItem(list, i, PyLong_FromDouble(((double *)buf)[i])); break;
            case UNKNOWN: break;
        }
    }
}

void
fill_array_object_with_py_init(arrayObject *a, PyObject *obj, arrayDims *dims)
{
    if (obj == Py_None) {
        return;
    }
    if (dims->len == 1) {
        fill_buf_from_py_list(a->data, 0, obj, a->dims[0], a->dtype);
    } else {
        PyObject *v = NULL;
        for (int i = 0; i < a->dims[0]; i++) {
            v = PySequence_GetItem(obj, i);
            fill_buf_from_py_list(a->data, i * a->dims[1], v, a->dims[1], a->dtype);
            Py_DECREF(v);
        }
    }
}

int
py_seq_to_intp(PyObject *obj, arrayDims *out)
{
    PyObject *valObj = NULL;
    out->ptr = NULL;
    out->len = 0;

    if (obj == Py_None) {
        return 1;
    }

    int *dims = calloc(ARRAY_NUM_DIMS, sizeof(int));
    if (PyNumber_Check(obj)) {
        int v = PyLong_AsLong(obj);
        if (v <= 0) {
            PyErr_SetString(PyExc_ValueError,
                "Expected positive value");
            goto fail;
        }
        dims[0] = v;
        out->len = 1;
        out->ptr = dims;
        return 1;
    }

    Py_ssize_t len = PySequence_Size(obj);
    if (validate_nd(len)) {
        PyErr_Format(PyExc_ValueError,
            "Expected sequence of length [1, 2], got %d", len);
        goto fail;
    }

    for (int i = 0; i < len; i++) {
        valObj = PySequence_GetItem(obj, i);
        if (valObj == Py_None) {
            PyErr_SetString(PyExc_ValueError,
                "Expected sequence of ints, saw None");
            goto fail;
        }
        if (!PyLong_Check(valObj)) {
            PyErr_SetString(PyExc_ValueError,
                "Expected sequence of ints");
            goto fail;
        }
        int v = PyLong_AsLong(valObj);
        Py_DECREF(valObj);
        if (v < 0) {
            PyErr_SetString(PyExc_ValueError,
                "Expected sequence of positive ints");
            goto fail;
        }
        dims[i] = v;
    }

    out->len = len;
    out->ptr = dims;
    return 1;

fail:
    if(valObj) Py_DECREF(valObj);
    if(dims) free(dims);
    return 0;
}

PyObject *
py_tup_from_intp(int *vals, int n)
{
    PyObject *tup = PyTuple_New(n);
    if (tup == NULL) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *v = PyLong_FromLong((long)vals[i]);
        if (v == NULL) {
            Py_DECREF(tup);
            return NULL;
        }
        PyTuple_SET_ITEM(tup, i, v);
    }

    return tup;
}
