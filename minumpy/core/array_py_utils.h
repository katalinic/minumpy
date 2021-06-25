#ifndef ARRAY_PY_UTILS_H
#define ARRAY_PY_UTILS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "array_dtypes.h"

typedef struct arrayDims {
    int *ptr;
    int len;
} arrayDims;

int py_seq_to_intp(PyObject *obj, arrayDims *out);
PyObject *py_tup_from_intp(int *vals, int n);

int check_py_list_uniform_type(PyObject *obj, PyTypeObject *type);
int check_array_object_py_initialiser(PyObject *obj, arrayDims *dims, ARRAY_DTYPE *dtype);

void fill_array_object_with_py_init(arrayObject *a, PyObject *obj, arrayDims *dims);
void fill_buf_from_py_list(char *buf, int offset, PyObject *list, int n, ARRAY_DTYPE dtype);
void fill_py_list_from_buf(PyObject *list, void *buf, int n, ARRAY_DTYPE dtype);


#endif
