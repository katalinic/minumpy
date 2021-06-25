#ifndef ARRAY_PY_H
#define ARRAY_PY_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdio.h>

#include "array.h"

typedef struct pyArrayObject {
    PyObject_HEAD
    arrayObject *arr;
} pyArrayObject;

#endif
