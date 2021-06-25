#ifndef ARRAY_H
#define ARRAY_H

#include <stdint.h>
#include <stdio.h>

#include "array_dtypes.h"
#include "array_utils.h"

typedef struct arrayObject {
    char *data;
    ARRAY_DTYPE dtype;
    int nd;
    int *dims;
    int *strides;
} arrayObject;

#define NUM_ARRAY_ELEMS(a) prod(a->dims, a->nd)

arrayObject *array_alloc(int *dims, int n, ARRAY_DTYPE dtype);
void array_free(arrayObject *a);
arrayObject *array_copy(const arrayObject *a);

void array_fill_val(arrayObject *a, double val, ARRAY_DTYPE dtype);
void array_fill_vals(arrayObject *a, const void *vals, ARRAY_DTYPE dtype);
void array_fill_uniform_int(arrayObject *a, int low, int high, ARRAY_DTYPE dtype);

void *array_ravel(const arrayObject *a);
void array_transpose(arrayObject *a, int *perm);

arrayObject *array_sum(arrayObject *a, int axis);
arrayObject *array_dot(arrayObject *a, arrayObject *b);

char *array_str(arrayObject *);

#endif
