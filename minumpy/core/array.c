#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array.h"
#include "array_dtypes.h"
#include "array_utils.h"

arrayObject*
array_alloc(int *dims, int nd, ARRAY_DTYPE dtype)
{
    if (validate_nd(nd)) return NULL;

    arrayObject *a = malloc(sizeof(arrayObject));

    int ret_nd = ARRAY_NUM_DIMS;
    int *ret_dims = promote_dims(dims, nd, ARRAY_NUM_DIMS);
    int *ret_strides = cumprod_reverse(ret_dims, ret_nd);

    a->data = calloc(prod(ret_dims, ret_nd), array_dtype_size(dtype));
    a->dtype = dtype;
    a->nd = ret_nd;
    a->dims = ret_dims;
    a->strides = ret_strides;

    return a;
}

void
array_free(arrayObject *a)
{
    if (!a) return;
    free(a->data);
    free(a->dims);
    free(a->strides);
    free(a);
}

arrayObject*
array_copy(const arrayObject *a)
{
    arrayObject *ret = array_alloc(a->dims, a->nd, a->dtype);
    memcpy(ret->data, a->data, NUM_ARRAY_ELEMS(a) * array_dtype_size(a->dtype));
    return ret;
}

void
array_fill_val(arrayObject *a, double val, ARRAY_DTYPE dtype) {
    buf_fill_val(a->data, val, NUM_ARRAY_ELEMS(a), dtype);
}

void
array_fill_vals(arrayObject *a, const void *vals, ARRAY_DTYPE dtype) {
    buf_fill_vals(a->data, vals, NUM_ARRAY_ELEMS(a), dtype);
}

void
array_fill_uniform_int(arrayObject *a, int low, int high, ARRAY_DTYPE dtype) {
    buf_fill_uniform_int(a->data, low, high, NUM_ARRAY_ELEMS(a), dtype);
}

void
*array_ravel(const arrayObject *a)
{
    size_t dtype_size = array_dtype_size(a->dtype);
    void *ret = malloc(NUM_ARRAY_ELEMS(a) * dtype_size);
    for (int i = 0; i < a->dims[0]; i++) {
        int row_offset = i * a->strides[0];
        for (int j = 0; j < a->dims[1]; j++) {
            buf_set_val(
                ret + (i * a->dims[1] + j) * dtype_size,
                a->data + (row_offset + j * a->strides[1]) * dtype_size,
                a->dtype
            );
        }
    }
    return ret;
}

void
array_transpose(arrayObject *a, int *perm)
{
    int *permuted_dims = malloc(a->nd * sizeof(int));
    int *permuted_strides = malloc(a->nd * sizeof(int));
    for (int i = 0; i < a->nd; i++) {
        permuted_dims[i] = a->dims[perm[i]];
        permuted_strides[i] = a->strides[perm[i]];
    }
    free(a->dims);
    a->dims = permuted_dims;
    free(a->strides);
    a->strides = permuted_strides;
}

arrayObject*
array_sum(arrayObject *a, int axis)
{
    int ret_nd = a->nd > 1 ? a->nd - 1 : 1;
    int *ret_dims = filter_idx(a->dims, a->nd, axis);
    arrayObject *ret = array_alloc(ret_dims, ret_nd, a->dtype);

    int *perm = range(a->nd);
    swap_idx(perm, axis, a->nd - 1);
    array_transpose(a, perm);

    size_t dtype_size = array_dtype_size(a->dtype);
    void *a_ravel = array_ravel(a);
    for (int i = 0; i < a->dims[0]; i++) {
        reduce_sum(
            ret->data + i * dtype_size,
            a_ravel + i * a->dims[1] * dtype_size,
            a->dims[1],
            a->dtype
        );
    }
    free(a_ravel);

    array_transpose(a, perm);

    return ret;
}

arrayObject
*array_dot(arrayObject *a, arrayObject *b)
{
    if (a->dims[a->nd - 1] != b->dims[0]) {
        printf("Dims mismatch (%d %d)\n", a->dims[a->nd - 1], b->dims[0]);
        return NULL;
    }
    if (a->dtype != b->dtype) {
        printf("dtype mismatch (%d %d)\n", a->dtype, b->dtype);
        return NULL;
    }

    int ret_nd = ARRAY_NUM_DIMS;
    int ret_dims[] = {a->dims[0], b->dims[1]};
    arrayObject *ret = array_alloc(ret_dims, ret_nd, a->dtype);

    int *perm = range(b->nd);
    swap_idx(perm, 0, b->nd - 1);
    array_transpose(b, perm);

    // TODO: avoid this memory overhead
    void *a_ravel = array_ravel(a);
    void *b_ravel = array_ravel(b);

    size_t dtype_size = array_dtype_size(a->dtype);
    char buf[sizeof(double)];
    for (int i = 0; i < a->dims[0]; i++) {
        int a_offset = i * a->dims[1];
        int ret_offset = i * b->dims[0];
        for (int j = 0; j < b->dims[0]; j++) {
            int b_offset = j * b->dims[1];
            buf_set_zero(buf, DOUBLE);
            reduce_mul_add(
                buf,
                a_ravel + a_offset * dtype_size,
                b_ravel + b_offset * dtype_size,
                a->dims[1],
                a->dtype
            );
            buf_add_val(
                ret->data + (ret_offset + j) * dtype_size,
                buf,
                a->dtype
            );
        }
    }
    free(a_ravel);
    free(b_ravel);

    array_transpose(b, perm);

    return ret;
}

char*
array_str(arrayObject *a)
{
    int perm[2] = {1, 0};
    int permuted = a->dims[1] == 1 ? 1 : 0;
    if (permuted) {
        array_transpose(a, perm);
    }

    size_t dtype_size = array_dtype_size(a->dtype);
    int entry_size = print_val(NULL, 0, a->data, a->dtype);
    int row_size = (entry_size + 1)  *a->dims[1] - 1;
    int buf_size = 0;
    buf_size += row_size * a->dims[0];
    buf_size += 2 * a->dims[0];  // brackets
    buf_size += a->dims[0];  // newlines
    buf_size += 1;  // termination
    char *buf = malloc(buf_size);

    int offset = 0;
    for (int i = 0; i < a->dims[0]; i++) {
        buf[offset++] = '[';
        for (int j = 0; j < a->dims[1]; j++) {
            offset += print_val(
                buf + offset,
                buf_size - offset,
                a->data + (i * a->strides[0] + j * a->strides[1]) * dtype_size,
                a->dtype
            );
            if (j < a->dims[1] - 1) {
                buf[offset++] = ' ';
            }
        }
        buf[offset++] = ']';
        buf[offset++] = '\n';
    }
    buf[offset] = '\0';

    if (permuted) {
        array_transpose(a, perm);
    }

    return buf;
}
