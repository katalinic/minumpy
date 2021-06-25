#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include "array_dtypes.h"

#define ARRAY_NUM_DIMS 2

int prod(int *vals, int n);
int *cumprod_reverse(int *vals, int n);
void swap_idx(int *vals, int i, int j);
int *range(int n);
int *filter_idx(int *vals, int n, int idx);

int validate_nd(int nd);
int *promote_dims(int *dims, int nd, int target_nd);

void buf_set_val(char *buf, void *val, ARRAY_DTYPE dtype);
void buf_add_val(char *buf, void *val, ARRAY_DTYPE dtype);
void buf_set_zero(char *buf, ARRAY_DTYPE dtype);
void buf_fill_val(char *buf, double val, int n, ARRAY_DTYPE dtype);
void buf_fill_vals(char *buf, const void *vals, int n, ARRAY_DTYPE dtype);
void buf_fill_uniform_int(char *buf, int low, int high, int n, ARRAY_DTYPE dtype);

void reduce_mul_add(char *buf, const void *a, const void *b, int n, ARRAY_DTYPE dtype);
void reduce_sum(char *buf, const void *vals, int n, ARRAY_DTYPE dtype);

int print_val(char *out, size_t n, char *buf, ARRAY_DTYPE dtype);

#endif
