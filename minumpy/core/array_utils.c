#include <stdlib.h>
#include <string.h>

#include "array_dtypes.h"
#include "array_utils.h"

int
prod(int *vals, int n)
{
    int num_elems = 1;
    for (int i = 0; i < n; i++) {
        num_elems *= vals[i];
    }
    return num_elems;
}

int *
cumprod_reverse(int *vals, int n)
{
    int *strides = malloc(n * sizeof(int));
    strides[n - 1] = 1;
    for (int i = n - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * vals[i + 1];
    }
    return strides;
}

int
validate_nd(int nd)
{
    return (nd <= 0 || nd > ARRAY_NUM_DIMS) ? 1 : 0;
}

int *
promote_dims(int *dims, int nd, int target_nd)
{
    int *ret = malloc(target_nd * sizeof(int));
    memcpy(ret, dims, nd * sizeof(int));
    for (int i = nd; i < target_nd; i++) {
        ret[i] = 1;
    }
    return ret;
}

void
swap_idx(int *vals, int i, int j)
{
    int tmp = vals[i];
    vals[i] = vals[j];
    vals[j] = tmp;
}

int *
range(int n)
{
    int *ret = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        ret[i] = i;
    }
    return ret;
}

int *
filter_idx(int *vals, int n, int idx)
{
    int ret_n = n > 1 ? n - 1 : 1;
    int *ret = malloc(ret_n * sizeof(int));
    ret[0] = 1;
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (i == idx) continue;
        ret[j++] = vals[i];
    }
    return ret;
}

typedef void (*buf_set_val_func)(char *, void *);
typedef void (*buf_add_val_func)(char *, void *);
typedef void (*buf_set_zero_func)(char *);
typedef void (*buf_fill_val_func)(char *, double, int);
typedef void (*buf_fill_vals_func)(char *, const void *, int);
typedef void (*buf_fill_uniform_int_func)(char *, int, int, int);
typedef void (*reduce_mul_add_func)(char *, const void *, const void *, int);
typedef void (*reduce_sum_func)(char *, const void *, int);
typedef int  (*print_val_func)(char *, size_t, char *);

void buf_set_val_func_int32(char *buf, void *val) {*(int32_t *)buf = *(int32_t *)val;}
void buf_set_val_func_int64(char *buf, void *val) {*(int64_t *)buf = *(int64_t *)val;}
void buf_set_val_func_float(char *buf, void *val) {*(float *)buf = *(float *)val;}
void buf_set_val_func_double(char *buf, void *val) {*(double *)buf = *(double *)val;}

void buf_add_val_func_int32(char *buf, void *val) {*(int32_t *)buf += *(int32_t *)val;}
void buf_add_val_func_int64(char *buf, void *val) {*(int64_t *)buf += *(int64_t *)val;}
void buf_add_val_func_float(char *buf, void *val) {*(float *)buf += *(float *)val;}
void buf_add_val_func_double(char *buf, void *val) {*(double *)buf += *(double *)val;}

void buf_set_zero_func_int32(char *buf) {*(int32_t *)buf = (int32_t)0;}
void buf_set_zero_func_int64(char *buf) {*(int64_t *)buf = (int64_t)0;}
void buf_set_zero_func_float(char *buf) {*(float *)buf = (float)0;}
void buf_set_zero_func_double(char *buf) {*(double *)buf = (double)0;}

void buf_fill_val_func_int32(char *buf, double val, int n) {
    for (int i = 0; i < n; i++) {
        ((int32_t *)buf)[i] = (int32_t)val;
    }
}
void buf_fill_val_func_int64(char *buf, double val, int n) {
    for (int i = 0; i < n; i++) {
        ((int64_t *)buf)[i] = (int64_t)val;
    }
}
void buf_fill_val_func_float(char *buf, double val, int n) {
    for (int i = 0; i < n; i++) {
        ((float *)buf)[i] = (float)val;
    }
}
void buf_fill_val_func_double(char *buf, double val, int n) {
    for (int i = 0; i < n; i++) {
        ((double *)buf)[i] = (double)val;
    }
}

void buf_fill_vals_func_int32(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        ((int32_t *)buf)[i] = ((int32_t *)vals)[i];
    }
}
void buf_fill_vals_func_int64(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        ((int64_t *)buf)[i] = ((int64_t *)vals)[i];
    }
}
void buf_fill_vals_func_float(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        ((float *)buf)[i] = ((float *)vals)[i];
    }
}
void buf_fill_vals_func_double(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        ((double *)buf)[i] = ((double *)vals)[i];
    }
}

void buf_fill_uniform_int_func_int32(char *buf, int low, int high, int n) {
    for (int i = 0; i < n; i++) {
        int v = low + rand() / (RAND_MAX / (high - low + 1) + 1);
        ((int32_t *)buf)[i] = (int32_t)v;
    }
}
void buf_fill_uniform_int_func_int64(char *buf, int low, int high, int n) {
    for (int i = 0; i < n; i++) {
        int v = low + rand() / (RAND_MAX / (high - low + 1) + 1);
        ((int64_t *)buf)[i] = (int64_t)v;
    }
}
void buf_fill_uniform_int_func_float(char *buf, int low, int high, int n) {
    for (int i = 0; i < n; i++) {
        int v = low + rand() / (RAND_MAX / (high - low + 1) + 1);
        ((float *)buf)[i] = (float)v;
    }
}
void buf_fill_uniform_int_func_double(char *buf, int low, int high, int n) {
    for (int i = 0; i < n; i++) {
        int v = low + rand() / (RAND_MAX / (high - low + 1) + 1);
        ((double *)buf)[i] = (double)v;
    }
}

void reduce_mul_add_func_int32(char *buf, const void *a, const void *b, int n) {
    for (int i = 0; i < n; i++) {
        *(int32_t *)buf += ((int32_t *)a)[i] *((int32_t *)b)[i];
    }
}
void reduce_mul_add_func_int64(char *buf, const void *a, const void *b, int n) {
    for (int i = 0; i < n; i++) {
        *(int64_t *)buf += ((int64_t *)a)[i] *((int64_t *)b)[i];
    }
}
void reduce_mul_add_func_float(char *buf, const void *a, const void *b, int n) {
    for (int i = 0; i < n; i++) {
        *(float *)buf += ((float *)a)[i] *((float *)b)[i];
    }
}
void reduce_mul_add_func_double(char *buf, const void *a, const void *b, int n) {
    for (int i = 0; i < n; i++) {
        *(double *)buf += ((double *)a)[i] *((double *)b)[i];
    }
}

void reduce_sum_func_int32(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        *(int32_t *)buf += ((int32_t *)vals)[i];
    }
}
void reduce_sum_func_int64(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        *(int64_t *)buf += ((int64_t *)vals)[i];
    }
}
void reduce_sum_func_float(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        *(float *)buf += ((float *)vals)[i];
    }
}
void reduce_sum_func_double(char *buf, const void *vals, int n) {
    for (int i = 0; i < n; i++) {
        *(double *)buf += ((double *)vals)[i];
    }
}

int print_val_func_int32(char *out, size_t n, char *buf) {
    return snprintf(out, n, "%1.2e", (double)*(int32_t *)buf);
}
int print_val_func_int64(char *out, size_t n, char *buf) {
    return snprintf(out, n, "%1.2e", (double)*(int64_t *)buf);
}
int print_val_func_float(char *out, size_t n, char *buf) {
    return snprintf(out, n, "%1.2e", (double)*(float *)buf);
}
int print_val_func_double(char *out, size_t n, char *buf) {
    return snprintf(out, n, "%1.2e", (double)*(double *)buf);
}

static buf_set_val_func buf_set_val_funcs[NUM_ARRAY_DTYPES] = {
    buf_set_val_func_int32,
    buf_set_val_func_int64,
    buf_set_val_func_float,
    buf_set_val_func_double,
};

static buf_add_val_func buf_add_val_funcs[NUM_ARRAY_DTYPES] = {
    buf_add_val_func_int32,
    buf_add_val_func_int64,
    buf_add_val_func_float,
    buf_add_val_func_double,
};

static buf_set_zero_func buf_set_zero_funcs[NUM_ARRAY_DTYPES] = {
    buf_set_zero_func_int32,
    buf_set_zero_func_int64,
    buf_set_zero_func_float,
    buf_set_zero_func_double,
};

static buf_fill_val_func buf_fill_val_funcs[NUM_ARRAY_DTYPES] = {
    buf_fill_val_func_int32,
    buf_fill_val_func_int64,
    buf_fill_val_func_float,
    buf_fill_val_func_double,
};

static buf_fill_vals_func buf_fill_vals_funcs[NUM_ARRAY_DTYPES] = {
    buf_fill_vals_func_int32,
    buf_fill_vals_func_int64,
    buf_fill_vals_func_float,
    buf_fill_vals_func_double,
};

static buf_fill_uniform_int_func buf_fill_uniform_int_funcs[NUM_ARRAY_DTYPES] = {
    buf_fill_uniform_int_func_int32,
    buf_fill_uniform_int_func_int64,
    buf_fill_uniform_int_func_float,
    buf_fill_uniform_int_func_double,
};

static reduce_mul_add_func reduce_mul_add_funcs[NUM_ARRAY_DTYPES] = {
    reduce_mul_add_func_int32,
    reduce_mul_add_func_int64,
    reduce_mul_add_func_float,
    reduce_mul_add_func_double,
};

static reduce_sum_func reduce_sum_funcs[NUM_ARRAY_DTYPES] = {
    reduce_sum_func_int32,
    reduce_sum_func_int64,
    reduce_sum_func_float,
    reduce_sum_func_double,
};

static print_val_func print_val_funcs[NUM_ARRAY_DTYPES] = {
    print_val_func_int32,
    print_val_func_int64,
    print_val_func_float,
    print_val_func_double,
};

void
buf_set_val(char *buf, void *val, ARRAY_DTYPE dtype)
{
    buf_set_val_funcs[dtype](buf, val);
}

void
buf_add_val(char *buf, void *val, ARRAY_DTYPE dtype)
{
    buf_add_val_funcs[dtype](buf, val);
}

void
buf_set_zero(char *buf, ARRAY_DTYPE dtype)
{
    buf_set_zero_funcs[dtype](buf);
}

void
buf_fill_val(char *buf, double val, int n, ARRAY_DTYPE dtype)
{
    buf_fill_val_funcs[dtype](buf, val, n);
}

void
buf_fill_vals(char *buf, const void *vals, int n, ARRAY_DTYPE dtype)
{
    buf_fill_vals_funcs[dtype](buf, vals, n);
}

void
buf_fill_uniform_int(char *buf, int low, int high, int n, ARRAY_DTYPE dtype)
{
    buf_fill_uniform_int_funcs[dtype](buf, low, high, n);
}

void
reduce_mul_add(char *buf, const void *a, const void *b, int n, ARRAY_DTYPE dtype)
{
    reduce_mul_add_funcs[dtype](buf, a, b, n);
}

void
reduce_sum(char *buf, const void *vals, int n, ARRAY_DTYPE dtype)
{
    reduce_sum_funcs[dtype](buf, vals, n);
}

int
print_val(char *out, size_t n, char *buf, ARRAY_DTYPE dtype)
{
    return print_val_funcs[dtype](out, n, buf);
}
