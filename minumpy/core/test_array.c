#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array.h"
#include "array_dtypes.h"
#include "array_utils.h"

#define EPSILON 1e-8

static int
float_equal(double a, double b) {
    return fabs(a - b) > EPSILON;
}

static int
arrays_equal(void *a, void *b, int n, ARRAY_DTYPE dtype)
{
    double v_a = 0;
    double v_b = 0;
    for (int i = 0; i < n; i++) {
        switch (dtype) {
            case INT32:
                v_a = (double)*((int32_t *)a + i);
                v_b = (double)*((int32_t *)b + i);
                break;
            case INT64:
                v_a = (double)*((int64_t *)a + i);
                v_b = (double)*((int64_t *)b + i);
                break;
            case FLOAT:
                v_a = (double)*((float *)a + i);
                v_b = (double)*((float *)b + i);
                break;
            case DOUBLE:
                v_a = (double)*((double *)a + i);
                v_b = (double)*((double *)b + i);
                break;
            case UNKNOWN: return 1;
        }
        if (float_equal(v_a, v_b)) {
            printf("Unequal: %d - %f %f\n", i, v_a, v_b);
            return 1;
        }
    }
    return 0;
}

static int
char_arrays_equal(char *a, char *b, int n)
{
    return strncmp(a, b, n);
}

static int
check_array_metadata(arrayObject *a, ARRAY_DTYPE et, int *ed, int *es)
{
    if (a->dtype != et) return 1;
    for (int i = 0; i < a->nd; i++) {
        if (a->dims[i] != ed[i]) return 1;
        if (a->strides[i] != es[i]) return 1;
    }
    return 0;
}

static void *
cast_test_values(int *vals, int n, ARRAY_DTYPE dtype) {
    switch (dtype) {
        case INT32: {
            int32_t *ret = malloc(n * sizeof(int32_t));
            for (int i = 0; i < n; i++) {
                ret[i] = (int32_t)(vals[i]);
            }
            return ret;
        }
        case INT64: {
            int64_t *ret = malloc(n * sizeof(int64_t));
            for (int i = 0; i < n; i++) {
                ret[i] = (int64_t)(vals[i]);
            }
            return ret;
        }
        case FLOAT: {
            float *ret = malloc(n * sizeof(float));
            for (int i = 0; i < n; i++) {
                ret[i] = (float)(vals[i]);
            }
            return ret;
        }
        case DOUBLE: {
            double *ret = malloc(n * sizeof(double));
            for (int i = 0; i < n; i++) {
                ret[i] = (double)(vals[i]);
            }
            return ret;
        }
        case UNKNOWN: return NULL;
    }
}

int
test_instantiation(ARRAY_DTYPE dtype)
{
    arrayObject *a1 = NULL;
    arrayObject *a2 = NULL;
    int ds1[] = {3};
    int ed1[] = {3, 1};
    int es1[] = {1, 1};
    int ds2[] = {4, 2};
    int ed2[] = {4, 2};
    int es2[] = {2, 1};

    a1 = array_alloc(ds1, 1, dtype);
    if (!a1) goto fail;
    if (check_array_metadata(a1, dtype, ed1, es1)) goto fail;

    a2 = array_alloc(ds2, 2, dtype);
    if (!a2) goto fail;
    if (check_array_metadata(a2, dtype, ed2, es2)) goto fail;

    if (array_alloc(NULL, 0, dtype) != NULL) goto fail;
    if (array_alloc(NULL, 3, dtype) != NULL) goto fail;

    return 0;

fail:
    array_free(a1);
    array_free(a2);
    return 1;
}

int
test_fill(ARRAY_DTYPE dtype)
{
    arrayObject *a1 = NULL;
    arrayObject *a2 = NULL;
    void *r1 = NULL;
    void *r2 = NULL;
    void *cvs1 = NULL;
    void *cvs2 = NULL;
    int ds1[] = {5};
    int vs1[] = {4, 11, 3, 8, 0};
    int ds2[] = {4, 2};
    int vs2[] = {8, 3, 9, 1, 4, 2, 0, 6};

    a1 = array_alloc(ds1, 1, dtype);
    if (!a1) goto fail;
    cvs1 = cast_test_values(vs1, 5, dtype);
    array_fill_vals(a1, cvs1, dtype);
    r1 = array_ravel(a1);
    if (arrays_equal(r1, cvs1, 5, dtype)) goto fail;

    a2 = array_alloc(ds2, 2, dtype);
    if (!a2) goto fail;
    cvs2 = cast_test_values(vs2, prod(ds2, 2), dtype);
    array_fill_vals(a2, cvs2, dtype);
    r2 = array_ravel(a2);
    if (arrays_equal(r2, cvs2, 8, dtype)) goto fail;

    return 0;

fail:
    array_free(a1);
    array_free(a2);
    free(r1);
    free(r2);
    free(cvs1);
    free(cvs2);
    return 1;
}

int
test_copy(ARRAY_DTYPE dtype)
{
    arrayObject *a = NULL;
    arrayObject *b = NULL;
    void *cvs = NULL;
    int ds[] = {3, 5};
    int vs[] = {3, 1, 5, 2, 0,
                4, 4, 1, 0, 3,
                0, 2, 7, 1, 9};

    a = array_alloc(ds, 2, dtype);
    if (!a) goto fail;
    cvs = cast_test_values(vs, prod(ds, 2), dtype);
    array_fill_vals(a, cvs, dtype);

    b = array_copy(a);
    if (!b) goto fail;
    if (a->nd != b->nd) goto fail;
    if (char_arrays_equal(a->data, b->data, prod(ds, 2) * array_dtype_size(dtype))) goto fail;
    if (arrays_equal(a->dims, b->dims, 2, INT32)) goto fail;
    if (arrays_equal(a->strides, b->strides, 2, INT32)) goto fail;

    return 0;

fail:
    array_free(a);
    array_free(b);
    free(cvs);
    return 1;
}

int
test_transpose(ARRAY_DTYPE dtype)
{
    arrayObject *a = NULL;
    void *cvs = NULL;
    void *r = NULL;
    void *cts = NULL;
    int ds[] = {4, 2};
    int ed[] = {2, 4};
    int es[] = {1, 2};
    int vs[] = {8, 3, 9, 1, 4, 2, 0, 6};
    int ts[] = {8, 9, 4, 0, 3, 1, 2, 6};
    int perm[] = {1, 0};
    int nelems = prod(ds, 2);

    a = array_alloc(ds, 2, dtype);
    cvs = cast_test_values(vs, nelems, dtype);
    array_fill_vals(a, cvs, dtype);

    array_transpose(a, perm);
    if (check_array_metadata(a, dtype, ed, es)) goto fail;

    r = array_ravel(a);
    cts = cast_test_values(ts, nelems, dtype);
    if (arrays_equal(r, cts, nelems, dtype)) goto fail;

    return 0;

fail:
    array_free(a);
    free(cvs);
    free(r);
    free(cts);
    return 1;
}

int test_sum(ARRAY_DTYPE dtype)
{
    arrayObject *a1 = NULL;
    arrayObject *a2 = NULL;
    arrayObject *a3 = NULL;
    arrayObject *s1 = NULL;
    arrayObject *s20 = NULL;
    arrayObject *s21 = NULL;
    arrayObject *s30 = NULL;
    arrayObject *s31 = NULL;
    void *r1 = NULL;
    void *r20 = NULL;
    void *r21 = NULL;
    void *r30 = NULL;
    void *r31 = NULL;
    void *cv1 = NULL;
    void *cv2 = NULL;
    void *cv3 = NULL;
    void *ce1 = NULL;
    void *ce20 = NULL;
    void *ce21 = NULL;
    void *ce30 = NULL;
    void *ce31 = NULL;
    int ds1[] = {4};
    int ds2[] = {3, 5};
    int ds3[] = {3, 5};
    int v1[] = {3, 1, 5, 4};
    int v2[] = {3, 1, 5, 2, 0,
                4, 4, 1, 0, 3,
                0, 2, 7, 1, 9};
    int v3[] = {3, 1, 5, 2, 0,
                4, 4, 1, 0, 3,
                0, 2, 7, 1, 9};
    int e1[] = {13};
    int e20[] = {7, 7, 13, 3, 12};
    int e21[] = {11, 12, 19};
    int e30[] = {11, 12, 19};
    int e31[] = {7, 7, 13, 3, 12};
    int t3[] = {1, 0};

    a1 = array_alloc(ds1, 1, dtype);
    if (!a1) goto fail;
    cv1 = cast_test_values(v1, 4, dtype);
    array_fill_vals(a1, cv1, dtype);
    s1 = array_sum(a1, 0);
    if (!s1) goto fail;
    r1 = array_ravel(s1);
    ce1 = cast_test_values(e1, 1, dtype);
    if (arrays_equal(ce1, r1, 1, dtype)) goto fail;

    a2 = array_alloc(ds2, 2, dtype);
    if (!a2) goto fail;
    cv2 = cast_test_values(v2, prod(ds2, 2), dtype);
    array_fill_vals(a2, cv2, dtype);
    s20 = array_sum(a2, 0);
    s21 = array_sum(a2, 1);
    if (!s20) goto fail;
    if (!s21) goto fail;
    r20 = array_ravel(s20);
    r21 = array_ravel(s21);
    ce20 = cast_test_values(e20, 5, dtype);
    ce21 = cast_test_values(e21, 3, dtype);
    if (arrays_equal(ce20, r20, 5, dtype)) goto fail;
    if (arrays_equal(ce21, r21, 3, dtype)) goto fail;

    a3 = array_alloc(ds3, 2, dtype);
    if (!a3) goto fail;
    cv3 = cast_test_values(v3, prod(ds3, 2), dtype);
    array_fill_vals(a3, cv3, dtype);
    array_transpose(a3, t3);
    s30 = array_sum(a3, 0);
    s31 = array_sum(a3, 1);
    r30 = array_ravel(s30);
    r31 = array_ravel(s31);
    if (!s30) goto fail;
    if (!s31) goto fail;
    ce30 = cast_test_values(e30, 3, dtype);
    ce31 = cast_test_values(e31, 5, dtype);
    if (arrays_equal(ce30, r30, 3, dtype)) goto fail;
    if (arrays_equal(ce31, r31, 5, dtype)) goto fail;

    return 0;

fail:
    array_free(a1);
    array_free(a2);
    array_free(a3);
    array_free(s1);
    array_free(s20);
    array_free(s21);
    array_free(s30);
    array_free(s31);
    free(r1);
    free(r20);
    free(r21);
    free(r30);
    free(r31);
    free(cv1);
    free(cv2);
    free(cv3);
    free(ce1);
    free(ce20);
    free(ce21);
    free(ce30);
    free(ce31);
    return 1;
}

int test_dot(ARRAY_DTYPE dtype)
{
    arrayObject *a11 = NULL;
    arrayObject *a12 = NULL;
    arrayObject *a21 = NULL;
    arrayObject *a22 = NULL;
    arrayObject *a31 = NULL;
    arrayObject *a32 = NULL;
    arrayObject *d1 = NULL;
    arrayObject *d2 = NULL;
    arrayObject *d3 = NULL;
    void *cv11 = NULL;
    void *cv12 = NULL;
    void *cv21 = NULL;
    void *cv22 = NULL;
    void *cv31 = NULL;
    void *cv32 = NULL;
    void *rd1 = NULL;
    void *rd2 = NULL;
    void *rd3 = NULL;
    void *ce1 = NULL;
    void *ce2 = NULL;
    void *ce3 = NULL;
    int ds11[] = {1, 4};
    int ds12[] = {4};
    int ds21[] = {3, 5};
    int ds22[] = {5};
    int ds31[] = {2, 3};
    int ds32[] = {3, 4};
    int v11[] = {3, 1, 5, 4};
    int v12[] = {0, 2, 6, 3};
    int v21[] = {3, 1, 5, 2, 0,
                4, 4, 1, 0, 3,
                0, 2, 7, 1, 9};
    int v22[] = {1, 1, 0, 3, 2};
    int v31[] = {3, 1, 5,
                 2, 0, 4};
    int v32[] = {4, 1, 0, 2,
                 3, 3, 1, 1,
                 0, 2, 2, 6};
    int e1[] = {44};
    int e2[] = {10, 14, 23};
    int e3[] = {15, 16, 11, 37,
                8, 10, 8, 28};

    a11 = array_alloc(ds11, 2, dtype);
    a12 = array_alloc(ds12, 1, dtype);
    cv11 = cast_test_values(v11, 4, dtype);
    cv12 = cast_test_values(v12, 4, dtype);
    array_fill_vals(a11, cv11, dtype);
    array_fill_vals(a12, cv12, dtype);
    d1 = array_dot(a11, a12);
    rd1 = array_ravel(d1);
    ce1 = cast_test_values(e1, 1, dtype);
    if (arrays_equal(ce1, rd1, 1, dtype)) goto fail;

    a21 = array_alloc(ds21, 2, dtype);
    a22 = array_alloc(ds22, 1, dtype);
    cv21 = cast_test_values(v21, prod(ds21, 2), dtype);
    cv22 = cast_test_values(v22, 5, dtype);
    array_fill_vals(a21, cv21, dtype);
    array_fill_vals(a22, cv22, dtype);
    d2 = array_dot(a21, a22);
    if (!d2) goto fail;
    if (d2->dims[0] != 3 || d2->dims[1] != 1) goto fail;
    rd2 = array_ravel(d2);
    ce2 = cast_test_values(e2, 3, dtype);
    if (arrays_equal(ce2, rd2, 3, dtype)) goto fail;

    a31 = array_alloc(ds31, 2, dtype);
    a32 = array_alloc(ds32, 2, dtype);
    cv31 = cast_test_values(v31, prod(ds31, 2), dtype);
    cv32 = cast_test_values(v32, prod(ds32, 2), dtype);
    array_fill_vals(a31, cv31, dtype);
    array_fill_vals(a32, cv32, dtype);
    d3 = array_dot(a31, a32);
    if (!d3) goto fail;
    if (d3->dims[0] != 2 || d3->dims[1] != 4) goto fail;
    rd3 = array_ravel(d3);
    ce3 = cast_test_values(e3, 8, dtype);
    if (arrays_equal(ce3, rd3, 8, dtype)) goto fail;

    return 0;

fail:
    array_free(a11);
    array_free(a12);
    array_free(a21);
    array_free(a22);
    array_free(a31);
    array_free(a32);
    array_free(d1);
    array_free(d2);
    array_free(d3);
    free(cv11);
    free(cv12);
    free(cv21);
    free(cv22);
    free(cv31);
    free(cv32);
    free(rd1);
    free(rd2);
    free(rd3);
    free(ce1);
    free(ce2);
    free(ce3);

    return 1;
}

static void
run_test(int (*test)(ARRAY_DTYPE), char *test_name)
{
    int test_output_len = 30;
    char test_output[test_output_len];
    strcpy(test_output, test_name);
    int pos;
    for (pos=strlen(test_name); pos<test_output_len - 1; pos++) {
        test_output[pos] = '.';
    }
    test_output[pos] = '\n';
    printf("%s", test_output);

    for (int i = 0; i < NUM_ARRAY_DTYPES; i++) {
        char subtest_output[test_output_len];
        for (pos = 0; pos < 10; pos++) {
            subtest_output[pos] = ' ';
        }
        strcpy(subtest_output+pos, ARRAY_DTYPE_NAMES[i]);
        pos += strlen(ARRAY_DTYPE_NAMES[i]);
        for (; pos < test_output_len - 6; pos++) {
            subtest_output[pos] = '.';
        }
        if (test(ARRAY_DTYPES[i])) {
            sprintf(subtest_output + pos, "FAIL\n");
        } else {
            sprintf(subtest_output + pos, "PASS\n");
        }
        printf("%s", subtest_output);
    }
}

int main() {
    run_test(test_instantiation, "instantiation");
    run_test(test_fill, "fill");
    run_test(test_copy, "copy");
    run_test(test_transpose, "transpose");
    run_test(test_sum, "sum");
    run_test(test_dot, "dot");

    return 0;
}
