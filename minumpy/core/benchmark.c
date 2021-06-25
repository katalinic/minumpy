#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "array.h"
#include "array_dtypes.h"

int main() {
    for (int i = 0; i < NUM_ARRAY_DTYPES; i++) {
        ARRAY_DTYPE dtype = ARRAY_DTYPES[i];
        const char *dtype_name = ARRAY_DTYPE_NAMES[i];

        int N = 512;
        int ds_a[] = {N, N};
        int ds_b[] = {N, N};
        arrayObject *a = array_alloc(ds_a, 2, dtype);
        arrayObject *b = array_alloc(ds_b, 2, dtype);

        array_fill_uniform_int(a, 0, 1, dtype);
        array_fill_uniform_int(b, 0, 1, dtype);

        clock_t start_time = clock();
        arrayObject *d = array_dot(a, b);
        double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Dot %s %f seconds\n", dtype_name, elapsed_time);

        assert(d != NULL);
        assert(d->dims[0] == N && d->dims[1] == N);

        start_time = clock();
        arrayObject *d1 = array_sum(d, 1);
        elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Sum 1 %s %f seconds\n", dtype_name, elapsed_time);

        start_time = clock();
        arrayObject *d2 = array_sum(d1, 0);
        elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Sum 0 %s %f seconds\n", dtype_name, elapsed_time);

        printf("%s\n", array_str(d2));

        array_free(a);
        array_free(b);
        array_free(d);
        array_free(d1);
        array_free(d2);
    }

    return 0;
}
