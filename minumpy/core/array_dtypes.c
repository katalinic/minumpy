#include "array_dtypes.h"

ARRAY_DTYPE ARRAY_DTYPES[NUM_ARRAY_DTYPES] = {INT32, INT64, FLOAT, DOUBLE};
const char *ARRAY_DTYPE_NAMES[NUM_ARRAY_DTYPES] = {"INT32", "INT64", "FLOAT", "DOUBLE"};
static size_t ARRAY_DTYPE_SIZES[NUM_ARRAY_DTYPES] = {
    sizeof(int32_t), sizeof(int64_t), sizeof(float), sizeof(double),
};

size_t
array_dtype_size(ARRAY_DTYPE dtype) {
    return ARRAY_DTYPE_SIZES[dtype];
}

int
array_dtype_valid(ARRAY_DTYPE dtype)
{
    return (dtype == UNKNOWN) ? 1 : 0;
}
