#ifndef ARRAY_DTYPES_H
#define ARRAY_DTYPES_H

#include <stdio.h>

#define NUM_ARRAY_DTYPES 4
typedef enum {INT32, INT64, FLOAT, DOUBLE, UNKNOWN} ARRAY_DTYPE;

extern ARRAY_DTYPE ARRAY_DTYPES[NUM_ARRAY_DTYPES];
extern const char *ARRAY_DTYPE_NAMES[NUM_ARRAY_DTYPES];

size_t array_dtype_size(ARRAY_DTYPE dtype);
int array_dtype_valid(ARRAY_DTYPE dtype);

#endif
