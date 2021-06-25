C_DIR := minumpy/core
C_ARR_SRC := $(C_DIR)/array_dtypes.c $(C_DIR)/array_utils.c $(C_DIR)/array.c

build_c_test:
	gcc $(C_ARR_SRC) $(C_DIR)/test_array.c -o $(C_DIR)/test.o

build_c_benchmark:
	gcc $(C_ARR_SRC) $(C_DIR)/benchmark.c -o $(C_DIR)/benchmark.o

c_test:
	$(C_DIR)/test.o

c_benchmark:
	$(C_DIR)/benchmark.o
