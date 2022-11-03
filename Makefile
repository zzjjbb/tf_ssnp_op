CXX := /share/pkg.7/gcc/8.3.0/install/bin/g++
PYTHON_BIN_PATH := /projectnb/cisldnn/conda/envs/tf2/bin/python
CUDA_ROOT := /share/pkg.7/cuda/11.2/install
NVCC := $(CUDA_ROOT)/bin/nvcc

ZERO_OUT_SRCS = $(wildcard tensorflow_zero_out/cc/kernels/*.cc) $(wildcard tensorflow_zero_out/cc/ops/*.cc)
SSNP_SRCS = gpu_src/cc/kernels/time_two_kernels.cc $(wildcard gpu_src/cc/kernels/*.h) $(wildcard gpu_src/cc/ops/*.cc)

# TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
# TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

TF_CFLAGS := -I/projectnb/cisldnn/conda/envs/tf2/lib/python3.10/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64
TF_LFLAGS := -L/projectnb/cisldnn/conda/envs/tf2/lib/python3.10/site-packages/tensorflow -l:libtensorflow_framework.so.2

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = tensorflow_zero_out/python/ops/_zero_out_ops.so
SSNP_GPU_ONLY_TARGET_LIB = gpu_src/python/ops/_time_two_ops.cu.o
SSNP_TARGET_LIB = gpu_src/python/ops/_time_two_ops.so

# zero_out op for CPU
# 
# zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
# 	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py
# 
# zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
# 	./build_pip_pkg.sh make artifacts

ssnp_gpu_only: $(SSNP_GPU_ONLY_TARGET_LIB)

$(SSNP_GPU_ONLY_TARGET_LIB): gpu_src/cc/kernels/time_two_kernels.cu.cc
	$(NVCC) -ccbin $(CXX) -std=c++14 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

ssnp_op: $(SSNP_TARGET_LIB)
$(SSNP_TARGET_LIB): $(SSNP_SRCS) $(SSNP_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I$(CUDA_ROOT)/targets/x86_64-linux/include -L$(CUDA_ROOT)/targets/x86_64-linux/lib -lcudart

ssnp_test: SHELL := /bin/bash -l
ssnp_test: gpu_src/python/ops/time_two_ops_test.py gpu_src/python/ops/time_two_ops.py $(SSNP_TARGET_LIB)
	source /usr3/graduate/zjb/tf-dev; $(PYTHON_BIN_PATH) gpu_src/python/ops/time_two_ops_test.py

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(SSNP_GPU_ONLY_TARGET_LIB) $(SSNP_TARGET_LIB)
