CXX := /share/pkg.7/gcc/8.3.0/install/bin/g++
PYTHON_BIN_PATH := python
CUDA_ROOT := /share/pkg.7/cuda/11.2/install
NVCC := $(CUDA_ROOT)/bin/nvcc

# TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
# TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

TF_CFLAGS := -I/projectnb/cisldnn/conda/envs/tf2/lib/python3.10/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64
TF_LFLAGS := -L/projectnb/cisldnn/conda/envs/tf2/lib/python3.10/site-packages/tensorflow -l:libtensorflow_framework.so.2

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14 -Wall
LDFLAGS = -shared ${TF_LFLAGS}

CUDA_CFLAGS = $(CFLAGS) -I$(CUDA_ROOT)/targets/x86_64-linux/include
CUDA_LDFLAGS = $(LDFLAGS) -L$(CUDA_ROOT)/targets/x86_64-linux/lib -lcudart


SSNP_KERNELS_DIR = gpu_src/cc/kernels
SSNP_HEADER = $(SSNP_KERNELS_DIR)/ssnp.h
SSNP_CUFFT = $(SSNP_KERNELS_DIR)/ssnp_cufft.cc
SSNP_CUDA = $(SSNP_KERNELS_DIR)/ssnp_kernels.cu.cc
FFT_TEST_KERNELS = $(SSNP_KERNELS_DIR)/fft_test_kernels.cc
SSNP_KERNELS = $(SSNP_KERNELS_DIR)/ssnp_kernels.cc
SSNP_OPS = $(wildcard gpu_src/cc/ops/*.cc)

SSNP_LIB_DIR = gpu_src/python/ops
SSNP_CUFFT_LIB = $(SSNP_LIB_DIR)/_ssnp_cufft.o
SSNP_CUDA_LIB = $(SSNP_LIB_DIR)/_ssnp_kernels.cu.o
FFT_TEST_KERNELS_LIB = $(SSNP_LIB_DIR)/_fft_kernels.o
SSNP_KERNELS_LIB = $(SSNP_LIB_DIR)/_ssnp_kernels.o
SSNP_OPS_LIB = $(SSNP_LIB_DIR)/_ssnp_ops.so
SSNP_LIB_ALL = $(SSNP_CUFFT_LIB) $(SSNP_CUDA_LIB) $(SSNP_KERNELS_LIB) $(SSNP_OPS_LIB)

# zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
# 	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py
# 
# zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
# 	./build_pip_pkg.sh make artifacts

$(SSNP_CUDA_LIB): $(SSNP_CUDA) $(SSNP_HEADER)
	$(NVCC) -ccbin $(CXX) -std=c++14 -c -o $@ $<  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

$(SSNP_CUFFT_LIB): $(SSNP_CUFFT) $(SSNP_HEADER)
	$(CXX) $(CUDA_CFLAGS) -o $@ $< $(CUDA_LDFLAGS) -D GOOGLE_CUDA=1

$(FFT_TEST_KERNELS_LIB): $(FFT_TEST_KERNELS) $(SSNP_HEADER)
	$(CXX) $(CUDA_CFLAGS) -o $@ $< $(CUDA_LDFLAGS) -D GOOGLE_CUDA=1

$(SSNP_KERNELS_LIB): $(SSNP_KERNELS) $(SSNP_HEADER)
	$(CXX) $(CUDA_CFLAGS) -o $@ $< $(CUDA_LDFLAGS) -D GOOGLE_CUDA=1

$(SSNP_OPS_LIB): $(SSNP_OPS) $(SSNP_CUFFT_LIB) $(FFT_TEST_KERNELS_LIB) $(SSNP_KERNELS_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) -D GOOGLE_CUDA=1

build_ssnp_ops: $(SSNP_OPS_LIB)

ssnp_test: SHELL := /bin/bash -l
ssnp_test: gpu_src/python/ops/ssnp_ops_test.py gpu_src/python/ops/ssnp_ops.py $(SSNP_OPS_LIB)
	source /usr3/graduate/zjb/tf-dev; $(PYTHON_BIN_PATH) gpu_src/python/ops/ssnp_ops_test.py

clean:
	rm -f $(SSNP_LIB_ALL)
