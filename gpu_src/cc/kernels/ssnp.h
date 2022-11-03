// Created by Jiabei, last modified 10/27/2022

#ifndef KERNELS_SSNP_H_
#define KERNELS_SSNP_H_

#include <unsupported/Eigen/CXX11/Tensor>
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class OpKernelContext;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename T, bool FORWARD>
struct FFTFunctor {
  const int64_t size;
  const stream_executor::Stream *stream;
  std::unique_ptr<se::fft::Plan> plan;
  uint64 input_output_distance = 1;
  FFTFunctor(const TensorShape &shape, OpKernelContext *ctx);
  bool operator()(const T* in_mem, T* out_mem);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNELS_SSNP_H_
