// Created by Jiabei, last modified 10/27/2022

#ifndef KERNELS_SSNP_CUFFT_H_
#define KERNELS_SSNP_CUFFT_H_

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow { namespace functor { namespace scatt_lib {

template<typename Device, typename T, bool FORWARD>
struct FFTFunctor {
  const int64_t size;
  const stream_executor::Stream *stream;
  std::unique_ptr<se::fft::Plan> plan;
  uint64 input_output_distance = 1;

  FFTFunctor(const TensorShape &shape, OpKernelContext *ctx);

  bool operator()(const T *in_mem, T *out_mem);
};

}}}  // namespace scatt_lib, functor, tensorflow

#endif //KERNELS_SSNP_CUFFT_H_
