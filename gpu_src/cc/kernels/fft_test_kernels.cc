// Created by Jiabei, last modified 01/30/2023
//

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "ssnp_cufft.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow { namespace functor { namespace scatt_lib {

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename T, bool FORWARD>
class FFTTestOp : public OpKernel {
public:
  // This is for testing purposes about cuFFT only
  explicit FFTTestOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    LOG(WARNING) << "FFTTest is for SSNP library testing only, please use tf.signal.fft2d in computation";
    // Grab the input tensor
    const Tensor &input_tensor = context->input(0);

    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    auto fft = FFTFunctor<Device, T, FORWARD>(input_tensor.shape(), context);
    OP_REQUIRES(
        context,
        fft(input_tensor.flat<T>().data(), output_tensor->flat<T>().data()),
        errors::Internal(FORWARD ? "forward fft failed, " : "inverse fft failed, ",
                         "in.shape=", input_tensor.shape().DebugString()));
  }
};

// Register the CPU kernels.
//#define REGISTER_CPU(T)                                          \
//  REGISTER_KERNEL_BUILDER(                                       \
//      Name("SSNPTest").Device(DEVICE_CPU).TypeConstraint<T>("Field"), \
//      TimeTwoOp<CPUDevice, T>);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                               \
    extern template struct FFTFunctor<GPUDevice, T, true>;            \
    extern template struct FFTFunctor<GPUDevice, T, false>;           \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("ScattLibTestFFT").Device(DEVICE_GPU).TypeConstraint<T>("Field"),  \
      FFTTestOp<GPUDevice, T, true>);                                 \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("ScattLibTestIFFT").Device(DEVICE_GPU).TypeConstraint<T>("Field"), \
      FFTTestOp<GPUDevice, T, false>);
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA
}}} // namespace functor, tensorflow