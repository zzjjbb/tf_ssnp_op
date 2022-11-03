// Created by Jiabei, last modified 10/27/2022
//

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "ssnp.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace functor {

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename DTYPE, bool FORWARD>
class SSNPOp : public OpKernel {
public:
  // This is for testing purposes about cuFFT only
  explicit SSNPOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &u1_tensor = context->input(0);
    const Tensor &u2_tensor = context->input(1);

    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

//    Tensor *u2_tensor = nullptr;
//    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<DTYPE>::value, input_tensor.shape(),
//                                                   &u2_tensor));

    // Do the computation.
    auto fft = FFTFunctor<Device, DTYPE, FORWARD>(input_tensor.shape(), context);
    OP_REQUIRES(
        context,
        fft(input_tensor.flat<DTYPE>().data(), output_tensor->flat<DTYPE>().data()),
        errors::Internal(FORWARD ? "forward fft failed, " : "inverse fft failed, ",
                         "in.shape=", input_tensor.shape().DebugString()));
  }
}; // SSNPOp

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                               \
    extern template struct FFTFunctor<GPUDevice, T, true>;            \
    extern template struct FFTFunctor<GPUDevice, T, false>;           \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("SSNP").Device(DEVICE_GPU).TypeConstraint<T>("Field"),  \
      FFTTestOp<GPUDevice, T, true>);                                 \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("IFFTTest").Device(DEVICE_GPU).TypeConstraint<T>("Field"), \
      FFTTestOp<GPUDevice, T, false>);
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)
#endif  // GOOGLE_CUDA

}} // namespace functor, tensorflow
