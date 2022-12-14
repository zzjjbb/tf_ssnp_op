// Created by Jiabei, last modified 10/27/2022
//

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "ssnp_cufft.h"
#include "ssnp_cuda.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

typedef Eigen::GpuDevice GPUDevice;

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
    const Tensor &i1_tensor = context->input(0);
    const Tensor &i2_tensor = context->input(1);
    const auto shape = i1_tensor.shape();
    const Device d = context->eigen_device<Device>();
    const int dims = shape.dims();
    OP_REQUIRES(context, shape.num_elements() < 2LL<<30, errors::InvalidArgument("input arrays are too large"));
    OP_REQUIRES(context, shape == i2_tensor.shape(),
                errors::InvalidArgument("shapes of u1 ", shape.DebugString(),
                                        " and u2 ", i2_tensor.shape().DebugString(),
                                        " must be the same"));
    OP_REQUIRES(context, dims >= 2, errors::InvalidArgument("input arrays should have at least 2 dims"));

    // Create an output tensor
    Tensor *o1_tensor, *o2_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape,
                                                     &o1_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, shape,
                                                     &o2_tensor));

    // Create p tensor
    Tensor p00_tensor, p01_tensor, p10_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(CmplxToRealEnum<DTYPE>::value, shape, &p00_tensor));
    OP_REQUIRES_OK(context, context->allocate_temp(CmplxToRealEnum<DTYPE>::value, shape, &p01_tensor));
    OP_REQUIRES_OK(context, context->allocate_temp(CmplxToRealEnum<DTYPE>::value, shape, &p10_tensor));


//    Tensor *u2_tensor = nullptr;
//    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<DTYPE>::value, input_tensor.shape(),
//                                                   &u2_tensor));

    // Do the computation.
    auto fft = FFTFunctor<Device, DTYPE, true>(shape, context);
    auto ifft = FFTFunctor<Device, DTYPE, false>(shape, context);
    auto rot_ang_spec = rotAngSpecFunctor<Device, DTYPE>(d, shape.num_elements(), 1);
    LOG(INFO) << "rotAngSpec kernel function, size " << shape.num_elements() << " batch " << 1;
    // give the memory a name related to physics quantity
    auto *u1 = i1_tensor.flat<DTYPE>().data(), *u2 = i2_tensor.flat<DTYPE>().data();
    auto *a1 = o1_tensor->flat<DTYPE>().data(), *a2 = o2_tensor->flat<DTYPE>().data();
    auto *p00 = p00_tensor.flat<typename CmplxToRealType<DTYPE>::Type>().data();
    auto *p01 = p01_tensor.flat<typename CmplxToRealType<DTYPE>::Type>().data();
    auto *p10 = p10_tensor.flat<typename CmplxToRealType<DTYPE>::Type>().data();
    string shape_str = shape.DebugString();
    auto fft_err = errors::Internal("forward fft failed, ssnp shape=", shape_str);
    auto ifft_err = errors::Internal("inverse fft failed, ssnp shape=", shape_str);
    for (int i=0; i<1; i++) {
      OP_REQUIRES(context, fft(u1, a1), fft_err);
      OP_REQUIRES(context, fft(u2, a2), fft_err);
      rot_ang_spec(a1, a2, p00, p01, p10);
      OP_REQUIRES(context, ifft(a1, a1), ifft_err);
      OP_REQUIRES(context, ifft(a2, a2), ifft_err);
      u1 = a1; u2 = a2; // all in-place operation from now
    }
  }
}; // SSNPOp

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                               \
    extern template struct FFTFunctor<GPUDevice, T, true>;            \
    extern template struct FFTFunctor<GPUDevice, T, false>;           \
    extern template struct rotAngSpecFunctor<GPUDevice, T>;           \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("SSNP").Device(DEVICE_GPU).TypeConstraint<T>("Field"),     \
      SSNPOp<GPUDevice, T, true>);

REGISTER_GPU(complex64)
REGISTER_GPU(complex128)
#endif  // GOOGLE_CUDA

}} // namespace functor, tensorflow
