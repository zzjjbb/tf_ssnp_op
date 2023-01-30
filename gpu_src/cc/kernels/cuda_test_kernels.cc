// Created by Jiabei, last modified on 11/11/2022
//

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "ssnp_cuda.h"

namespace tensorflow { namespace functor { namespace scatt_lib {

typedef Eigen::GpuDevice GPUDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename DTYPE>
class MakePOp : public OpKernel {
public:
  // This is for testing purposes about cuFFT only
  explicit MakePOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("res", &res));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dz", &dz));
  }

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    const Tensor &i1_tensor = ctx->input(0);
    const auto &shape = i1_tensor.shape();
    auto p_shape = shape;
    p_shape.AddDim(3);
    const Device &d = ctx->eigen_device<Device>();
    const int dims = shape.dims();
    OP_REQUIRES(ctx, shape.dim_size(0) < 2LL << 30, errors::InvalidArgument("input arrays are too large"));
    OP_REQUIRES(ctx, shape.dim_size(1) < 2LL << 30, errors::InvalidArgument("input arrays are too large"));
    OP_REQUIRES(ctx, dims >= 2, errors::InvalidArgument("input arrays should have at least 2 dims"));

    // Create an output tensor
    Tensor *p_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, p_shape, &p_tensor));

    // Do the computation.
    auto *pAB = p_tensor->flat<DTYPE>().data();
    PhysParam param = {res[0], res[1], res[2], dz};
    make_p(d, pAB, shape.dim_size(0), shape.dim_size(1), &param);
#ifdef SCATT_LIB_DEBUG
    string shape_str = p_shape.DebugString();
    LOG(INFO) << "p_shape " << shape_str << " p dtype " << typeid(pAB).name();
#endif
  }
private:
  std::vector<float> res;
  float dz;
}; // MakePOp

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                               \
    extern template Status make_p(const GPUDevice &, CmplxToRealType<T>::Type *, int, int, const PhysParam *); \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("ScattLibTestMakeP").Device(DEVICE_GPU).TypeConstraint<CmplxToRealType<T>::Type>("dtype"),     \
      MakePOp<GPUDevice, CmplxToRealType<T>::Type>);

REGISTER_GPU(complex64)

REGISTER_GPU(complex128)
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA

}}} // namespace scatt_lib, functor, tensorflow