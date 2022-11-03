/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

#include "time_two.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T>
struct TestFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out, OpKernelContext* ctx) {
    for (int i = 0; i < size; ++i) {
      out[i] = (T)2. * in[i];
    }
  }
};

namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
class CufftScratchAllocator : public se::ScratchAllocator {
 public:
  ~CufftScratchAllocator() override {}
  CufftScratchAllocator(int64_t memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  int64_t GetMemoryLimitInBytes() override { return memory_limit_; }
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64_t byte_size) override {
    Tensor temporary_memory;
    if (byte_size > memory_limit_) {
      return se::port::StatusOr<se::DeviceMemory<uint8>>();
    }
    AllocationAttributes allocation_attr;
    allocation_attr.retry_on_failure = false;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<se::DeviceMemory<uint8>>();
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return se::port::StatusOr<se::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64_t TotalByteSize() { return total_byte_size_; }

 private:
  int64_t memory_limit_;
  int64_t total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};
} // end namespace

template <typename T>
struct TestFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, int size, const T* in_mem, T* out_mem, OpKernelContext* ctx) {
		uint64 fft_shape[2] = {4, 4};
		uint64 input_embed[2] = {4,4};
		uint64 output_embed[2] = {4,4};
		auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
		const Tensor& in = ctx->input(0);
		const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = in.shape();
		CufftScratchAllocator scratch_allocator(1<<30, ctx);
		auto plan =
        stream->parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
            stream, 2, fft_shape, input_embed, 1,
            16, output_embed, 1, 16,
            se::fft::Type::kZ2ZForward, false, 1, &scratch_allocator);
		OP_REQUIRES(
        ctx, plan != nullptr,
        errors::Internal(
            "Failed to create cuFFT batched plan with scratch allocator"));
		auto src = AsDeviceMemory<complex128>(in_mem, input_shape.num_elements());
		auto dst = AsDeviceMemory<complex128>(out_mem,
                                    output_shape.num_elements());
		OP_REQUIRES(
        ctx, stream->ThenFft(plan.get(), src, &dst).ok(),
        errors::Internal("fft failed : type=", static_cast<int>(se::fft::Type::kZ2ZForward),
                         " in.shape=", input_shape.DebugString()));

	}
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class TimeTwoOp : public OpKernel {
 public:
  explicit TimeTwoOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    TestFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data(),
				context
		);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SSNPTest").Device(DEVICE_CPU).TypeConstraint<T>("Field"), \
      TimeTwoOp<CPUDevice, T>);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SSNPTest").Device(DEVICE_GPU).TypeConstraint<T>("Field"), \
      TimeTwoOp<GPUDevice, T>);
  // extern template struct TestFunctor<GPUDevice, T>;           
//REGISTER_GPU(complex64);
REGISTER_GPU(complex128);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
