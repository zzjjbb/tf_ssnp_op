// Created by Jiabei, last modified 10/27/2022
// Wrapper for cuFFT. Only 2D complex fft with dtype complex64/complex128 (C2C/Z2Z) is implemented.
// Support batch FFT2d (i.e. supports any dim_size but only calculates on the innermost 2 dims)

#include "ssnp_cufft.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

static constexpr bool DEBUG = true;

namespace tensorflow {

namespace {

// The following part is copied from https://github.com/tensorflow/tensorflow/blob/r2.9/tensorflow/core/kernels/fft_ops.cc
// Copyright 2015 The TensorFlow Authors.
// SPDX-License-Identifier: Apache-2.0
// Start of copy from TensorFlow implementation
template<typename T>
se::DeviceMemory<T> AsDeviceMemory(const T *cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T *>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

class CufftScratchAllocator : public se::ScratchAllocator {
public:
    ~CufftScratchAllocator() override {} // NOLINT(modernize-use-equals-default)

    CufftScratchAllocator(int64_t memory_limit, OpKernelContext *context)
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
    OpKernelContext *context_;
    std::vector<Tensor> allocated_tensors_;
};

// Start of copy from TensorFlow implementation

template<typename T>
constexpr se::fft::Type get_fft_type(bool forward) {
  return se::fft::Type::kInvalid;
}

template<>
constexpr se::fft::Type get_fft_type<complex64>(bool forward) {
  return forward ? se::fft::Type::kC2CForward : se::fft::Type::kC2CInverse;
}

template<>
constexpr se::fft::Type get_fft_type<complex128>(bool forward) {
  return forward ? se::fft::Type::kZ2ZForward : se::fft::Type::kZ2ZInverse;
}

} // end namespace

namespace functor {
namespace scatt_lib {
typedef Eigen::GpuDevice GPUDevice;

template<typename T, bool FORWARD>
struct FFTFunctor<GPUDevice, T, FORWARD> {
    static constexpr auto FFT_TYPE=get_fft_type<T>(FORWARD);

    int64_t size;
    stream_executor::Stream *stream;
    std::unique_ptr<se::fft::Plan> plan;
    uint64 input_output_distance = 1;
    FFTFunctor(const TensorShape &shape, OpKernelContext *ctx) :
      size(shape.num_elements()), stream(ctx->op_device_context()->stream())
    {
      static const int FFT_RANK = 2;
      static const uint64 STRIDE = 1;
      static const bool in_place_fft = false;

      OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));


      int dims = shape.dims();
      uint64 fft_shape[2], *input_output_embed = fft_shape;
      for (int i = 0; i < FFT_RANK; ++i) {
        fft_shape[i] = static_cast<uint64>(shape.dim_size(dims - FFT_RANK + i));
        input_output_distance *= fft_shape[i];
      }

      int batch_count = 1;
      for (int i = 0; i < dims - FFT_RANK; ++i)
        batch_count *= static_cast<int>(shape.dim_size(i));

      // Worst case of cuFFT plan temp memory: 8*batch*n[0]*..*n[rank-1] items (NOT bytes!)
      // Best case: 1*batch*n[0]*..*n[rank-1]
      // See https://docs.nvidia.com/cuda/cufft/index.html#cufft-setup
      // Typical case (if worst): 1536MB <-- 24 LED (batch), 1024 * 1024 (image dim), complex128
      // TODO (Jiabei): investigate how scratch_allocator works, and exit gracefully rather than crash with segfault
      CufftScratchAllocator scratch_allocator(1LL<<32, ctx); // gives 4GB limit should be fine?

      if (DEBUG) {
        LOG(INFO) << "Making FFT plan: forward:" << FORWARD
                  << " dtype: complex" << sizeof(T) * 8
                  << " shape:(" << fft_shape[0] << ", " << fft_shape[1] << ")"
                  << " input_distance/output_distance:" << input_output_distance
                  << " batch_count:" << batch_count;
      };

      plan = stream->parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
          stream, FFT_RANK, fft_shape,
          input_output_embed, STRIDE, input_output_distance, // input (same as output)
          input_output_embed, STRIDE, input_output_distance, // output (same as input)
          FFT_TYPE, in_place_fft, batch_count, &scratch_allocator);
      OP_REQUIRES(
          ctx, plan != nullptr,
          errors::Internal("Failed to create cuFFT batched plan with scratch allocator"));
    }

    bool operator()(const T *in_mem, T *out_mem) {
      auto src = AsDeviceMemory<T>(in_mem, size);
      auto dst = AsDeviceMemory<T>(out_mem, size);
      return stream->ThenFft(plan.get(), src, &dst).ok(); // NOLINT(readability-redundant-smartptr-get)
    }
};
template struct FFTFunctor<GPUDevice, complex64, true>;
template struct FFTFunctor<GPUDevice, complex128, true>;
template struct FFTFunctor<GPUDevice, complex64, false>;
template struct FFTFunctor<GPUDevice, complex128, false>;
}}  // namespace scatt_lib, functor
}  // namespace tensorflow