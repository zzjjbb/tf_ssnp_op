//
// Created by b on 11/3/2022.
//

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "ssnp_cuda.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

namespace functor {

// generate P matrix
template <typename T>
__host__ __device__ static __inline__ T near_0(const int i, const int size, const T pos_0, const bool edge) {
  return frac((static_cast<T>(i) + static_cast<T>(0.5) * !edge) / size + 0.5 - pos_0) - 0.5;
}

template <typename T>
__host__ __device__ static __inline__ T near_0(const int i, const int size) {
  return near_0(i, size, static_cast<T>(0), true);
}

template <typename T>
__host__ __device__ static __inline__ T c_gamma(const int i, const int j,
                                                const int i_size, const int j_size,
                                                const T i_res, const T j_res) {
  return sqrt(max(
    1 - (near_0<T>(i, i_size) / i_res) * (near_0<T>(i, i_size) / i_res)
      - (near_0<T>(j, j_size) / j_res) * (near_0<T>(j, j_size) / j_res), 1E-8));
}

template <typename T, bool A, bool B>
__device__ static __inline__ T make_p(T* pAB, const int i, const int j,
                                           const int i_size, const int j_size,
                                           const T i_res, const T j_res, const T z_res, const T dz) {
  T kz_ij, gamma_ij, eva_ij;
  for (int i : GpuGridRangeX(i_size))
    for (int j : GpuGridRangeY(j_size)) {
      gamma_ij = c_gamma(i, j, i_size, j_size, i_res, j_res);
      eva_ij = exp(min(gamma_ij * 5. - 1., 0.));
      kz_ij = gamma_ij * 2 * z_res;
      if (A ^ B) {
          if (A)  pAB[i * j_size + j] = sinpi(kz_ij * dz) / kz_ij * eva_ij;  // p10
          else  pAB[i * j_size + j] = -sinpi(kz_ij * dz) * kz_ij * eva_ij; // p01
      }
      else  pAB[i * j_size + j] = cospi(kz_ij * dz) * eva_ij; // p00 or p11
  }
}

// apply P operator
template <typename T>
__global__ void rotAngSpecCudaKernel(const int size, const int batch, T* a0, T* a1,
                                     const typename CmplxToRealType<T>::Type* p00,
                                     const typename CmplxToRealType<T>::Type* p01,
                                     const typename CmplxToRealType<T>::Type* p10) {
  T temp;
  for (int j = 0; j < batch * size; j += size)
    for (int i : GpuGridRangeX(size)) {
      temp = ldg(a0 + i + j) * ldg(p00 + i) + ldg(a1 + i + j) * ldg(p01 + i);
      a1[i] = ldg(a0 + i + j) * ldg(p10 + i) + ldg(a1 + i + j) * ldg(p00 + i);
      a0[i] = temp;
    }
}

template <typename T>
struct rotAngSpecFunctor<GPUDevice, T> {
  typedef typename CmplxToRealType<T>::Type realT;
  const GPUDevice& d_;
  const int size_, batch_;
  rotAngSpecFunctor(const GPUDevice& d, const int size, const int batch) : d_(d), size_(size), batch_(batch) {};
  Status operator()(T* a0, T* a1, const realT* p00, const realT* p01, const realT* p10) {
    auto cfg = GetGpuLaunchConfig(size_, d_);
    return GpuLaunchKernel(rotAngSpecCudaKernel<T>, cfg.block_count, cfg.thread_per_block, 0, d_.stream(),
                           size_, batch_, a0, a1, p00, p01, p10);
  }
};

template struct rotAngSpecFunctor<GPUDevice, complex64>;
template struct rotAngSpecFunctor<GPUDevice, complex128>;

template <typename T>
Status makePFunctor<GPUDevice, T>(const GPUDevice& d, int size,
                                  T* pAB, int i, int j, int i_size, int j_size,
                                  T i_res, T j_res, T z_res, T dz) {
  auto cfg = GetGpuLaunchConfig(size, d);
  return GpuLaunchKernel(make_p<T>, cfg.block_count, cfg.thread_per_block, 0, d.stream(),
                         pAB, i, j, i_size, j_size, i_res, j_res, z_res, dz);
};

}} // namespace functor, tensorflow

#endif  // GOOGLE_CUDA
