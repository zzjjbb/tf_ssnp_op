// Created by Jiabei, last modified on 11/11/2022
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
__host__ __device__ static __inline__ T near_0(int i, int size, T pos_0, bool edge) {
  return frac((static_cast<T>(i) + static_cast<T>(0.5) * !edge) / size + 0.5 - pos_0) - 0.5;
}

template <typename T>
__host__ __device__ static __inline__ T near_0(int i, int size) {
  return near_0(i, size, static_cast<T>(0), true);
}

template <typename RT>
__host__ __device__ static __inline__ RT c_gamma(int i, int j, int i_size, int j_size,
                                                 RT i_res, RT j_res) {
  return sqrt(max(
    1 - (near_0<RT>(i, i_size) / i_res) * (near_0<RT>(i, i_size) / i_res)
      - (near_0<RT>(j, j_size) / j_res) * (near_0<RT>(j, j_size) / j_res), 1E-8));
}

template <typename RT, bool A, bool B>
__device__ static __inline__ RT make_p(RT* pAB, int i, int j, int i_size, int j_size,
                                       RT i_res, RT j_res, RT z_res, RT dz) {
  RT kz_ij, gamma_ij, eva_ij;
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
template <typename CT>
__global__ void rotAngSpecCudaKernel(const int size, const int batch, CT* a0, CT* a1,
                                     const typename CmplxToRealType<CT>::Type* p00,
                                     const typename CmplxToRealType<CT>::Type* p01,
                                     const typename CmplxToRealType<CT>::Type* p10) {
  CT temp;
  for (int j = 0; j < batch * size; j += size)
    for (int i : GpuGridRangeX(size)) {
      temp = ldg(a0 + i + j) * ldg(p00 + i) + ldg(a1 + i + j) * ldg(p01 + i);
      a1[i] = ldg(a0 + i + j) * ldg(p10 + i) + ldg(a1 + i + j) * ldg(p00 + i);
      a0[i] = temp;
    }
}

template <typename CT>
struct rotAngSpecFunctor<GPUDevice, CT> {
  typedef typename CmplxToRealType<CT>::Type realT;
  const GPUDevice& d_;
  const int size_, batch_;
  rotAngSpecFunctor(const GPUDevice& d, const int size, const int batch) : d_(d), size_(size), batch_(batch) {};
  Status operator()(CT* a0, CT* a1, const realT* p00, const realT* p01, const realT* p10) {
    auto cfg = GetGpuLaunchConfig(size_, d_);
    return GpuLaunchKernel(rotAngSpecCudaKernel<CT>, cfg.block_count, cfg.thread_per_block, 0, d_.stream(),
                           size_, batch_, a0, a1, p00, p01, p10);
  }
};

template struct rotAngSpecFunctor<GPUDevice, complex64>;
template struct rotAngSpecFunctor<GPUDevice, complex128>;

template <typename RT>
Status makePFunctor<GPUDevice, RT>(const GPUDevice& d, int size,
                                  RT* pAB, int i, int j, int i_size, int j_size,
                                  RT i_res, RT j_res, RT z_res, RT dz) {
  auto cfg = GetGpuLaunchConfig(size, d);
  return GpuLaunchKernel(make_p<RT>, cfg.block_count, cfg.thread_per_block, 0, d.stream(),
                         pAB, i, j, i_size, j_size, i_res, j_res, z_res, dz);
};

}} // namespace functor, tensorflow

#endif  // GOOGLE_CUDA
