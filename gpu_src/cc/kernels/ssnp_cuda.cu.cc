// Created by Jiabei, last modified on 11/11/2022
//

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <math_constants.h>
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "ssnp_cuda.h"

namespace tensorflow { namespace functor { namespace scatt_lib {

typedef Eigen::GpuDevice GPUDevice;

__constant__ PhysParam phys_param;

// generate P matrix
template<typename T>
__host__ __device__ __forceinline__ T near_0(int i, int size, T pos_0, bool edge) {
  T unused;
  return modf((static_cast<T>(i) + 0.5 * !edge) / size + 0.5 - pos_0, &unused) - 0.5;
}

template<typename T>
__host__ __device__ __forceinline__ T near_0(int i, int size) {
  return near_0(i, size, static_cast<T>(0), true);
}

template<typename RT>
__host__ __device__ __forceinline__ RT c_gamma(int i, int j, int i_size, int j_size) {
  return sqrt(max(
      1 - (near_0<RT>(i, i_size) / phys_param.i_res) * (near_0<RT>(i, i_size) / phys_param.i_res)
      - (near_0<RT>(j, j_size) / phys_param.j_res) * (near_0<RT>(j, j_size) / phys_param.j_res), 1E-8));
}

template<typename RT>
__global__ void make_p_kernel(RT *pAB, int i_size, int j_size) {
  RT kz_ij, gamma_ij, eva_ij, *ptr_ij;
  for (int i: GpuGridRangeX(i_size))
    for (int j: GpuGridRangeY(j_size)) {
      gamma_ij = c_gamma<RT>(i, j, i_size, j_size);
      eva_ij = exp(min(gamma_ij * 20. - 4., static_cast<RT>(0)));
      kz_ij = gamma_ij * 2 * CUDART_PI * phys_param.z_res;
      ptr_ij = pAB + (i * j_size + j) * 3;
      ptr_ij[0] = cos(kz_ij * phys_param.dz) * eva_ij; // p00 and p11
      ptr_ij[1] = -sin(kz_ij * phys_param.dz) * kz_ij * eva_ij; // p01
      ptr_ij[2] = sin(kz_ij * phys_param.dz) / kz_ij * eva_ij;  // p10
    }
}

template<typename RT>
Status make_p(const GPUDevice &d, RT *pAB, int i_size, int j_size,
              const PhysParam *res_and_dz_input) {
  auto cfg = GetGpu2DLaunchConfig(i_size, j_size, d);
  cudaMemcpyToSymbol(phys_param, res_and_dz_input, sizeof(PhysParam));
  return GpuLaunchKernel(make_p_kernel<RT>, cfg.block_count, cfg.thread_per_block, 0, d.stream(),
                         pAB, i_size, j_size);
};

template Status make_p(const GPUDevice &, float *, int, int, const PhysParam *);

template Status make_p(const GPUDevice &, double *, int, int, const PhysParam *);
//template struct rotAngSpecFunctor<GPUDevice, complex128>;

// apply P operator
template<typename CT>
__global__ void rotAngSpecCudaKernel(int size, int batch, CT *a0, CT *a1,
                                     const typename CmplxToRealType<CT>::Type *p) {
  CT temp;
  for (int j = 0; j < batch * size; j += size)
    for (int i: GpuGridRangeX(size)) {
      // new_a0 = a0 * p00 + a1 * p01
      temp = ldg(a0 + i + j) * ldg(p + i * 3) + ldg(a1 + i + j) * ldg(p + i * 3 + 1);
      // new_a1 = a0 * p10 + a1 * p11
      a1[i + j] = ldg(a0 + i + j) * ldg(p + i * 3 + 2) + ldg(a1 + i + j) * ldg(p + i * 3);
      a0[i + j] = temp;
    }
}

template<typename CT>
struct rotAngSpecFunctor<GPUDevice, CT> {
  typedef typename CmplxToRealType<CT>::Type realT;
  const GPUDevice &d_;
  const int size_, batch_;

  rotAngSpecFunctor(const GPUDevice &d, const int size, const int batch) : d_(d), size_(size), batch_(batch) {};

  Status operator()(CT *a0, CT *a1, const realT *p) {
    auto cfg = GetGpuLaunchConfig(size_, d_);
    return GpuLaunchKernel(rotAngSpecCudaKernel<CT>, cfg.block_count, cfg.thread_per_block, 0, d_.stream(),
                           size_, batch_, a0, a1, p);
  }
};

template
struct rotAngSpecFunctor<GPUDevice, complex64>;
template
struct rotAngSpecFunctor<GPUDevice, complex128>;


}}} // namespace scatt_lib, functor, tensorflow

#endif  // GOOGLE_CUDA
