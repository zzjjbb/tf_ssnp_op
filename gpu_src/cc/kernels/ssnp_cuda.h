// Created by Jiabei, last modified on 01/25/2023
//

#ifndef KERNELS_SSNP_CUDA_H_
#define KERNELS_SSNP_CUDA_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow { namespace functor { namespace scatt_lib {

struct PhysParam {
  float i_res, j_res, z_res, dz;
};

template<class T>
struct CmplxToRealEnum {};

template<>
struct CmplxToRealEnum<complex64> {
  static constexpr DataType value = DT_FLOAT;
};

template<>
struct CmplxToRealEnum<complex128> {
  static constexpr DataType value = DT_DOUBLE;
};

template<class T>
struct CmplxToRealType {};

template<>
struct CmplxToRealType<complex64> {
  typedef float Type;
};

template<>
struct CmplxToRealType<complex128> {
  typedef double Type;
};

template<typename RT>
Status make_p(const Eigen::GpuDevice &d, RT *pAB, int i_size, int j_size, const PhysParam *);

template<typename Device, typename T>
struct rotAngSpecFunctor {
  typedef typename CmplxToRealType<T>::Type realT;
  const Device &d_;
  const int size_, batch_;

  rotAngSpecFunctor(const Device &d, const int size, const int batch);

  Status operator()(T *a0, T *a1, const realT *pAB);
};

}  // namespace scatt_lib
}}  // namespace functor, tensorflow


#endif //KERNELS_SSNP_CUDA_H_
