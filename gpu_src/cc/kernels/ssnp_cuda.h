//
// Created by b on 11/10/2022.
//

#ifndef KERNELS_SSNP_CUDA_H_
#define KERNELS_SSNP_CUDA_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {

template <class T>
struct CmplxToRealEnum {};

template <>
struct CmplxToRealEnum<complex64> {
  static constexpr DataType value = DT_FLOAT;
};

template <>
struct CmplxToRealEnum<complex128> {
  static constexpr DataType value = DT_DOUBLE;
};

template <class T>
struct CmplxToRealType {};

template <>
struct CmplxToRealType<complex64> {
  typedef float Type;
};

template <>
struct CmplxToRealType<complex128> {
  typedef double Type;
};

template <typename Device, typename T>
Status makePFunctor(const Device& d, int size,
                    T* pAB, int i, int j, int i_size, int j_size, T i_res, T j_res, T z_res, T dz);

template <typename Device, typename T>
struct rotAngSpecFunctor {
  typedef typename CmplxToRealType<T>::Type realT;
  const Device& d_;
  const int size_, batch_;
  rotAngSpecFunctor(const Device& d, const int size, const int batch);
  Status operator()(T* a0, T* a1, const realT* p00, const realT* p01, const realT* p10);
};

}}  // namespace functor, tensorflow


#endif //KERNELS_SSNP_CUDA_H_
