#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("FFTTest")
    .Attr("Field: {complex64, complex128}")
    .Input("in: Field")
    .Output("out: Field")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("IFFTTest")
    .Attr("Field: {complex64, complex128}")
    .Input("in: Field")
    .Output("out: Field")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

//REGISTER_OP("SSNP")
//    .Attr("Field: {complex64, complex128}")
//    .Input("in: Field")
//    .Output("out: Field")
//    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//        c->set_output(0, c->input(0));
//        return Status::OK();
//    });