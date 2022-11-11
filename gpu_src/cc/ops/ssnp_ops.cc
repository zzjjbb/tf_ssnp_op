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

REGISTER_OP("SSNP")
    .Attr("Field: {complex64, complex128}")
    .Attr("res: list(float) >= 3")
    .Input("u1_i: Field")
    .Input("u2_i: Field")
    .Output("u1_o: Field")
    .Output("u2_o: Field")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });