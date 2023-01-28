// Created by Jiabei, last modified 01/10/2023
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using shape_inference::InferenceContext;

REGISTER_OP("ScattLibTestFFT")
    .Attr("Field: {complex64, complex128}")
    .Input("in: Field")
    .Output("out: Field")
    .SetShapeFn([](InferenceContext* c) {
        return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });

REGISTER_OP("ScattLibTestIFFT")
    .Attr("Field: {complex64, complex128}")
    .Input("in: Field")
    .Output("out: Field")
    .SetShapeFn([](InferenceContext* c) {
        return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });

REGISTER_OP("ScattLibTestMakeP")
        .Attr("Field: {complex64, complex128}")
        .Attr("P: {float, double}")
        .Input("in: Field")
        .Output("out: P")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

REGISTER_OP("ScattLibSSNP")
    .Attr("Field: {complex64, complex128}")
    .Attr("res: list(float) >= 3")
    .Input("u1_i: Field")
    .Input("u2_i: Field")
    .Output("u1_o: Field")
    .Output("u2_o: Field")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });