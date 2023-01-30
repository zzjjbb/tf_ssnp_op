"""
Created by Jiabei, last modified 01/29/2023

Tests for SSNP ops.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops, test_util
from tensorflow.python.platform import test, tf_logging

try:
    from . import ssnp_ops
except ImportError:
    import ssnp_ops


class MakePTest(test.TestCase):
    """
    Test for tf.scatt_lib._makeP
    """
    XY_SHAPE = (1024, 1024)
    BATCH_SHAPE = (8, 1, 3)

    def makeP_test_body(self, shape, dtype):
        with self.cached_session(force_gpu=True):
            in_tensor = tf.zeros(shape, dtype=dtype)
            out_tensor = tf.scatt_lib._makeP(in_tensor, res=[0.25, 0.25, 0.25],
                                             dtype={tf.complex64: tf.float32, tf.complex128: tf.float64}[dtype])
            tf_logging.info(f"out shape {out_tensor.shape}, type {out_tensor.dtype}")
            # tf_logging.info(f"L2 difference against numpy.fft.fft2 with {dtype}: "
            #                 f"{np.linalg.norm(out_tensor.numpy() - out_arr):.6g}")
            # if dtype == tf.complex64:
            #     atol = 1e-8
            # else:
            #     atol = 1e-16
            # custom_atol = max(atol * np.prod(self.XY_SHAPE), atol * 10)
            # self.assertAllClose(out_tensor.numpy(), out_arr, atol=custom_atol, rtol=0)

    @test_util.run_cuda_only
    def test_complex64(self):
        self.makeP_test_body(shape=self.XY_SHAPE, dtype=tf.complex64)

    @test_util.run_cuda_only
    def test_complex128(self):
        self.makeP_test_body(shape=self.XY_SHAPE, dtype=tf.complex128)


if __name__ == '__main__':
    test.main()
