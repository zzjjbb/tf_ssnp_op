"""
Created by Jiabei, last modified 10/27/2022

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

rng = np.random.default_rng(1227)


class SSNPTest(test.TestCase):
    FORWARD = True
    XY_SHAPE = (512, 512)
    BATCH = 8
    dz = 1

    def ssnp_test_body(self, shape, dtype):
        in_arr = rng.random(shape, np.float64) + rng.random(shape, np.float64) * 1j - (0.5 + 0.5j)
        with self.cached_session(force_gpu=True):
            in1_tensor = tf.constant(in_arr, dtype=dtype)
            in2_tensor = tf.constant(in_arr, dtype=dtype)
            out1_tensor, out2_tensor = tf.scatt_lib.ssnp(in1_tensor, in2_tensor, res=(1, 2, 3))
            tf_logging.info(f"shape: {out1_tensor.shape}, {out2_tensor.shape}")

    @test_util.run_cuda_only
    def test_free_c64(self):
        self.ssnp_test_body(shape=self.XY_SHAPE, dtype=tf.complex64)


if __name__ == '__main__':
    test.main()
