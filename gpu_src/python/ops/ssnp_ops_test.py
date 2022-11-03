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


class FFTTest(test.TestCase):
    """
    Test for tf.scatt_lib._ifft2d / _ssnp_ops_lib.fft_test
    """
    FORWARD = True
    XY_SHAPE = (1024, 1024)
    BATCH_SHAPE = (8, 1, 3)

    def fft_test_body(self, is_forward, shape, dtype):
        rng = np.random.default_rng(1227)
        in_arr = rng.random(shape, np.float64) + rng.random(shape, np.float64) * 1j - (0.5 + 0.5j)
        out_arr = np.fft.fft2(in_arr) if is_forward else np.fft.ifft2(in_arr) * np.prod(self.XY_SHAPE)
        with self.cached_session(force_gpu=True):
            in_tensor = tf.constant(in_arr, dtype=dtype)
            if is_forward:
                out_tensor = tf.scatt_lib._fft2d(in_tensor)
            else:
                out_tensor = tf.scatt_lib._ifft2d(in_tensor)
            tf_logging.info(f"L2 difference against numpy.fft.fft2 with {dtype}: "
                            f"{np.linalg.norm(out_tensor.numpy() - out_arr):.6g}")
            if dtype == tf.complex64:
                atol = 1e-8
            else:
                atol = 1e-16
            custom_atol = max(atol * np.prod(self.XY_SHAPE), atol * 10)
            self.assertAllClose(out_tensor.numpy(), out_arr, atol=custom_atol, rtol=0)

    @test_util.run_cuda_only
    def test_complex64(self):
        self.fft_test_body(is_forward=self.FORWARD, shape=self.XY_SHAPE, dtype=tf.complex64)

    @test_util.run_cuda_only
    def test_complex128(self):
        self.fft_test_body(is_forward=self.FORWARD, shape=self.XY_SHAPE, dtype=tf.complex128)

    @test_util.run_cuda_only
    def test_fft_batch(self):
        self.fft_test_body(is_forward=self.FORWARD, shape=(*self.BATCH_SHAPE, *self.XY_SHAPE), dtype=tf.complex128)


class IFFTTest(FFTTest):
    FORWARD = False  # inver


if __name__ == '__main__':
    test.main()
