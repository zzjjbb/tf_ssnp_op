# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for time_two ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from . import time_two_ops
except ImportError:
  import time_two_ops


class TimeTwoTest(test.TestCase):

  @test_util.run_gpu_only
  def testTimeTwo(self):
    in_arr = np.random.rand(4, 4) + np.random.rand(4, 4) * 1j
    out_arr = np.fft.fft2(in_arr)
    with self.test_session():
      with ops.device("/gpu:0"):
        self.assertAllClose(
            time_two_ops.time_two(in_arr), out_arr
        )
        print(time_two_ops.time_two(in_arr))
        print(out_arr)


if __name__ == '__main__':
  test.main()
