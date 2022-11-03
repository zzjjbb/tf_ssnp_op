from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
import tensorflow

_ssnp_ops_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_ssnp_ops.so'))

tensorflow.scatt_lib = type('ScattLib', (), {
    # 'ssnp': _ssnp_ops_lib.fft_test,
    '_fft2d': _ssnp_ops_lib.fft_test,
    '_ifft2d': _ssnp_ops_lib.ifft_test
})
