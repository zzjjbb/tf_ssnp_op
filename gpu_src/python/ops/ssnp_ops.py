from tensorflow.python.framework import load_library, ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import resource_loader
import tensorflow

_ssnp_ops_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_ssnp_ops.so'))

scatt_lib = type('ScattLib', (), {
    'ssnp': _ssnp_ops_lib.scatt_lib_ssnp,
    # 'ssnp_grad': _ssnp_ops_lib.ssnp,
    '_fft2d': _ssnp_ops_lib.scatt_lib_test_fft,
    '_ifft2d': _ssnp_ops_lib.scatt_lib_test_ifft,
    '_makeP': _ssnp_ops_lib.scatt_lib_test_make_p
})

tensorflow.scatt_lib = scatt_lib

# @ops.RegisterGradient("SSNP")
# def _ssnp_grad(op, grad):
#     u1_tensor = op.inputs[0]
#     shape = array_ops.shape(u1_tensor)
#     index = array_ops.zeros_like(shape)
#     u1_grad_in, u2_grad_in = grad
#     u1_grad_out, u2_grad_out = tensorflow.scatt_lib.ssnp_grad(u1_grad_in, u2_grad_in)
#     return [u1_grad_out, u2_grad_out]  # List of one Tensor, since we have one input
