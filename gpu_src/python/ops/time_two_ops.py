from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

time_two_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_time_two_ops.so'))
time_two = time_two_ops.ssnp_test
