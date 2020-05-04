import numpy as np

def read_as_int(input_array):

    if input_array.dtype.type is np.float32:
        dtype = np.int32
    elif input_array.dtype.type is np.float64:
        dtype = np.int64
    else:
        raise TypeError("dtype not supported")
    ints = np.copy(np.frombuffer(np.ascontiguousarray(input_array),
                                 dtype=dtype))

    return ints.reshape(input_array.shape)


def uls_difference(input1, input2):
    ints1 = read_as_int(input1)
    ints2 = read_as_int(input2)
    return np.abs(ints1 - ints2)


def compare_floats(input1, input2, fixed_tolerance, uls_tolerance):
    assert input1.dtype.type is input2.dtype.type
    check1 = uls_difference(input1, input2) < uls_tolerance
    check2 = np.abs(input1 - input2) < (fixed_tolerance
                                        * np.finfo(input1.dtype.type).eps)

    return np.all(np.logical_or(check1, check2))
