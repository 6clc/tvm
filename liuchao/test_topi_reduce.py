import os
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi, tir


def test_reduce_map():
    target = "cuda"
    reduce_type = "sum"
    in_shape = [16, 4, 3]
    axis = [0, 2]
    dtype = "float32"
    keepdims = False
    target = tvm.target.Target(target)
    if target.kind.name == "vulkan" and reduce_type in ["sum", "prod", "any", "all"]:
        pytest.xfail(f"Vulkan backend has known errors on {reduce_type}")


    # Build the logic and compile the function
    A = te.placeholder(shape=in_shape, name="A", dtype=dtype)
    A1 = topi.sqrt(topi.exp(A))
    out_dtype = dtype
    if reduce_type == "sum":
        if dtype == "bool":
            B = topi.sum(A, axis=axis, keepdims=keepdims)
        else:
            B = topi.sum(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "prod":
        B = topi.prod(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "all":
        B = topi.all(A, axis=axis, keepdims=keepdims)
    elif reduce_type == "any":
        B = topi.any(A, axis=axis, keepdims=keepdims)
    elif reduce_type == "max":
        B = topi.max(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "min":
        B = topi.min(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "argmax":
        B = topi.argmax(A1, axis=axis, keepdims=keepdims)
        out_dtype = "int32"
    elif reduce_type == "argmin":
        B = topi.argmin(A1, axis=axis, keepdims=keepdims)
        out_dtype = "int32"
    else:
        raise NotImplementedError

    print("6clc topi", B)

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_reduce_schedule(target)(B)

    foo = tvm.build(s, [A, B], target, name=reduce_type)

import os
import sys
print(sys.version)
print(os.getpid())
# os.system("read REPLY")
test_reduce_map()
