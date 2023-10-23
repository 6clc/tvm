import pytest
import torch
import torch.nn.functional as F
from torch import fx
from torch.nn import Module
import torch.nn as nn

import tvm
from tvm import relax
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch import from_fx
from tvm import dlight as dl


def verify_model(torch_model, input_info, binding, expected):
    import torch
    from torch import fx
    from tvm.relax.frontend.torch import from_fx

    graph_model = fx.symbolic_trace(torch_model)
    mod = from_fx(graph_model, input_info)
    expected = mod
    print(expected)

    # target = tvm.target.Target("cuda", host="llvm")
    target = tvm.target.Target("cuda")
    # with target, tvm.transform.PassContext(opt_level=3):
    with target:
        expected = tvm.relax.transform.LegalizeOps()(expected) # 有init block
        # expected = tvm.relax.transform.AnnotateTIROpPattern()(expected)
        # expected = tvm.relax.transform.FuseOps()(expected)
        # print(expected)
        # expected = tvm.relax.transform.FuseTIR()(expected)
        # print(expected)
        # expected = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(expected) # 只针对reduce才有用
        expected = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(expected)
        expected = tvm.tir.transform.DefaultGPUSchedule()(expected)
        # expected = tvm.tir.transform.LowerCrossThreadReduction()(expected)
    ex = relax.build(expected, target)
def test_linear():
    # nn.Linear
    class Dense1(Module):
        def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0):
            super().__init__()
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        def forward(self, x):
            x = self.conv2d(x)
            return x


    in_channels = 3
    out_channels = 128
    input_info = [([1, 3, 8, 8], "float32")]

    model = Dense1(in_channels, out_channels )
    verify_model(model, input_info, None, None)

import os
import sys
print(sys.version)
print(os.getpid())
# os.system("read REPLY")

test_linear()

