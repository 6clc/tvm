import pytest
import torch
import torch.nn.functional as F
from torch import fx
from torch.nn import Module

import tvm
from tvm import relax
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch import from_fx


def verify_model(torch_model, input_info, binding, expected):
    import torch
    from torch import fx
    from tvm.relax.frontend.torch import from_fx

    graph_model = fx.symbolic_trace(torch_model)
    mod = from_fx(graph_model, input_info)
    expected = mod
    print(expected)

    target = tvm.target.Target("cuda", host="llvm")
    with target, tvm.transform.PassContext(opt_level=3):
        expected = tvm.relax.transform.LegalizeOps()(expected) # 有init block
        expected = tvm.relax.transform.AnnotateTIROpPattern()(expected)
        expected = tvm.relax.transform.FuseOps()(expected)
        expected = tvm.relax.transform.FuseTIR()(expected)
        expected = tvm.tir.transform.LowerCrossThreadReduction()(expected)
        expected = tvm.tir.transform.DefaultGPUSchedule()(expected)
    print(expected)
    ex = relax.build(expected, target)
def test_linear():
    # nn.Linear
    class Dense1(Module):
        def __init__(self):
            super().__init__()
            # self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            # 执行加法操作
            add_result =  input * 2.

            # 对结果进行求和
            sum_result = torch.sum(add_result, 3)
            # sum_result = add_result + input
            return sum_result


    input_info = [([2, 3, 10, 10], "float32")]

    model = Dense1()
    # binding = {"w1": model.linear.weight.detach().numpy(), "w2": model.linear.bias.detach().numpy()}
    verify_model(model, input_info, None, None)


 
test_linear()

