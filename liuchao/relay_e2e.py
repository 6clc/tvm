import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay

import torch
import torch.nn as nn

# 定义网络类
class AddAndReduceSum(nn.Module):
    def __init__(self):
        super(AddAndReduceSum, self).__init__()

    def forward(self, x1, x2):
        # 执行加法操作
        add_result = x1 + x2

        # 对结果进行求和
        sum_result = torch.sum(add_result)

        return sum_result

# 创建网络实例
net = AddAndReduceSum()

if __name__=='__main__':
  #prepare model and input
  shape_list = [("input0",(1,3,224,224)), ("input1", (1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(net,(fake_input, fake_input))
  #main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  #optimize the mod
  target = tvm.target.Target("cuda", host="llvm")
    
  with tvm.transform.PassContext(opt_level=3):
    graph_json, mod, params = relay.build(mod, target=target, params=params)
