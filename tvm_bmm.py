import numpy as np
from tvm import relay
import tvm
import time

from tvm.contrib import graph_runtime

batch_size=2048
k=128
m=27
dtype="float32"

input1_data = np.random.rand(batch_size, m, k).astype(dtype)
input2_data = np.random.rand(batch_size, m, k).astype(dtype)

input_shape = (batch_size, m, k)
input1 = relay.var("input1", shape=input_shape, dtype=dtype)
input2 = relay.var("input2", shape=input_shape, dtype=dtype)
#input2 = relay.copy(input1)
Z = relay.nn.batch_matmul(input1, input2)
func = relay.Function([input1, input2], Z)

# params = {}
# target = 'llvm'
target = 'llvm -mcpu=cascadelake'
# target = 'llvm -libs=mkl'
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(func, target) #, params=params)

ctx = tvm.context(target, 0)
model = graph_runtime.create(graph, lib, ctx)

model.set_input('input1', tvm.nd.array(input1_data.astype(dtype)))
model.set_input('input2', tvm.nd.array(input2_data.astype(dtype)))
# m.set_input(**params)

num_iter = 10000
ftimer = model.module.time_evaluator("run", ctx, number=num_iter, repeat=3)
prof_res = np.array(ftimer().results) * 1000
print("Mean inference time (std dev): %.5f ms (%.5f ms)" %
        (np.mean(prof_res), np.std(prof_res)))


