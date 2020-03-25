import torch
import torch.nn as nn
import numpy as np
import time

batch_size = 2048
m = 27
k = 128 

input1_data = torch.randn(batch_size, m, k, requires_grad=True)
input1_data_trans = torch.transpose(input1_data, 1, 2)
# input_data2 = torch.randn(2048, m, m)

num_iter = 100000
t_begin = time.time()
for i in range(num_iter):
    # output = torch.bmm(input1_data, input1_data_trans)
    # output.backward(input_data2)
    torch.bmm(input1_data, input1_data_trans)
t_end = time.time()
print("time (matmul with transpose): " + str((t_end - t_begin) * 1000 / num_iter))
