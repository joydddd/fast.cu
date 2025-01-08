from torch.utils.cpp_extension import load
matmul_cuda = load(name='matmul_cuda', sources=['matmul_api.cpp', 'matmul.cu'])

import torch
M, N, K = 8192, 8192, 8192
a = torch.rand(M, N).cuda()
b = torch.rand(N, K).cuda()

c = matmul_cuda.fwd(a, b)
torch.testing.assert_close(c, torch.matmul(a, b))
print(c)
