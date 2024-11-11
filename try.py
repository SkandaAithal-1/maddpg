import torch

a = torch.Tensor([[1, 2, 3], [2, 3, 4]])
b = torch.Tensor([[1, 2], [1, 1]])

c = torch.concat([torch.Tensor([1,2,3]), torch.Tensor([4,5,6]), torch.Tensor([3,2, 1])], dim=1)
print(c)
