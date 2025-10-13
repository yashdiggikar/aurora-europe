import torch

def rmse(a: torch.Tensor, b: torch.Tensor):
    return torch.sqrt(torch.mean((a - b) ** 2))

def acc(a: torch.Tensor, b: torch.Tensor):
    while a.dim() > 2:
        a = a.flatten(start_dim=1)
        b = b.flatten(start_dim=1)
    a_ = a - a.mean(dim=1, keepdim=True)
    b_ = b - b.mean(dim=1, keepdim=True)
    num = (a_ * b_).sum(dim=1)
    den = torch.sqrt((a_**2).sum(dim=1) * (b_**2).sum(dim=1)) + 1e-8
    return (num / den).mean()
