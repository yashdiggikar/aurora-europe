import torch

def cosine_lat_weights(lat_2d):
    return torch.cos(torch.deg2rad(lat_2d)).clamp_min(0.0)

def weighted_mse(pred, truth, lat_2d):
    if pred.dim() == 3: pred = pred.unsqueeze(1); truth = truth.unsqueeze(1)
    w = cosine_lat_weights(lat_2d).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return ((pred - truth) ** 2 * w).mean()
