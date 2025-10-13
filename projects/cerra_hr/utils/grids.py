import torch

def latlon_mesh(lat_min, lat_max, lon_min, lon_max, H, W):
    lat = torch.linspace(lat_max, lat_min, H).unsqueeze(1).repeat(1, W)  # [H,W]
    lon = torch.linspace(lon_min, lon_max, W).unsqueeze(0).repeat(H, 1)  # [H,W]
    return lat.float(), lon.float()
