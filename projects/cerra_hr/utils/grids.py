# aurora-europe/projects/cerra_hr/utils/grids.py
import torch

def latlon_mesh(lat_min, lat_max, lon_min, lon_max, H, W, *, device=None, dtype=torch.float32):
    """
    Build 2D latitude/longitude tensors of shape [H, W].

    Args:
        lat_min, lat_max: float degrees (south/north bounds). If lat_max > lat_min,
                          the output goes from north->south (common in met grids).
        lon_min, lon_max: float degrees (west/east bounds). Assumes lon_min < lon_max
                          in the chosen convention (e.g., -25..45 or 0..360).
        H, W: integers, grid height/width.
        device: optional torch.device (e.g., "cuda").
        dtype:  torch dtype (default float32).

    Returns:
        lat2d, lon2d: torch.Tensor, shape [H, W], on `device` and with `dtype`.
    """
    if not (isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0):
        raise ValueError(f"H and W must be positive integers, got H={H}, W={W}")

    # Latitude: north -> south if lat_max > lat_min (typical)
    lat_line = torch.linspace(lat_max, lat_min, H, device=device, dtype=dtype)
    lon_line = torch.linspace(lon_min, lon_max, W, device=device, dtype=dtype)

    lat2d = lat_line.unsqueeze(1).expand(H, W)  # [H,1] -> [H,W]
    lon2d = lon_line.unsqueeze(0).expand(H, W)  # [1,W] -> [H,W]
    return lat2d, lon2d


def latlon_from_coords(lat_1d, lon_1d, *, device=None, dtype=torch.float32):
    """
    Build lat/lon 2D meshes from 1D coordinate arrays (numpy or torch).

    Args:
        lat_1d: 1D array-like of latitudes (N)
        lon_1d: 1D array-like of longitudes (M)
        device: optional torch.device
        dtype:  torch dtype

    Returns:
        lat2d, lon2d: torch.Tensor, shape [N, M]
    """
    lat_1d = torch.as_tensor(lat_1d, device=device, dtype=dtype)
    lon_1d = torch.as_tensor(lon_1d, device=device, dtype=dtype)
    H, W = lat_1d.numel(), lon_1d.numel()
    lat2d = lat_1d.view(H, 1).expand(H, W)
    lon2d = lon_1d.view(1, W).expand(H, W)
    return lat2d, lon2d
