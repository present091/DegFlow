import torch

def normalize(z, mean, std):
    """
    Normalize the tensor z.
    Assumes that z has shape [B, C, *] and that mean and std have shape [C].
    
    Args:
        z (Tensor): Input tensor with shape [B, C, *].
        mean (Tensor or None): Tensor of shape [C] (or broadcastable to [1, C, 1, ...]).
        std (Tensor or None): Tensor of shape [C] (or broadcastable to [1, C, 1, ...]).
    
    Returns:
        Tensor: Normalized tensor.
    """
    if mean is None or std is None:
        return z
    # Expand dimensions to match the channel dimension (z.ndim - 2): excluding batch and channel dimensions.
    view_shape = [1, -1] + [1] * (z.ndim - 2)
    mean = mean.view(*view_shape).to(z.device)
    std = std.view(*view_shape).to(z.device)
    return (z - mean) / std

def denormalize(z, mean, std):
    """
    Restore the normalized tensor z back to its original scale.
    
    Args:
        z (Tensor): Normalized tensor with shape [B, C, *].
        mean (Tensor or None): Tensor of shape [C] (or broadcastable to [1, C, 1, ...]).
        std (Tensor or None): Tensor of shape [C] (or broadcastable to [1, C, 1, ...]).
    
    Returns:
        Tensor: Denormalized tensor.
    """
    if mean is None or std is None:
        return z
    view_shape = [1, -1] + [1] * (z.ndim - 2)
    mean = mean.view(*view_shape).to(z.device)
    std = std.view(*view_shape).to(z.device)
    return z * std + mean
