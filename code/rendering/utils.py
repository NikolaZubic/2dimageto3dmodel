import torch
import torch.nn.functional as F

from packaging import version

def grid_sample_bilinear(input, grid):
    # PyTorch 1.3 introduced an API change (breaking change in version 1.4), therefore we check this explicitly
    # to make sure that the behavior is consistent across different versions
    if version.parse(torch.__version__) < version.parse('1.3'):
        return F.grid_sample(input, grid, mode='bilinear')
    else:
        return F.grid_sample(input, grid, mode='bilinear', align_corners=True)


def symmetrize_texture(x):
    # Apply even symmetry along the x-axis (from length N to 2N)
    x_flip = torch.flip(x, (len(x.shape) - 1,))
    return torch.cat((x_flip[:, :, :, x_flip.shape[3]//2:], x, x_flip[:, :, :, :x_flip.shape[3]//2]), dim=-1)


def adjust_poles(tex):
    # Average top and bottom rows (corresponding to poles) -- for mesh only
    top = tex[:, :, :1].mean(dim=3, keepdim=True).expand(-1, -1, -1, tex.shape[3])
    middle = tex[:, :, 1:-1]
    bottom = tex[:, :, -1:].mean(dim=3, keepdim=True).expand(-1, -1, -1, tex.shape[3])
    return torch.cat((top, middle, bottom), dim=2)
    

def circpad(x, amount=1):
    # Circular padding along x-axis (before a convolution)
    left = x[:, :, :, :amount]
    right = x[:, :, :, -amount:]
    return torch.cat((right, x, left), dim=3)


def qrot(q, v):
    """
    Quaternion-vector multiplication (rotation of a vector)
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    
    qvec = q[:, 1:].unsqueeze(1).expand(-1, v.shape[1], -1)
    uv = torch.cross(qvec, v, dim=2)
    uuv = torch.cross(qvec, uv, dim=2)
    return v + 2 * (q[:, :1].unsqueeze(1) * uv + uuv)

def qmul(q, r):
    """
    Quaternion-quaternion multiplication
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)