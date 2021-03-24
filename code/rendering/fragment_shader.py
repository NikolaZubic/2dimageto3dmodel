import torch
import torch.nn

from .utils import grid_sample_bilinear

def texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw, filtering='bilinear'):
    
    imtexcoord_bxhxwx2 = imtexcoord_bxhxwx2 * 2 - 1  # [0, 1] to [-1, 1]
    imtexcoord_bxhxwx2 = imtexcoord_bxhxwx2 * torch.FloatTensor([1, -1]).to(imtexcoord_bxhxwx2.device) # Flip y
    
    if filtering == 'bilinear':
        # Ensures consistent behavior across different PyTorch versions
        texcolor = grid_sample_bilinear(texture_bx3xthxtw, imtexcoord_bxhxwx2)
    else:
        texcolor = torch.nn.functional.grid_sample(texture_bx3xthxtw,
                                                   imtexcoord_bxhxwx2,
                                                   mode=filtering)
    
    texcolor = texcolor.permute(0, 2, 3, 1)
    return texcolor

def fragmentshader(imtexcoord_bxhxwx2,
                   texture_bx3xthxtw,
                   improb_bxhxwx1,
                   filtering='bilinear',
                   background_image=None):

    texcolor_bxhxwx3 = texinterpolation(imtexcoord_bxhxwx2,
                                        texture_bx3xthxtw,
                                        filtering=filtering)

    if background_image is None:
        color = texcolor_bxhxwx3 * improb_bxhxwx1
    else:
        color = torch.lerp(background_image, texcolor_bxhxwx3, improb_bxhxwx1)

    return color
