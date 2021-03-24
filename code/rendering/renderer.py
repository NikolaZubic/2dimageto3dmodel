from kaolin.graphics.dib_renderer.rasterizer import linear_rasterizer
from kaolin.graphics.dib_renderer.utils import datanormalize

from .fragment_shader import fragmentshader

import torch
import torch.nn as nn

def ortho_projection(points_bxpx3, faces_fx3):
    xy_bxpx3 = points_bxpx3 # xyz
    xy_bxpx2 = xy_bxpx3[:, :, :2] # xy

    pf0_bxfx3 = points_bxpx3[:, faces_fx3[:, 0], :]
    pf1_bxfx3 = points_bxpx3[:, faces_fx3[:, 1], :]
    pf2_bxfx3 = points_bxpx3[:, faces_fx3[:, 2], :]
    points3d_bxfx9 = torch.cat((pf0_bxfx3, pf1_bxfx3, pf2_bxfx3), dim=2)

    xy_f0 = xy_bxpx2[:, faces_fx3[:, 0], :]
    xy_f1 = xy_bxpx2[:, faces_fx3[:, 1], :]
    xy_f2 = xy_bxpx2[:, faces_fx3[:, 2], :]
    points2d_bxfx6 = torch.cat((xy_f0, xy_f1, xy_f2), dim=2)

    v01_bxfx3 = pf1_bxfx3 - pf0_bxfx3
    v02_bxfx3 = pf2_bxfx3 - pf0_bxfx3

    normal_bxfx3 = torch.cross(v01_bxfx3, v02_bxfx3, dim=2)

    return points3d_bxfx9, points2d_bxfx6, normal_bxfx3

class Renderer(nn.Module):

    def __init__(self, height, width, filtering='bilinear'):
        super().__init__()

        self.height = height
        self.width = width
        self.filtering = filtering

    def forward(self, points, uv_bxpx2, texture_bx3xthxtw, ft_fx3=None, background_image=None, return_hardmask=False):

        points_bxpx3, faces_fx3 = points
        
        if ft_fx3 is None:
            ft_fx3 = faces_fx3
            
        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = ortho_projection(points_bxpx3, faces_fx3)

        # Detect front/back faces
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]

        # Ensure that normals are unit length
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = torch.ones_like(c0[:, :, :1])
        uv_bxfx9 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)

        imfeat, improb_bxhxwx1 = linear_rasterizer(
            self.height,
            self.width,
            points3d_bxfx9,
            points2d_bxfx6,
            normalz_bxfx1,
            uv_bxfx9,
        )

        imtexcoords = imfeat[:, :, :, :2]
        hardmask = imfeat[:, :, :, 2:3]

        imrender = fragmentshader(imtexcoords, texture_bx3xthxtw, hardmask,
                                  filtering=self.filtering, background_image=background_image)

        if return_hardmask:
            improb_bxhxwx1 = hardmask
        return imrender, improb_bxhxwx1, normal1_bxfx3
