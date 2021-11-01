import torch
import torch.nn as nn
import torch.nn.functional as F

from rendering.utils import circpad, symmetrize_texture, adjust_poles

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, pad_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_in, 3, padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(ch_in, ch_out, 3, padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.pad_fn = pad_fn
        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        else:
            self.shortcut = lambda x: x
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(self.pad_fn(x, 1))))
        x = self.relu(self.bn2(self.conv2(self.pad_fn(x, 1))))
        return x + shortcut
    

class ReconstructionNetwork(nn.Module):
    def __init__(self, symmetric=True, texture_res=64, mesh_res=32, interpolation_mode='nearest'):
        super().__init__()
        
        self.symmetric = symmetric
        
        if symmetric:
            self.pad = lambda x, amount: F.pad(x, (amount, amount, 0, 0), mode='replicate')
        else:
            self.pad = lambda x, amount: circpad(x, amount)
        
        self.relu = nn.ReLU(inplace=True)
        
        if interpolation_mode == 'nearest':
            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        elif interpolation_mode == 'bilinear':
            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            raise
            
        assert mesh_res >= 32
        assert texture_res >= 64
        
        self.conv1e = nn.Conv2d(4, 64, 5, stride=2, padding=2, bias=False) # 128 -> 64
        self.bn1e = nn.BatchNorm2d(64)
        self.conv2e = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False) # 64 > 32
        self.bn2e = nn.BatchNorm2d(128)
        self.conv3e = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False) # 32 -> 16
        self.bn3e = nn.BatchNorm2d(256)
        self.conv4e = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False) # 16 -> 8
        self.bn4e = nn.BatchNorm2d(512)
        
        bottleneck_dim = 256
        self.conv5e = nn.Conv2d(512, 64, 3, stride=2, padding=1, bias=False) # 8 -> 4
        self.bn5e = nn.BatchNorm2d(64)
        self.fc1e = nn.Linear(64*8*8, bottleneck_dim, bias=False)
        self.bnfc1e = nn.BatchNorm1d(bottleneck_dim)
            
        self.fc3e = nn.Linear(bottleneck_dim, 1024, bias=False)
        self.bnfc3e = nn.BatchNorm1d(1024)
        
        # Texture generation
        self.base_res_h = 4
        self.base_res_w = 2 if symmetric else 4
            
        self.fc1_tex = nn.Linear(1024, self.base_res_h*self.base_res_w*256)
        self.blk1 = ResBlock(256, 512, self.pad) # 4 -> 8
        self.blk2 = ResBlock(512, 256, self.pad) # 8 -> 16
        self.blk3 = ResBlock(256, 256, self.pad) # 16 -> 32 (k=1)
        
        assert texture_res in [64, 128, 256]
        self.texture_res = texture_res
        if texture_res >= 128:
            self.blk3b_tex = ResBlock(256, 256, self.pad) # k = 2
        if texture_res >= 256:
            self.blk3c_tex = ResBlock(256, 256, self.pad) # k = 4
        
        self.blk4_tex = ResBlock(256, 128, self.pad) # k*32 -> k*64
        self.blk5_tex = ResBlock(128, 64, self.pad) # k*64 -> k*64 (no upsampling)
        
        self.conv_tex = nn.Conv2d(64, 3, 5, padding=(2, 0))
        
        # Mesh generation
        self.blk4_mesh = ResBlock(256, 64, self.pad) # 32 -> 32 (no upsampling)
        self.conv_mesh = nn.Conv2d(64, 3, 5, padding=(2, 0))
        
        # Zero-initialize mesh output layer for stability (avoids self-intersections)
        self.conv_mesh.bias.data[:] = 0
        self.conv_mesh.weight.data[:] = 0
            
        total_params = 0
        for param in self.parameters():
            total_params += param.nelement()
        print('Model parameters: {:.2f}M'.format(total_params/1000000))
        
    def forward(self, x):
        # Generate latent code
        x = self.relu(self.bn1e(self.conv1e(x)))
        x = self.relu(self.bn2e(self.conv2e(x)))
        x = self.relu(self.bn3e(self.conv3e(x)))
        x = self.relu(self.bn4e(self.conv4e(x)))
        x = self.relu(self.bn5e(self.conv5e(x)))
        
        x = x.view(x.shape[0], -1) # Flatten
        z = self.relu(self.bnfc1e(self.fc1e(x)))
        z = self.relu(self.bnfc3e(self.fc3e(z)))
        
        bb = self.fc1_tex(z).view(z.shape[0], -1, self.base_res_h, self.base_res_w)
        bb = self.up(self.blk1(bb))
        bb = self.up(self.blk2(bb))
        bb = self.up(self.blk3(bb))
        bb_mesh = bb
        if self.texture_res >= 128:
            bb = self.up(self.blk3b_tex(bb))
        if self.texture_res >= 256:
            bb = self.up(self.blk3c_tex(bb))
        
        mesh_map = self.blk4_mesh(bb_mesh)
        mesh_map = self.conv_mesh(self.pad(self.relu(mesh_map), 2))
        mesh_map = adjust_poles(mesh_map)

        tex = self.up(self.blk4_tex(bb))
        tex = self.blk5_tex(tex)
        tex = self.conv_tex(self.pad(self.relu(tex), 2)).tanh_()
        
        if self.symmetric:
            tex = symmetrize_texture(tex)
            mesh_map = symmetrize_texture(mesh_map)      

        return tex, mesh_map
    
    
class DatasetParams(nn.Module):
    def __init__(self, args, dataset_size):
        super().__init__()
        # Dataset offsets
        self.dataset_size = dataset_size
        if args.optimize_deltas:
            self.ds_translation = nn.Parameter(torch.zeros(dataset_size, 2))
            self.ds_scale = nn.Parameter(torch.zeros(dataset_size, 1))
        if args.optimize_z0:
            self.ds_z0 = nn.Parameter(torch.ones(dataset_size, 1))
            
    def forward(self, indices, mode):
        assert mode in ['deltas', 'z0']
        if indices is not None:
            # Indices between N and 2N indicate that the image is mirrored (data augmentation)
            # Therefore, we flip the sign of the x translation
            x_sign = (1 - 2*(indices // self.dataset_size).float()).unsqueeze(-1)
            indices = indices % self.dataset_size
        else:
            x_sign = 1
        
        if mode == 'deltas':
            if indices is not None:
                translation_delta = self.ds_translation[indices]
            else:
                translation_delta = self.ds_translation.mean(dim=0, keepdim=True)

            translation_delta = torch.cat((translation_delta[:, :1] * x_sign,
                                           translation_delta[:, 1:2],
                                           torch.zeros_like(translation_delta[:, :1])), dim=1)
            if indices is not None:
                scale_delta = self.ds_scale[indices]
            else:
                scale_delta = self.ds_scale.mean(dim=0, keepdim=True)
            return translation_delta, scale_delta
        else: # z0
            if indices is not None:
                z0 = self.ds_z0[indices]
            else:
                z0 = self.ds_z0.mean(dim=0, keepdim=True)
            return 1 + torch.exp(z0)