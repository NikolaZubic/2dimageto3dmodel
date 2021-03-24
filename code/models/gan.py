import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rendering.utils import circpad, symmetrize_texture, adjust_poles


def positional_encoding(Ny, Nx):
    # Sine-cosine positional embedding which smoothly wraps around the x axis
    symmetric = (Nx == Ny // 2)
    Nx = Ny
    ty = np.linspace(0, np.pi, Ny, endpoint=False)
    tx = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    Y, X = np.meshgrid(tx, ty)
    result = np.stack((np.cos(X), np.sin(X), np.cos(Y), np.sin(Y)))
    if symmetric:
        return result[:, :, result.shape[2]//4:-result.shape[2]//4]
    else:
        return result
    
    
class MeshDiscriminator(nn.Module):
    def __init__(self, args, nc, circular=True, positional_embeddings=True):
        super().__init__()
        
        if args.norm_d == 'instance':
            norm_layer = lambda nc: nn.InstanceNorm2d(nc, affine=True)
            bias = False
        elif args.norm_d == 'none':
            norm_layer = lambda nc: lambda x: x
            bias = True
        else:
            raise
        
        self.args = args
        
        if args.conditional_text:
            self.att = SpatialAttention(256, args.text_embedding_dim)
        
        self.circular = circular
        self.positional_embeddings = positional_embeddings
        kw = 5 # Kernel size
        pw = kw // 2 # Padding
        if circular:
            self.pad = lambda x: circpad(x, pw) # For 5x5 convolutions
            self.pad2 = lambda x: circpad(x, 1) # For 4x4 convolutions
            pad_amount = (pw, 0) # Zero-pad along y axis
        else:
            self.pad = lambda x: x
            pad_amount = pw
            
        if positional_embeddings:
            self.pos_emb = None
            nc += 4 # cos(x), sin(x), cos(y), sin(y)
            
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nc, 64, kw, padding=pad_amount, stride=1))
        
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, padding=(1, 0), stride=2, bias=bias))
        self.bn2 = norm_layer(128)
        
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, padding=(1, 0), stride=2, bias=bias))
        self.bn3 = norm_layer(256)
        
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 1, kw, padding=pad_amount, stride=1))
    
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        if args.conditional_class:
            self.projector = nn.Embedding(args.n_classes[0], 256)
            if args.conditional_color:
                self.projector_col1 = nn.Embedding(args.n_classes[1], 256)
        
        total_params = 0
        for param in self.parameters():
            total_params += param.nelement()
        print('MeshDiscriminator parameters: {:.2f}M'.format(total_params/1000000))
        
    def forward(self, texture, mesh_map, c=None, caption=None):
        # Downsample texture to the same resolution as the mesh (typically 32x32)
        x = F.avg_pool2d(texture, texture.shape[2]//mesh_map.shape[2])
        
        cat_list = []
        cat_list.append(mesh_map)
            
        if self.positional_embeddings:
            if self.pos_emb is None:
                self.pos_emb = torch.FloatTensor(positional_encoding(x.shape[2], x.shape[3])).unsqueeze(0)
            emb = self.pos_emb.to(x.device).expand(x.shape[0], -1, -1, -1)
            cat_list.append(emb)
        
        if len(cat_list) > 0:
            x = torch.cat((x, *cat_list), dim=1)
        
        if self.args.mask_output:
            with torch.no_grad():
                downsampled_mask = F.avg_pool2d(x[:, 3:4], 4)
            
        x = self.relu(self.conv1(self.pad(x))) # /1 (32 -> 32)
        x = self.relu(self.bn2(self.conv2(self.pad2(x)))) # /2 (32 -> 16)
        x = self.relu(self.bn3(self.conv3(self.pad2(x)))) # /4 (16 -> 8)
        y = self.conv4(self.pad(x)) # /4 (8 -> 8)
        
        if self.args.conditional_class:
            # Projection discriminator
            if self.args.conditional_color:
                # Class + color
                c_emb = self.projector(c[:, 0]) + self.projector_col1(c[:, 1])
            else:
                # Only class
                c_emb = self.projector(c[:, 0])
            y += torch.sum(x * c_emb.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
            
        elif self.args.conditional_text:
            attention_output, _ = self.att(x, *caption)
            y += torch.sum(x * attention_output, dim=1, keepdim=True)
        
        if self.args.mask_output:
            return y, downsampled_mask
        else:
            return y, None
    
class TextureDiscriminator(nn.Module):
    def __init__(self, args, nc, downsample=1, circular=True, positional_embeddings=True):
        super().__init__()
        
        if args.norm_d == 'instance':
            norm_layer = lambda nc: nn.InstanceNorm2d(nc, affine=True)
            bias = False
        elif args.norm_d == 'none':
            norm_layer = lambda nc: lambda x: x
            bias = True
        else:
            raise
        
        self.args = args
        
        if args.conditional_text:
            self.att = SpatialAttention(512, args.text_embedding_dim)
        
        self.circular = circular
        self.positional_embeddings = positional_embeddings
        kw = 5 # Kernel size
        pw = kw // 2 # Padding
        if circular:
            self.pad = lambda x: circpad(x, pw) # Circular convolutions along x axis
            self.pad2 = lambda x: circpad(x, 1)
            pad_amount = (pw, 0) # Zero-pad along y axis
        else:
            self.pad = lambda x: x
            pad_amount = pw
            
        if positional_embeddings:
            self.pos_emb = None
            nc += 4
            
        # If texture is 512x512, use stride 2 in the first layer
        self.stride_first = (downsample == 1 and args.texture_resolution >= 512) \
                            or args.texture_resolution >= 1024 \
                            or args.conditional_text
        if self.stride_first:
            self.padconv1 = self.pad2
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nc, 64, 4, padding=(1, 0), stride=2))
        else:
            self.padconv1 = self.pad
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nc, 64, kw, padding=pad_amount, stride=1))
        
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, padding=(1, 0), stride=2, bias=bias))
        self.bn2 = norm_layer(128)
        
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, padding=(1, 0), stride=2, bias=bias))
        self.bn3 = norm_layer(256)
        
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, padding=(1, 0), stride=2, bias=bias))
        self.bn4 = norm_layer(512)
        
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(512, 1, kw, padding=pad_amount, stride=1))
    
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        
        if args.conditional_class:
            self.projector = nn.Embedding(args.n_classes[0], 512)
            if args.conditional_color:
                self.projector_col1 = nn.Embedding(args.n_classes[1], 512)
        
        total_params = 0
        for param in self.parameters():
            total_params += param.nelement()
        print('TextureDiscriminator parameters: {:.2f}M'.format(total_params/1000000))
        
    def forward(self, x, c=None, caption=None):
        if self.downsample > 1:
            x = F.avg_pool2d(x, self.downsample)
            
        if self.args.mask_output:
            with torch.no_grad():
                ds_factor = 16 if self.stride_first else 8
                downsampled_mask = F.avg_pool2d(x[:, 3:4], ds_factor)
        
        cat_list = []
            
        if self.positional_embeddings:
            if self.pos_emb is None:
                self.pos_emb = torch.FloatTensor(positional_encoding(x.shape[2], x.shape[3])).unsqueeze(0)
            emb = self.pos_emb.to(x.device).expand(x.shape[0], -1, -1, -1)
            cat_list.append(emb)
        
        if len(cat_list) > 0:
            x = torch.cat((x, *cat_list), dim=1)
            
        x = self.relu(self.conv1(self.padconv1(x))) # /2 (256 -> 128)
        x = self.relu(self.bn2(self.conv2(self.pad2(x)))) # /4 (128 -> 64)
        x = self.relu(self.bn3(self.conv3(self.pad2(x)))) # /8 (64 -> 32)
        x = self.relu(self.bn4(self.conv4(self.pad2(x)))) # /16 (32 -> 16)
        y = self.conv5(self.pad(x)) # /16 (16 -> 16)
        
        if self.args.conditional_class:
            # Projection discriminator
            if self.args.conditional_color:
                c_emb = self.projector(c[:, 0]) + self.projector_col1(c[:, 1])
            else:
                c_emb = self.projector(c[:, 0])
            y += torch.sum(x * c_emb.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
            
        elif self.args.conditional_text:
            attention_output, _ = self.att(x, *caption)
            y += torch.sum(x * attention_output, dim=1, keepdim=True)
        
        if self.args.mask_output:
            return y, downsampled_mask
        else:
            return y, None
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, args, nc):
        super().__init__()
        
        self.args = args
        print(f'Initializing multi-scale discriminator ({args.num_discriminators} scales)...')
        self.d1 = TextureDiscriminator(args, nc, 1)
        if not args.texture_only:
            self.d2 = MeshDiscriminator(args, nc + 3)
        else:
            downsample = 2 # Downsample by a factor of 2
            self.d2 = TextureDiscriminator(args, nc, downsample)
        
        if args.num_discriminators == 3:
            self.d3 = TextureDiscriminator(args, nc, 4)
        elif args.num_discriminators != 2:
            raise
        
    def forward(self, x, mesh_map=None, c=None, caption=None):
        d1, m1 = self.d1(x, c, caption)
        d2, m2 = self.d2(x, mesh_map, c, caption)
        if self.args.num_discriminators == 3:
            d3, m3 = self.d3(x, c, caption)
            return [d1, d2, d3], [m1, m2, m3]
        else:
            return [d1, d2], [m1, m2]



class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, args, ch, emb_dim):
        super().__init__()
        if args.norm_g == 'syncbatch':
            from sync_batchnorm import SynchronizedBatchNorm2d
            self.norm = SynchronizedBatchNorm2d(ch, affine=False)
        elif args.norm_g == 'batch':
            self.norm = nn.BatchNorm2d(ch, affine=False)
        elif args.norm_g == 'instance':
            self.norm = nn.InstanceNorm2d(ch, affine=False)
        elif args.norm_g == 'none':
            self.norm = lambda x: x # Identity
        else:
            raise
            
        self.fc_gamma = nn.Linear(emb_dim, ch)
        self.fc_beta = nn.Linear(emb_dim, ch)
        
    def forward(self, x, z):
        x = self.norm(x)
        gamma = self.fc_gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(z).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
    
class ResBlockUp(nn.Module):
    def __init__(self, args, ch_in, ch_out, emb_dim, pad_fn):
        super().__init__()
        ch_middle = min(ch_in, ch_out)
        self.ch_out = ch_out
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_middle, 3, padding=(1, 0), bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(ch_middle, ch_out, 3, padding=(1, 0), bias=False))
        self.norm1 = ConditionalBatchNorm2d(args, ch_middle, emb_dim)
        self.norm2 = ConditionalBatchNorm2d(args, ch_out, emb_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.pad = pad_fn
        if ch_in != ch_out:
            self.shortcut = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 1, bias=False))
        else:
            self.shortcut = lambda x: x
        
    def forward(self, x, z):
        shortcut = self.shortcut(x)
        
        x = self.relu(self.norm1(self.conv1(self.pad(x, 1)), z))
        x = self.relu(self.norm2(self.conv2(self.pad(x, 1)), z))
        
        return x + shortcut

class Generator(nn.Module):
    def __init__(self, args, emb_dim, symmetric=True, mesh_head=True):
        super().__init__()
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        
        self.width = 8
        self.height = 8
        
        self.args = args
        self.symmetric = symmetric
        if symmetric:
            self.width //= 2
            # For 3x3 convolutions, even-mirror padding can be emulated with replication
            self.pad = lambda x, amount: F.pad(x, (amount, amount, 0, 0), mode='replicate')
        else:
            self.pad = lambda x, amount: circpad(x, amount) # Circular convolutions
        
        if args.conditional_class and args.conditional_color:
            self.emb_class = nn.Embedding(args.n_classes[0], emb_dim//2)
            self.emb_color = nn.Embedding(args.n_classes[1], emb_dim//2)
            emb_dim += emb_dim
        elif args.conditional_class:
            self.emb_class = nn.Embedding(args.n_classes[0], emb_dim)
            emb_dim += emb_dim
        
        self.fc = nn.Linear(emb_dim, self.height*self.width*512)
        
        self.blk1 = ResBlockUp(args, 512, 512, emb_dim, self.pad) # 8 -> 16
        self.blk2 = ResBlockUp(args, 512, 256, emb_dim, self.pad) # 16 -> 32
        
        if args.texture_resolution >= 256:
            self.blk3a = ResBlockUp(args, 256, 256, emb_dim, self.pad) # 32 -> 64
        if args.texture_resolution >= 512:
            self.blk3b = ResBlockUp(args, 256, 256, emb_dim, self.pad) # 64 -> 128
        if args.texture_resolution >= 1024:
            self.blk3c = ResBlockUp(args, 256, 256, emb_dim, self.pad) # 64 -> 128
            
        if args.conditional_text:
            self.att = SpatialAttention(256, args.text_embedding_dim)
        
        self.blk4 = ResBlockUp(args, 256, 128, emb_dim, self.pad) # 32|64|128 -> 64|128|256
        self.blk5 = ResBlockUp(args, 128, 128, emb_dim, self.pad) # 64|128|256 -> 128|256|512
        self.blk6 = ResBlockUp(args, 128, 64, emb_dim, self.pad) # 128|256|512 (no upscale)
        self.conv_final = nn.Conv2d(64, 3, 5, padding=(2, 0))
        
        self.mesh_head = mesh_head
        if mesh_head:
            self.blk3_mesh = ResBlockUp(args, 256, 64, emb_dim, self.pad) # 32 (no upscale)
            self.conv_mesh = nn.Conv2d(64, 3, 5, padding=(2, 0))
            
            # Zero-initialize for smoothness
            self.conv_mesh.weight.data[:] = 0
            self.conv_mesh.bias.data[:] = 0
        
        total_params = 0
        for param in self.parameters():
            total_params += param.nelement()

    def forward(self, z, c=None, caption=None, return_attention=False):
        if self.args.conditional_class:
            assert c is not None
            c_emb = self.emb_class(c[:, 0])
            if self.args.conditional_color:
                c_col = self.emb_color(c[:, 1])
                z = torch.cat((z, c_emb, c_col), dim=1) # Concatenate class and color to random vector
            else:
                z = torch.cat((z, c_emb), dim=1) # Concatenate class embedding to random vector
        
        x = self.fc(z)
        x = x.view(x.shape[0], -1, self.height, self.width)
        x = self.up(self.blk1(x, z)) # x2
        x = self.blk2(x, z) # x4
        
        if self.args.conditional_text:
            attention_output, attention_map = self.att(x, *caption)
            x += attention_output
        else:
            attention_map = None
        
        x = self.up(x)
        
        x_tex = x
        if self.args.texture_resolution >= 256:
            x_tex = self.up(self.blk3a(x_tex, z)) # x8
        if self.args.texture_resolution >= 512:
            x_tex = self.up(self.blk3b(x_tex, z)) # x16
        if self.args.texture_resolution >= 1024:
            x_tex = self.up(self.blk3c(x_tex, z)) # x32
        x_tex = self.up(self.blk4(x_tex, z))
        x_tex = self.up(self.blk5(x_tex, z))
        x_tex = self.relu(self.blk6(x_tex, z)) # No upscaling
        x_tex = self.conv_final(self.pad(x_tex, 2)).tanh_()
        
        if self.mesh_head:
            x_mesh = self.relu(self.blk3_mesh(x, z)) # No upscaling
            x_mesh = self.conv_mesh(self.pad(x_mesh, 2))
            x_mesh = adjust_poles(x_mesh)
        else:
            x_mesh = None
        
        if self.symmetric:
            x_tex = symmetrize_texture(x_tex)
            if self.mesh_head:
                x_mesh = symmetrize_texture(x_mesh)
            if attention_map is not None:
                attention_map = symmetrize_texture(attention_map)
        
        if return_attention:
            return x_tex, x_mesh, attention_map
        else:
            return x_tex, x_mesh
        
        

# Attention mechanism adapted from the implementation in AttnGAN:
# https://github.com/taoxugit/AttnGAN/blob/0d000e652b407e976cb88fab299e8566f3de8a37/code/GlobalAttention.py#L72-L121
# In this revision, we fixed the masking and removed deprecation warnings.
class SpatialAttention(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        #input_dim = amount of channels from input (=images)
        #context_dim = dimension from context (= text [batch_size, context_dim, sequence_length])
        self.conv_context = nn.Conv2d(context_dim, input_dim, 1, stride=1, padding=0, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """

        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = mask.unsqueeze(1).expand(-1, queryL, -1).contiguous().view(batch_size*queryL, -1)
            attn = attn + mask.float() * -10000
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn