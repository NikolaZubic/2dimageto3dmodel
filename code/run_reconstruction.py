import os
# Workaround for PyTorch spawning too many threads
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from packaging import version

try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. Install it to see the progress bar.')
    def tqdm(x): return x

import cv2
cv2.setNumThreads(0) # Prevent opencv from spawning too many threads in the data loaders

import kaolin as kal
from rendering.renderer import Renderer
from rendering.utils import qrot, qmul, circpad, symmetrize_texture, adjust_poles
from rendering.mesh_template import MeshTemplate
from utils.losses import loss_flat

from models.reconstruction import ReconstructionNetwork, DatasetParams

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True, help='(p3d|cub)')
parser.add_argument('--mesh_path', type=str, default='autodetect')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--image_resolution', type=int, default=256)
parser.add_argument('--symmetric', type=bool, default=True)
parser.add_argument('--texture_resolution', type=int, default=128)
parser.add_argument('--mesh_resolution', type=int, default=32)
parser.add_argument('--loss', type=str, default='mse', help='(mse|l1)')

parser.add_argument('--checkpoint_freq', type=int, default=100) # Epochs
parser.add_argument('--evaluate_freq', type=int, default=10) # Epochs
parser.add_argument('--save_freq', type=int, default=10) # Epochs

parser.add_argument('--tensorboard', action='store_true') # Epochs
parser.add_argument('--image_freq', type=int, default=10) # Epochs

parser.add_argument('--no_augmentation', action='store_true')
parser.add_argument('--optimize_deltas', type=bool, default=True)
parser.add_argument('--optimize_z0', action='store_true')
parser.add_argument('--generate_pseudogt', action='store_true')
parser.add_argument('--pseudogt_resolution', type=int, default=512) # Output texture resolution
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--continue_train', action='store_true') # Resume from checkpoint
parser.add_argument('--which_epoch', type=str, default='latest') # Epoch from which to resume (or evaluate)
parser.add_argument('--mesh_regularization', type=float, default=0.00005)

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_dataset', type=float, default=0.0001)
parser.add_argument('--lr_decay_every', type=int, default=250)
parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()


if args.mesh_path == 'autodetect':
    if args.dataset == 'p3d':
        args.mesh_path = 'mesh_templates/uvsphere_31rings.obj'
    elif args.dataset == 'cub':
        args.mesh_path = 'mesh_templates/uvsphere_16rings.obj'
    else:
        raise
    print('Using autodetected mesh', args.mesh_path)

mesh_template = MeshTemplate(args.mesh_path, is_symmetric=args.symmetric)

if args.generate_pseudogt:
    # Ideally, the renderer should run at a higher resolution than the input image,
    # or a sufficiently high resolution at the very least.
    renderer_res = max(1024, 2*args.pseudogt_resolution)
else:
    # Match neural network input resolution
    renderer_res = args.image_resolution
    
renderer = Renderer(renderer_res, renderer_res)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, cmr_dataset, img_size):
        self.cmr_dataset = cmr_dataset
        self.paths = cmr_dataset.get_paths()
        
        self.extra_img_keys = []
        if isinstance(img_size, list):
            for res in img_size[1:]:
                self.extra_img_keys.append(f'img_{res}')

    def __len__(self):
        return len(self.cmr_dataset)

    def __getitem__(self, idx):
        item = self.cmr_dataset[idx]

        # Rescale img to [-1, 1]
        img = item['img'].astype('float32')*2 - 1
        mask = item['mask'].astype('float32')
        img *= mask[np.newaxis, :, :]

        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        ind = torch.LongTensor([idx])
        if item['mirrored']:
            # Indices from 0 to N-1 are straight, from N to 2N-1 are mirrored
            ind += len(self.cmr_dataset)

        scale = torch.FloatTensor(item['sfm_pose'][:1])
        translation = torch.FloatTensor([item['sfm_pose'][1], item['sfm_pose'][2], 0])
        rot = torch.FloatTensor(item['sfm_pose'][-4:])
        output = torch.cat((img, mask), dim=0)
        
        extra_imgs = []
        for k in self.extra_img_keys:
            img_k, mask_k = item[k]
            img_k = img_k.astype('float32')*2 - 1
            mask_k = mask_k.astype('float32')
            img_k *= mask_k[np.newaxis, :, :]
            img_k = torch.FloatTensor(img_k)
            extra_imgs.append(img_k)

        return (output, *extra_imgs, scale, translation, rot, ind)



dataset_type = args.dataset

if not args.generate_pseudogt:
    dataloader_resolution = args.image_resolution
    dataloader_resolution_val = args.image_resolution
else:
    # We need images at different scales
    inception_resolution = 299
    dataloader_resolution = [args.image_resolution, inception_resolution, renderer_res]
    dataloader_resolution_val = inception_resolution

is_train = not (args.no_augmentation or args.evaluate or args.generate_pseudogt)
if dataset_type == 'p3d':
    from cmr_data.p3d import P3dDataset
    
    cmr_ds_train = P3dDataset('train', is_train, dataloader_resolution)
    mesh_ds_train = ImageDataset(cmr_ds_train, dataloader_resolution)
    
    if not args.generate_pseudogt:
        cmr_ds_val = P3dDataset('val', False, dataloader_resolution_val)
        mesh_ds_val = ImageDataset(cmr_ds_val, dataloader_resolution_val)
    else:
        mesh_ds_val = None
    
    debug_ids = [6, 9, 16, 23, 34, 39, 40, 54, 60, 61, 64, 66, 67, 75, 77, 84] # For TensorBoard

elif dataset_type == 'cub':
    from cmr_data.cub import CUBDataset
    
    cmr_ds_train = CUBDataset('train', is_train, dataloader_resolution)
    mesh_ds_train = ImageDataset(cmr_ds_train, dataloader_resolution)
    
    cmr_ds_val = CUBDataset('testval', False, dataloader_resolution_val)
    mesh_ds_val = ImageDataset(cmr_ds_val, dataloader_resolution_val)
    
    debug_ids = [0, 1, 12, 18, 20, 42, 72, 100, 101, 115, 123, 125, 142, 158, 188, 203] # For TensorBoard

else:
    raise


batch_size = args.batch_size
shuffle = not (args.generate_pseudogt or args.evaluate)
train_loader = torch.utils.data.DataLoader(mesh_ds_train, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=args.num_workers, pin_memory=True)

if mesh_ds_val is not None:
    val_loader = torch.utils.data.DataLoader(mesh_ds_val, batch_size=batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)


def render_multiview(raw_vtx, pred_tex, idx=0):
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    with torch.no_grad():
        # Render from multiple viewpoints
        rad = -90 / 180 * np.pi
        q0 = torch.Tensor([np.cos(-rad/2), 0, 0, np.sin(-rad/2)]).cuda()
        rad = 110 / 180 * np.pi
        q1 = torch.Tensor([np.cos(-rad/2), 0, np.sin(-rad/2), 0]).cuda()
        q0 = qmul(q0, q1)

        rot = []
        for angle in angles:
            rad = angle / 180 *np.pi * 0.8
            q = torch.Tensor([np.cos(-rad/2), 0, 0, np.sin(-rad/2)]).cuda()
            q = qmul(q0, q)
            rot.append(q)
        rot = torch.stack(rot, dim=0)

        raw_vtx = raw_vtx[idx:idx+1].expand(rot.shape[0], -1, -1)
        pred_tex = pred_tex[idx:idx+1].expand(rot.shape[0], -1, -1, -1)
        
        vtx = qrot(rot, raw_vtx)*0.9
        vtx[:, :, 1:] *= -1

        pred_view, _ = mesh_template.forward_renderer(renderer, vtx, pred_tex)

        pred_view = pred_view.cpu()
        nrows = 2
        ncols= 4
        pred_view = pred_view.view(nrows, ncols, pred_view.shape[1], pred_view.shape[2], pred_view.shape[3])
        pred_view = pred_view.permute(0, 2, 1, 3, 4).contiguous()
        pred_view = pred_view.view(args.image_resolution*nrows, args.image_resolution*ncols, 3)
        render = (pred_view.cpu().numpy() + 1)/2
        return render



def mean_iou(alpha_pred, alpha_real):
    alpha_pred = alpha_pred > 0.5
    alpha_real = alpha_real > 0.5
    intersection = (alpha_pred & alpha_real).float().sum(dim=[1, 2])
    union = (alpha_pred | alpha_real).float().sum(dim=[1, 2])
    iou = intersection / union
    return torch.mean(iou)


def to_grid(x):
    return torchvision.utils.make_grid((x[:16, :3]+1)/2, nrow=4)

def transform_vertices(vtx, gt_scale, gt_translation, gt_rot, gt_idx):
    if args.optimize_deltas:
        translation_delta, scale_delta = dataset_params(gt_idx, 'deltas')
    else:
        scale_delta = 0
        translation_delta = 0
    vtx = qrot(gt_rot, (gt_scale + scale_delta).unsqueeze(-1)*vtx) + (gt_translation + translation_delta).unsqueeze(1)
    vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)
    if args.optimize_z0:
        z0 = dataset_params(gt_idx, 'z0').unsqueeze(-1)
        z = vtx[:, :, 2:]
        factor = (z0 + z/2)/(z0 - z/2)
        vtx = torch.cat((vtx[:, :, :2]*factor, z), dim=2)
    else:
        assert 'ds_z0' not in dataset_params.__dict__, 'Model was trained with --optimize_z0'
    return vtx

# Full test
def evaluate_all(loader, writer=None, it=0):
    with torch.no_grad():
        generator.eval()
        N = 0
        total_recon_loss = 0
        total_flat_loss = 0
        total_miou = 0
        debug_images_real = []
        debug_images_fake = []
        debug_images_render = []
        for i, (X, gt_scale, gt_translation, gt_rot, _) in enumerate(loader):
            X_real = X.cuda()
            gt_scale = gt_scale.cuda()
            gt_translation = gt_translation.cuda()
            gt_rot = gt_rot.cuda()

            pred_tex, mesh_map = generator(X_real)
            raw_vtx = mesh_template.get_vertex_positions(mesh_map)
        
            vtx = transform_vertices(raw_vtx, gt_scale, gt_translation, gt_rot, None)
            
            image_pred, alpha_pred = mesh_template.forward_renderer(renderer, vtx, pred_tex)
            X_fake = torch.cat((image_pred, alpha_pred), dim=3).permute(0, 3, 1, 2)

            recon_loss = criterion(X_fake, X_real)
            flat_loss = loss_flat(mesh_template.mesh, mesh_template.compute_normals(raw_vtx))
            miou = mean_iou(X_fake[:, 3], X_real[:, 3]) # Done on alpha channel

            total_recon_loss += X.shape[0] * recon_loss.item()
            total_flat_loss += X.shape[0] * flat_loss.item()
            total_miou += X.shape[0] * miou.item()
            N += X.shape[0]
            if writer is not None:
                # Save images
                dbg_ids = np.array(debug_ids)
                min_idx = i * args.batch_size
                max_idx = (i+1) * args.batch_size
                dbg_ids = dbg_ids[(dbg_ids >= min_idx) & (dbg_ids < max_idx)] - min_idx
                if len(dbg_ids) > 0:
                    debug_images_real.append(X_real[dbg_ids].cpu())
                    debug_images_fake.append(X_fake[dbg_ids].cpu())
                    for idx in dbg_ids:
                        if len(debug_images_render) >= 4:
                            break
                        debug_images_render.append(render_multiview(raw_vtx, pred_tex, idx))

        total_recon_loss /= N
        total_flat_loss /= N
        total_miou /= N
        
        if writer is not None:
            writer.add_scalar(args.loss + '/val', total_recon_loss, it)
            writer.add_scalar('flat/val', total_flat_loss, it)
            writer.add_scalar('iou/val', total_miou, it)

        log('[TEST] recon_loss {:.5f}, flat_loss {:.5f}, mIoU {:.5f}, N {}'.format(
                                                                total_recon_loss, total_flat_loss,
                                                                total_miou, N
                                                                ))
        if writer is not None:
            writer.add_image('image_val/real', to_grid(torch.cat(debug_images_real, dim=0)), it)
            writer.add_image('image_val/fake', to_grid(torch.cat(debug_images_fake, dim=0)), it)
            debug_images_render = torch.FloatTensor(debug_images_render).permute(0, 3, 1, 2)
            render_grid = torchvision.utils.make_grid(debug_images_render, nrow=1)
            writer.add_image('image_val/render', render_grid, it)
        
def log_image(X_fake, X_real, writer, it):
    writer.add_image('image_train/real', to_grid(X_real), it)
    writer.add_image('image_train/fake', to_grid(X_fake), it)


import pathlib
import os
if args.tensorboard:
    import shutil
    from torch.utils.tensorboard import SummaryWriter
    import torchvision

generator = ReconstructionNetwork(symmetric=args.symmetric,
                                  texture_res=args.texture_resolution,
                                  mesh_res=args.mesh_resolution,
                                 ).cuda()

optimizer = optim.Adam(generator.parameters(), lr=args.lr)

if args.optimize_deltas or args.optimize_z0:
    dataset_params = DatasetParams(args, len(train_loader.dataset)).cuda()
    optimizer_dataset = optim.Adam(dataset_params.parameters(), lr=args.lr_dataset)
else:
    dataset_params = None
    optimizer_dataset = None

criteria = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
}

criterion = criteria[args.loss]
g_curve = []
total_it = 0
epoch = 0
flat_warmup = 10
    
checkpoint_dir = 'checkpoints_recon/' + args.name
if args.evaluate or args.generate_pseudogt:
    # Load last checkpoint
    chk = torch.load(os.path.join(checkpoint_dir, f'checkpoint_{args.which_epoch}.pth'), map_location=lambda storage, loc: storage)
    if 'epoch' in chk:
        epoch = chk['epoch']
        total_it = chk['iteration']
    generator.load_state_dict(chk['generator'])
    
    if dataset_params is not None:
        dataset_params.load_state_dict(chk['dataset_params'])
    else:
        assert 'dataset_params' not in chk or chk['dataset_params'] is None
        
    if args.continue_train:
        optimizer.load_state_dict(chk['optimizer'])
        if optimizer_dataset is not None:
            optimizer_dataset.load_state_dict(chk['optimizer_dataset_params'])
        
        print(f'Resuming from epoch {epoch}')
    else:
        if 'epoch' in chk:
            print(f'Evaluating epoch {epoch}')
        args.epochs = -1 # Disable training
    chk = None

if args.tensorboard and not (args.evaluate or args.generate_pseudogt):
    log_dir = 'tensorboard_recon/' + args.name
    shutil.rmtree(log_dir, ignore_errors=True) # Delete logs
    writer = SummaryWriter(log_dir)
else:
    writer = None

if not (args.generate_pseudogt or args.evaluate):
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(os.path.join(checkpoint_dir, 'log.txt'), 'a' if args.continue_train else 'w', buffering=1) # Line buffering
    print(' '.join(sys.argv), file=log_file)
else:
    log_file = None

def log(text):
    if log_file is not None:
        print(text, file=log_file)
    print(text)
    

try:
    while epoch < args.epochs:
        generator.train()

        start_time = time.time()
        for i, (X, gt_scale, gt_translation, gt_rot, gt_idx) in enumerate(train_loader):
            X_real = X.cuda()
            gt_scale = gt_scale.cuda()
            gt_translation = gt_translation.cuda()
            gt_rot = gt_rot.cuda()
            if args.optimize_deltas or args.optimize_z0:
                gt_idx = gt_idx.squeeze(-1).cuda()
            else:
                gt_idx = None

            optimizer.zero_grad()
            if optimizer_dataset is not None:
                optimizer_dataset.zero_grad()

            pred_tex, mesh_map = generator(X_real)
            raw_vtx = mesh_template.get_vertex_positions(mesh_map)

            vtx = transform_vertices(raw_vtx, gt_scale, gt_translation, gt_rot, gt_idx)

            image_pred, alpha_pred = mesh_template.forward_renderer(renderer, vtx, pred_tex)
            X_fake = torch.cat((image_pred, alpha_pred), dim=3).permute(0, 3, 1, 2)

            recon_loss = criterion(X_fake, X_real)
            flat_loss = loss_flat(mesh_template.mesh, mesh_template.compute_normals(raw_vtx))

            # Test losses
            with torch.no_grad():
                miou = mean_iou(X_fake[:, 3], X_real[:, 3]) # Done on alpha channel

            flat_coeff = args.mesh_regularization*flat_warmup
            flat_warmup = max(flat_warmup - 0.1, 1)
            loss = recon_loss + flat_coeff*flat_loss
            loss.backward()
            optimizer.step()
            if optimizer_dataset is not None:
                optimizer_dataset.step()
            g_curve.append(loss.item())


            if total_it % 10 == 0:
                log('[{}] epoch {}, {}/{}, recon_loss {:.5f} flat_loss {:.5f} total {:.5f} iou {:.5f}'.format(
                                                                        total_it, epoch, i, len(train_loader),
                                                                        recon_loss.item(), flat_loss.item(),
                                                                        loss.item(), miou.item(),
                                                                        ))

            if args.tensorboard:
                writer.add_scalar(args.loss + '/train', recon_loss.item(), total_it)
                writer.add_scalar('flat/train', flat_loss.item(), total_it)
                writer.add_scalar('iou/train', miou.item(), total_it)


            total_it += 1

        epoch += 1

        log('Time per epoch: {:.3f} s'.format(time.time() - start_time))


        if epoch % args.lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        def save_checkpoint(it):
            torch.save({
                'optimizer': optimizer.state_dict(),
                'generator': generator.state_dict(),
                'optimizer_dataset_params': optimizer_dataset.state_dict() if optimizer_dataset is not None else None,
                'dataset_params': dataset_params.state_dict() if dataset_params is not None else None,
                'epoch': epoch,
                'iteration': total_it,
                'args': vars(args),
            }, os.path.join(checkpoint_dir, f'checkpoint_{it}.pth'))

        if epoch % args.save_freq == 0:
            save_checkpoint('latest')
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(str(epoch))
        if args.tensorboard and epoch % args.image_freq == 0:
            log_image(X_fake, X_real, writer, total_it)
        if epoch % args.evaluate_freq == 0:
            evaluate_all(val_loader, writer, total_it)
        
except KeyboardInterrupt:
    print('Aborted.')
        
if not (args.generate_pseudogt or args.evaluate):
    save_checkpoint('latest')
elif args.evaluate:
    evaluate_all(val_loader, writer, total_it)
elif args.generate_pseudogt:
    from utils.fid import calculate_stats, init_inception, forward_inception_batch
    
    inception_model = init_inception().cuda().eval()
    
    print('Exporting pseudo-ground-truth data...')
    
    class InverseRenderer(nn.Module):
        def __init__(self, mesh, res_h, res_w):
            super().__init__()

            self.res = (res_h, res_w)
            self.inverse_renderer = Renderer(res_h, res_w)
            self.mesh = mesh

        def forward(self, predicted_vertices, target):
            with torch.no_grad():
                tex = target # The texture is the target image
                uvs = (predicted_vertices[..., :2] + 1)/2
                vertices = self.mesh.uvs.unsqueeze(0)*2 - 1
                vertices = torch.cat((vertices, torch.zeros_like(vertices[..., :1])), dim=-1)
                image_pred, alpha_pred, _ = self.inverse_renderer(points=[vertices.expand(target.shape[0], -1, -1),
                          self.mesh.face_textures],
                          uv_bxpx2=uvs,
                          texture_bx3xthxtw=tex,
                          ft_fx3=self.mesh.faces,
                          return_hardmask=True,
                         )
            return image_pred, alpha_pred

    inverse_renderer = InverseRenderer(mesh_template.mesh, args.pseudogt_resolution, args.pseudogt_resolution)

    cache_dir = os.path.join('cache', args.dataset)
    pseudogt_dir = os.path.join(cache_dir, f'pseudogt_{args.pseudogt_resolution}x{args.pseudogt_resolution}')
    pathlib.Path(pseudogt_dir).mkdir(parents=True, exist_ok=True)
    
    all_path = []
    all_gt_scale = []
    all_gt_translation = []
    all_gt_rotation = []
    all_inception_activation = []
    
    generator.eval()
    for net_image, inception_image, hd_image, gt_scale, gt_translation, gt_rot, indices in tqdm(train_loader):
        # Compute visibility mask
        with torch.no_grad():    
            net_image = net_image.cuda()
            gt_scale = gt_scale.cuda()
            gt_translation = gt_translation.cuda()
            gt_rot = gt_rot.cuda()
            if args.optimize_deltas or args.optimize_z0:
                gt_idx = indices.squeeze(-1).cuda()
            else:
                gt_idx = indices.cuda()

            pred_tex, mesh_map = generator(net_image)
            raw_vtx = mesh_template.get_vertex_positions(mesh_map)

            vtx = transform_vertices(raw_vtx, gt_scale, gt_translation, gt_rot, gt_idx)
            if pred_tex.shape[2] > renderer_res//8:
                # To reduce aliasing in the gradients from the renderer,
                # the rendering resolution must be much higher than the texture resolution.
                # As a rule of thumb, we came up with render_res >= 8*texture_res
                # This is already ensured by the default hyperparameters (1024 and 128).
                # If not, the texture is resized.
                pred_tex = F.interpolate(pred_tex, size=(renderer_res//8, renderer_res//8),
                                         mode='bilinear', align_corners=False)
        
        pred_tex.requires_grad_()
        image_pred, alpha_pred = mesh_template.forward_renderer(renderer, vtx, pred_tex)

        # Compute gradient
        visibility_mask, = torch.autograd.grad(image_pred, pred_tex, torch.ones_like(image_pred))
        
        with torch.no_grad():
            # Compute inception activations
            all_inception_activation.append(forward_inception_batch(inception_model, inception_image.cuda()/2 + 0.5))
            
            # Project ground-truth image onto the UV map
            inverse_tex, inverse_alpha = inverse_renderer(vtx, hd_image.cuda())

            # Mask projection using the visibility mask
            mask = F.interpolate(visibility_mask, args.pseudogt_resolution,
                                 mode='bilinear', align_corners=False).permute(0, 2, 3, 1).cuda()
            mask = (mask > 0).any(dim=3, keepdim=True).float()
            inverse_tex *= mask
            inverse_alpha *= mask
            
            inverse_tex = inverse_tex.permute(0, 3, 1, 2)
            inverse_alpha = inverse_alpha.permute(0, 3, 1, 2)
            
            # Convert to half to save disk space
            inverse_tex = inverse_tex.half().cpu()
            inverse_alpha = inverse_alpha.half().cpu()
            
            all_gt_scale.append(gt_scale.cpu().clone())
            all_gt_translation.append(gt_translation.cpu().clone())
            all_gt_rotation.append(gt_rot.cpu().clone())
            for i, idx in enumerate(indices):
                idx = idx.item()
                all_path.append(train_loader.dataset.paths[idx])
                
                pseudogt = {
                    'mesh': mesh_map[i].cpu().clone(),
                    'texture': inverse_tex[i].clone(),
                    'texture_alpha': inverse_alpha[i].clone(),
                    'image': inception_image[i].half().clone(),
                }
                
                np.savez_compressed(os.path.join(pseudogt_dir, f'{idx}'),
                                    data=pseudogt,
                                   )
    
    print('Saving pose metadata...')
    poses_metadata = {
        'scale': torch.cat(all_gt_scale, dim=0),
        'translation': torch.cat(all_gt_translation, dim=0),
        'rotation': torch.cat(all_gt_rotation, dim=0),
        'path': all_path,
    }
    np.savez_compressed(os.path.join(cache_dir, 'poses_metadata'),
                        data=poses_metadata,
                       )
    
    print('Saving precomputed FID statistics (train)...')
    all_inception_activation = np.concatenate(all_inception_activation, axis=0)
    
    if args.dataset == 'p3d':
        # For Pascal3D+, we only use images that are part of the ImageNet subset
        imagenet_indices = [i for i, p in enumerate(all_path) if p.startswith('car_imagenet')]
        all_inception_activation = all_inception_activation[imagenet_indices]
    
    m_real, s_real = calculate_stats(all_inception_activation)
    fid_save_path = os.path.join(cache_dir, f'precomputed_fid_{inception_resolution}x{inception_resolution}_train')
    np.savez_compressed(fid_save_path, 
                        stats_m=m_real,
                        stats_s=np.tril(s_real.astype(np.float32)), # The matrix is symmetric, we only store half of it
                        num_images=len(all_inception_activation),
                        resolution=inception_resolution,
                       )
    
    if args.dataset == 'cub':
        print('Saving precomputed FID statistics (testval)...')
        val_inception_activation = []
        for inception_image, _, _, _, _ in tqdm(val_loader):
            with torch.no_grad():
                val_inception_activation.append(
                    forward_inception_batch(inception_model, inception_image[:, :3].cuda()/2 + 0.5))
        
        val_inception_activation = np.concatenate(val_inception_activation, axis=0)
        m_real, s_real = calculate_stats(val_inception_activation)
        fid_save_path = os.path.join(cache_dir, f'precomputed_fid_{inception_resolution}x{inception_resolution}_testval')
        np.savez_compressed(fid_save_path, 
                            stats_m=m_real,
                            stats_s=np.tril(s_real.astype(np.float32)), # The matrix is symmetric, we only store half of it
                            num_images=len(val_inception_activation),
                            resolution=inception_resolution,
                           )
    
    print('Done.')

if writer is not None:
    writer.close()