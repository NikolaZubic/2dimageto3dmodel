# Adapted from:
# https://github.com/akanazawa/cmr/blob/c24cab6aececa1cb8416ccb2d3ee470915726937/data/base.py

"""
Base data loading class.

Should results:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

import os.path as osp
import numpy as np
import copy

import scipy.linalg
import scipy.ndimage.interpolation
from skimage.io import imread

import torch
from torch.utils.data import Dataset

from . import image_utils
from . import transformations


# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, kp, pose data loader
    '''

    def __init__(self, is_train, img_size):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        if not isinstance(img_size, list):
            self.img_sizes = [img_size]
        else:
            self.img_sizes = img_size
        self.jitter_frac = 0
        self.padding_frac = 0.05
        self.is_train = is_train
    
    def get_paths(self):
        paths = []
        for index, data in enumerate(self.anno):
            img_path_rel = str(data.rel_path).replace('\\', '/')
            paths.append(img_path_rel)
        return paths
    
    def forward_img(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]

        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path)).replace('\\', '/')
        img_path_rel = str(data.rel_path).replace('\\', '/')
        img = imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox
        if self.is_train:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        bbox = image_utils.square_bbox(bbox)
        true_resolution = bbox[2] - bbox[0] + 1

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)
        
        mirrored = self.is_train and (torch.randint(0, 2, size=(1,)).item() == 1)

        # scale image, and mask. And scale kps.
        img_ref, mask_ref, kp_ref, sfm_pose_ref = self.scale_image(img.copy(), mask.copy(),
                                                                   kp.copy(), vis.copy(),
                                                                   copy.deepcopy(sfm_pose),
                                                                   self.img_sizes[0])
        if mirrored:
            img_ref, mask_ref, kp_ref, sfm_pose_ref = self.mirror_image(img_ref, mask_ref, kp_ref, sfm_pose_ref)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img_ref.shape[:2]
        kp_norm, sfm_pose_ref = self.normalize_kp(kp_ref, sfm_pose_ref, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img_ref = np.transpose(img_ref, (2, 0, 1))
        
        # Compute other resolutions (if requested)
        extra_res = {}
        for res in self.img_sizes[1:]:
            img2, mask2, kp2, sfm_pose2 = self.scale_image(img.copy(), mask.copy(),
                                                           kp.copy(), vis.copy(),
                                                           copy.deepcopy(sfm_pose),
                                                           res)
            if mirrored:
                img2, mask2, kp2, sfm_pose2 = self.mirror_image(img2, mask2, kp2, sfm_pose2)
                
            img2 = np.transpose(img2, (2, 0, 1))
            extra_res[res] = (img2, mask2)

        return img_ref, kp_norm, mask_ref, sfm_pose_ref, mirrored, img_path_rel, extra_res

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0/img_w + 1.0/img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose, img_size):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        kp_perm = self.kp_perm
        
        # Need copy bc torch collate doesnt like neg strides
        img_flip = img[:, ::-1, :].copy()
        mask_flip = mask[:, ::-1].copy()

        # Flip kps.
        new_x = img.shape[1] - kp[:, 0] - 1
        kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
        kp_flip = kp_flip[kp_perm, :]
        # Flip sfm_pose Rot.
        R = transformations.quaternion_matrix(sfm_pose[2])
        flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
        sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
        # Flip tx
        tx = img.shape[1] - sfm_pose[1][0] - 1
        sfm_pose[1][0] = tx
        return img_flip, mask_flip, kp_flip, sfm_pose

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img, kp, mask, sfm_pose, mirrored, path, extra_res = self.forward_img(index)
        sfm_pose[0].shape = 1

        elem = {
            'img': img,
            'kp': kp,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),
            'mirrored': mirrored,
            'inds': index,
            'path': path,
        }
        
        for res, img2 in extra_res.items():
            elem[f'img_{res}'] = img2

        return elem
