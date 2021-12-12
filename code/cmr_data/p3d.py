# Adapted from:
# https://github.com/akanazawa/cmr/blob/de06f7189c7e84e9563c4eaf972808b4c81beb44/data/p3d.py

"""
Data loader for pascal VOC categories.

Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

import os.path as osp
import numpy as np

import scipy.io as sio

import torch

from . import base as base_data

# -------------- Dataset ------------- #
# ------------------------------------ #
class P3dDataset(base_data.BaseDataset):
    ''' 
    VOC Data loader
    '''

    def __init__(self, split, is_train, img_size, p3d_class='car'):
        super().__init__(is_train, img_size)
        
        curr_path = osp.dirname(osp.abspath(__file__))
        p3d_anno_path = osp.join(curr_path, '..', 'datasets', 'p3d')
        
        self.img_dir = osp.join(p3d_anno_path, 'PASCAL3D+_release1.1', 'Images')
        
        self.kp_path = osp.join(
            p3d_anno_path, 'data', '{}_kps.mat'.format(p3d_class))
        self.anno_path = osp.join(
            p3d_anno_path, 'data', '{}_{}.mat'.format(p3d_class, split))
        self.anno_sfm_path = osp.join(
            p3d_anno_path, 'sfm', '{}_{}.mat'.format(p3d_class, split))

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1

        self.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        
