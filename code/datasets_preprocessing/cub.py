# Adapted from:
# https://github.com/akanazawa/cmr/blob/c24cab6aececa1cb8416ccb2d3ee470915726937/data/cub.py

"""
CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.

"""

import os.path as osp
import numpy as np

import scipy.io as sio

import torch
from torch.utils.data import Dataset

from . import base as base_data

# -------------- Dataset ------------- #
# ------------------------------------ #
class CUBDataset(base_data.BaseDataset):
    '''
    CUB Data loader
    '''

    def __init__(self, split, is_train, img_size):
        super().__init__(is_train, img_size)
        
        curr_path = osp.dirname(osp.abspath(__file__))
        cache_path = osp.join(curr_path, '..', 'datasets', 'cub')
        self.data_cache_dir = cache_path
        self.data_dir = osp.join(cache_path, 'CUB_200_2011')

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % split)

        if not osp.exists(self.anno_path):
            raise ValueError('%s doesnt exist!' % self.anno_path)

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
