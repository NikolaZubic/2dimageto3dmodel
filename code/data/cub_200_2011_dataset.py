# Author: Nikola Zubic

from .abstract_dataset import AbstractDataset

import numpy as np
import torch
import os


class CubDataset(AbstractDataset):
    """
    Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the
    number of images per class and new part location annotations.

    Number of categories: 200
    Number of images: 11,788
    Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box
    """
    def __init__(self, args, **kwargs):
        """
        Class that represents CUB birds dataset
        :param args: all args of interest
        :param kwargs: all kwargs of interest
        """
        super().__init__(args, **kwargs)
        
        self.n_classes = (200,)  # 200 bird species (categories)
        args.n_classes = self.n_classes

        # Load CUB labels
        cub_path = 'datasets/cub/CUB_200_2011'

        with open(os.path.join(cub_path, 'images.txt'), 'r') as f:
            images = f.readlines()
            images = [x.split(' ') for x in images]
            ids = {k: v.strip() for k, v in images}

        with open(os.path.join(cub_path, 'image_class_labels.txt'), 'r') as f:
            classes = f.readlines()
            classes = [x.split(' ') for x in classes]
            classes = {k: int(v.strip())-1 for k, v in classes}

        self.filename_to_class = {}
        for k, c in classes.items():
            fname = ids[k]
            self.filename_to_class[fname] = c

        self.classes = [np.array([self.filename_to_class[x]]) for x in self.data['path']]

        num_images = len(self.data['path'])

        print('\nCUB 200-2011 dataset with {} images is successfully loaded.\n'.
              format(num_images))
    
    def name(self):
        return 'cub'
    
    def suggest_truncation_sigma(self):
        args = self.args
        if args.conditional_class:
            return 0.25
        elif args.conditional_text:
            return 0.5
        else:  # Unconditional
            return 1.0
        
    def suggest_num_discriminators(self):
        if self.args.texture_resolution >= 512:
            return 3
        else:
            return 2
    
    def suggest_mesh_template(self):
        return 'mesh_templates/uvsphere_16rings.obj'
    
    def get_random_caption(self, idx):
        # Randomly select a sentence belonging to image idx
        sent_ix = torch.randint(0, self.text_processor.embeddings_num, size=(1,)).item()
        new_sent_ix = self.image_index_to_caption_index[idx] * self.text_processor.embeddings_num + sent_ix
        return self.text_processor.get_caption(new_sent_ix)  # Tuple (padded tokens, lengths)
    
    def __getitem__(self, idx):
        ground_truth_dict = super().__getitem__(idx)
        
        if self.args.conditional_text:
            ground_truth_dict['caption'] = self.get_random_caption(idx)
            
        return ground_truth_dict
