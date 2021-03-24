# Author: Nikola Zubic

import os
import numpy as np
import glob
import torch

"""
An abstract class representing a Dataset.
All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite 
__getitem__(), supporting fetching a data sample for a given key. Subclasses could also optionally overwrite __len__(), 
which is expected to return the size of the dataset by many Sampler implementations and the default options of 
DataLoader.
"""


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, args, augment=True):
        """
        Abstract Dataset
        :param args: all args of interest
        :param augment: if needed
        """

        # cache for poses and metadata
        self.args, self.cache_dir = args, os.path.join("cache", args.dataset)

        # Load arrays or pickled objects from .npy, .npz or pickled files
        self.data = np.load(os.path.join(self.cache_dir, 'poses_metadata.npz'), allow_pickle=True)
        self.data = self.data['data'].item()
        number_of_images = len(self.data['path'])
        self.augment = augment
        
        """
        Check if pseudo-ground-truth is available.
        Projection of ground-truth images onto the UV map, producing pseudo-ground-truth textures
        """
        pseudo_ground_truth_files = glob.glob(os.path.join(self.cache_dir,
                                                f"pseudogt_{args.texture_resolution}x{args.texture_resolution}",
                                                "*.npz"))
        if len(pseudo_ground_truth_files) == 0:
            self.has_pseudo_ground_truth = False
        elif len(pseudo_ground_truth_files) == number_of_images:
            self.has_pseudo_ground_truth = True
        else:
            raise ValueError('Found pseudo-ground-truth directory, but number of files does not match! '
                            f'Expected {number_of_images}, got {len(pseudo_ground_truth_files)}. '
                             'Please check your dataset setup.')
            
        if not self.has_pseudo_ground_truth and not args.evaluate:
            raise ValueError('Training a model requires the pseudo-ground-truth to be setup beforehand.')
    
    def name(self):
        raise NotImplementedError()
        
    def suggest_truncation_sigma(self):
        raise NotImplementedError()
        
    def suggest_num_discriminators(self):
        raise NotImplementedError()
        
    def suggest_mesh_template(self):
        raise NotImplementedError()
    
    def __len__(self):
        return len(self.data['path'])
    
    def load_pseudo_ground_truth(self, idx):
        tex_res = self.args.texture_resolution
        data = np.load(os.path.join(self.cache_dir,
                       f'pseudogt_{tex_res}x{tex_res}',
                       f'{idx}.npz'), allow_pickle=True)
        
        data = data['data'].item()
        ground_truth_dict = {
            'image': data['image'][:3].float()/2 + 0.5,
            'texture': data['texture'].float(),
            'texture_alpha': data['texture_alpha'].float(),
            'mesh': data['mesh']
        }
        return ground_truth_dict
    
    def __getitem__(self, idx):
        ground_truth_dict = self.load_pseudo_ground_truth(idx)
        del ground_truth_dict['image']  # Not needed

        # Data augmentation that doesn't require re-rendering, mirroring in UV space
        if self.augment and not self.args.evaluate:
            if torch.randint(0, 2, size=(1,)).item() == 1:
                for k, v in ground_truth_dict.items():
                    ground_truth_dict[k] = AbstractDataset.mirror_tex(v)
        
        if self.args.conditional_class:
            ground_truth_dict['class'] = self.classes[idx]
        
        ground_truth_dict['idx'] = idx
        return ground_truth_dict
    
    @staticmethod
    def mirror_tex(tr):
        # "Virtually" flip a texture or displacement map of shape (nc, H, W)
        # This is achieved by mirroring the image and shifting the u coordinate,
        # which is consistent with reprojecting the mirrored 2D image.
        tr = torch.flip(tr, dims=(2,))
        tr = torch.cat((tr, tr), dim=2)
        tr = tr[:, :, tr.shape[2]//4:-tr.shape[2]//4]
        return tr
    

class AbstractDatasetForEvaluation(torch.utils.data.Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ground_truth_dict = {
            'scale': self.dataset.data['scale'][idx],
            'translation': self.dataset.data['translation'][idx],
            'rotation': self.dataset.data['rotation'][idx],
            'idx': idx,
        }
        
        if self.dataset.args.conditional_class:
            ground_truth_dict['class'] = self.dataset.classes[idx]
        
        if self.dataset.args.conditional_text:
            ground_truth_dict['caption'] = self.dataset.index_captions[idx]  # Tuple (padded tokens, lengths)
            
        if self.dataset.has_pseudo_ground_truth:
            # Add pseudo-ground-truth entries
            ground_truth_dict.update(self.dataset.load_pseudo_ground_truth(idx))

        return ground_truth_dict
