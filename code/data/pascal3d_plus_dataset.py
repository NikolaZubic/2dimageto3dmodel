from .abstract_dataset import AbstractDataset

import numpy as np


class Pascal3DPlusDataset(AbstractDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
            
        # Select only images that are part of ImageNet
        imagenet_indices = [i for i, p in enumerate(self.data['path']) if p.startswith('car_imagenet')]
        self.imagenet_indices = imagenet_indices
        paths = self.data['path']
        self.data['path'] = [paths[i] for i in imagenet_indices]
        paths = self.data['path']
        num_images = len(paths)  # Update

        self.data['scale'] = self.data['scale'][imagenet_indices]
        self.data['translation'] = self.data['translation'][imagenet_indices]
        self.data['rotation'] = self.data['rotation'][imagenet_indices]

        filenames = [p.split('/')[-1] for p in paths]
        path_synsets = [f.split('_')[0] for f in filenames]

        mapping, self.n_classes = Pascal3DPlusDataset.get_p3d_labels()
        args.n_classes = self.n_classes
        self.classes = [mapping[f] for f in filenames]

        print('\nPascal 3D+ dataset with {} images is successfully loaded.\n'.format(num_images))
    
    def name(self):
        return 'p3d'
    
    def suggest_truncation_sigma(self):
        args = self.args
        if args.conditional_class and args.conditional_color:
            return 0.5
        elif args.conditional_class:
            return 0.75
        else: # Unconditional
            return 1.0
        
    def suggest_num_discriminators(self):
        # For P3D, the default setting is to use 2 discriminators regardless of the texture resolution
        return 2
    
    def suggest_mesh_template(self):
        return 'mesh_templates/uvsphere_31rings.obj'
    
    def load_pseudo_ground_truth(self, idx):
        # Remap indices to ImageNet indices
        return super().load_pseudo_ground_truth(self.imagenet_indices[idx])
    
    @staticmethod
    def get_p3d_labels():
        with open('datasets/p3d/p3d_labels.csv', 'r') as csv:
            lines = csv.readlines()[1:]  # Skip header
            filenames = []
            colors1 = []
            colors2 = []
            shapes = []
            for line in lines:
                filename, col1, col2, shape, _ = line.strip().split(',')
                filenames.append(filename)
                colors1.append(col1)
                colors2.append(col2)
                shapes.append(shape)
            col1_names = sorted(set(colors1))
            col2_names = sorted(set(colors2))
            shape_names = sorted(set(shapes))
            col1_to_id = {x: i for i, x in enumerate(col1_names)}
            col2_to_id = {x: i for i, x in enumerate(col2_names)}
            shape_to_id = {x: i for i, x in enumerate(shape_names)}
            mapping = {}
            for filename, shape, col1, col2 in zip(filenames, shapes, colors1, colors2):
                mapping[filename] = np.array([shape_to_id[shape], col1_to_id[col1], col2_to_id[col2]])
            n_classes = (len(shape_names), len(col1_names), len(col2_names))
        return mapping, n_classes
