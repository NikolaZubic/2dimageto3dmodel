# author: Nikola Zubic

import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import numpy as np
import torchvision.transforms.functional as T
from PIL import Image
from scipy.io import loadmat
from ..quaternions.blender_camera_quaternions import blender_camera_position_to_torch_tensor_quaternion


"""
We are using splits and renders for Shape-Net dataset just like in the Insafutdinov and Dosovitskiy paper.
We train on train + test, and validate on validation set.
There is an option in dataset to load all the camera positions instead of other views. Here, we load all 5 renders by
default.
"""


def get_models(file_path=".", shape_net_id="03001627", split="train"):
    # Read model paths from the split file
    file_path = Path(file_path)

    assert split in ("train", "valid")
    split = file_path / f"{shape_net_id}.{split}"
    data = file_path / shape_net_id

    with open(split) as models:
        return [data / m.strip() for m in models]


class ShapeNet(Dataset):
    # Dataset with renders and views for ShapeNet category

    def __init__(self, models, camera=True, image_size=128):
        self.models = models
        self.camera = camera
        self.image_size = image_size

    def __getitem__(self, idx):
        model = self.models[idx]
        images, masks, cameras = [], [], []

        for name in sorted(os.listdir(model)):
            if name.startswith("render"):
                o = np.array(T.resize(Image.open(model / name), (self.image_size, self.image_size)))
                mask = o[..., -1].astype(np.float32) / 255.
                img = o[..., :-1].astype(np.float32) / 255.

                images.append(torch.tensor(img).permute(2, 0, 1))
                masks.append(torch.tensor(mask))

            if name.startswith("camera"):
                camera = loadmat(model / name)
                cameras.append(blender_camera_position_to_torch_tensor_quaternion(camera["pos"]))

        images = torch.stack(images)
        masks = torch.stack(masks)
        if self.camera:
            poses = torch.stack(cameras)
        else:
            poses = images

        return images, poses, masks

    def __len__(self):
        return len(self.models)


def multi_view_collate(batch):
    """
    Sampling of one image for point cloud generation. Views can be cameras or images (depending on the current dataset,
    so it is an information that tells us something about other views).

    :param batch: given batch
    :return: images, poses and masks
    """
    batch_size = len(batch)
    number_of_poses = batch[0][0].size(0)

    indexes = torch.randint(0, number_of_poses, size=(batch_size,))
    images, poses, masks = zip(*[(img[i], view, mask) for (img, view, mask), i in zip(batch, indexes)])

    images = torch.stack(images)
    poses = torch.cat(poses, dim=0)
    masks = torch.cat(masks, dim=0)

    return images, poses, masks


class DataBunch(object):
    # Datasets and data-loaders
    _ids = {
        "chairs": "03001627",
        "planes": "02691156",
        "cars": "02958343",
    }

    def __init__(self, file_path, category_of_choice="chairs", batch_size=10, image_size=128, is_camera_used=True):
        train = get_models(file_path, self._ids[category_of_choice], "train")
        valid = get_models(file_path, self._ids[category_of_choice], "valid")
        self.train_ds, self.valid_ds = ShapeNet(train, is_camera_used, image_size), ShapeNet(valid, is_camera_used,
                                                                                             image_size)
        self.train_dl = DataLoader(
            self.train_ds, batch_size,
            shuffle=True, collate_fn=multi_view_collate, drop_last=True,
            pin_memory=torch.cuda.is_available(), num_workers=4,
        )
        self.valid_dl = DataLoader(
            self.valid_ds, batch_size * 2,
            shuffle=False, collate_fn=multi_view_collate,
            pin_memory=torch.cuda.is_available(),
        )
