import torch
from pathlib import Path
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
from collections import defaultdict
import torch.nn.functional as F
from torchvision.utils import make_grid
from data.shape_net_dataset import DataBunch
from models.unsupervised_part import UnsupervisedPart, UnsupervisedLoss


def data_loader_loop(data_loader):
    # loop through data-loader
    while True:
        for obj in data_loader:
            yield obj


def parameter_adjustment(model, step, p=(0.07, 1.0), sigma=(3.0, 0.2)):
    # Linear scheduling of model parameters (dropout probability p and smoothing factor sigma)
    assert 0 <= step <= 1

    new_p = p[0] * (1 - step) + p[1] * step
    new_sigma = sigma[0] * (1 - step) + sigma[1] * step

    model.point_cloud_drop_out.p = new_p
    model.effective_loss_function.sigma = torch.empty_like(model.effective_loss_function.sigma).fill_(new_sigma)


class Learner(object):
    # Training model for ShapeNet Dataset
    def __init__(self, file_path, data, model, loss, learning_rate=1e-4, weight_decay=0.001, seed=100):
        torch.random.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_path = Path(file_path)

        if self.file_path.exists():
            # Recursively delete a directory tree
            shutil.rmtree(self.file_path)

        (self.file_path/"models").mkdir(exist_ok=True, parents=True)

        self.train_writer = SummaryWriter(log_dir=self.file_path/"logs"/"train")
        self.valid_writer = SummaryWriter(log_dir=self.file_path/"logs"/"valid")

        self.valid_losses = []

        self.data = data
        self.model = model.to(self.device)
        self.loss = loss

        """
        The authors show experimentally that AdamW yields better training loss and that the models generalize much 
        better than models trained with Adam allowing the new version to compete with stochastic gradient descent with 
        momentum.
        """
        self.adam_w_optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate,
                                                  weight_decay=weight_decay)

        self.step = None
        self.images = None
        self.poses = None
        self.masks = None

        self.progress_bar = None

    def one_batch(self):
        # Runs one batch of a model
        device = self.device

        images, poses, masks = self.images.to(device), self.poses.to(device), self.masks.to(device)

        projection = self.model(images, poses)

        t1 = time.perf_counter()

        loss = self.loss(projection, masks, training=self.model.training)

        if not self.model.training:
            self.valid_losses.append({key: l.item() for key, l in loss.items()})
            return

        loss["full_loss"].backward()

        self.adam_w_optimizer.step()
        self.adam_w_optimizer.zero_grad()

        dt = time.perf_counter() - t1

        self.progress_bar.set_postfix(time=dt, loss=loss["full_loss"].item())

        min_indexes = getattr(self.loss, "min_indexes", None)

        if min_indexes is not None:
            self.train_writer.add_histogram("other/predictors", min_indexes.cpu(), self.step)

        for key, l in loss.items():
            self.train_writer.add_scalar(key, l.item(), self.step)

    def write_validation_losses(self):
        # Calculating means for all validation losses and writing them to tensorboard
        all_means = defaultdict(int)

        for loss in self.valid_losses:
            for key, value in loss.items():
                all_means[key] += value

        for key in all_means.keys():
            self.valid_writer.add_scalar(key, all_means[key] / len(self.valid_losses), self.step)

        self.valid_losses = []

    @staticmethod
    def generate_image_projections(model, images, poses, masks):
        # Generates grid with model projections, input images

        # Get current model parameters to device
        device = next(model.parameters()).device

        # Get projections for current model based on images and poses
        projection, *_ = model(images[0].unsqueeze(0).to(device), poses.to(device))
        projection = projection.detach().cpu()

        grid = torch.cat([
            F.interpolate(images, scale_factor=1 / 2, mode='bilinear', align_corners=True),
            F.interpolate(masks.unsqueeze(1), scale_factor=1 / 2, mode='bilinear', align_corners=True).repeat(1, 3, 1,
                                                                                                              1),
            projection.unsqueeze(1).repeat(1, 3, 1, 1),
        ])

        grid = make_grid(grid, nrow=images.size(0))
        return F.interpolate(grid.unsqueeze(0), scale_factor=2)

    def fit(self, number_of_steps=300000, evaluation_frequency_steps=10000, visualization_frequency_steps=1000,
            p=(0.07, 1.0), sigma=(3.0, 0.2), restore=None, start=None):
        # Train model for a certain number of steps 'number_of_steps'
        start = 0

        if restore is not None:
            check_point = torch.load(restore, map_location="cpu")
            self.model.load_state_dict(check_point["model"])
            self.adam_w_optimizer.load_state_dict(check_point["opt"])
            start = check_point["step"] if start is None else start

        train_data_loader = data_loader_loop(data_loader=self.data.train_dl)  # train_dl from shape_net_dataset.py

        self.progress_bar = tqdm(range(start + 1, number_of_steps + 1), desc="Step")

        for step in self.progress_bar:
            self.model.train()

            parameter_adjustment(model=self.model, step=step/number_of_steps, p=p, sigma=sigma)

            self.step = step
            self.images, self.poses, self.masks = next(train_data_loader)
            self.one_batch()

            if step % evaluation_frequency_steps == 0:
                self.model.eval()

                with torch.no_grad():
                    for self.images, self.poses, self.masks in tqdm(self.data.valid_dl, leave=False):
                        self.one_batch()
                    self.write_validation_losses()

                torch.save(
                    dict(model=self.model.state_dict(), opt=self.adam_w_optimizer.state_dict(), step=self.step),
                    self.file_path/"models"/f"model_{self.step}.pth"
                )

            if step % visualization_frequency_steps == 0:
                self.model.eval()
                images, poses, masks = self.data.valid_ds[10]

                renders = self.generate_image_projections(self.model, images, poses, masks)

                self.train_writer.add_images("renders", renders, self.step)


if __name__ == "__main__":
    # chairs
    data = DataBunch(file_path="data", category_of_choice="chairs", batch_size=24, is_camera_used=False)
    learner = Learner(
        None,
        data,
        UnsupervisedPart(),
        UnsupervisedLoss(),
        learning_rate=1e-3,
    )

    learner.fit(
        number_of_steps=130_000,
        evaluation_frequency_steps=13_000,
        visualization_frequency_steps=2000,
    )

    # planes
    data = DataBunch(file_path="data", category_of_choice="planes", image_size=64, batch_size=16, is_camera_used=False)
    learner = Learner(
        None,
        data,
        UnsupervisedPart(image_size=64, voxel_size=32, number_of_point_cloud_points=4000),
        UnsupervisedLoss(),
        learning_rate=1e-4,
    )

    learner.fit(
        number_of_steps=30_000,
        start=0,
        evaluation_frequency_steps=10_000,
        visualization_frequency_steps=1000,
        p=(0.256, 1.0),
        sigma=(2.44, 0.2),
        restore="./planes_unsupervised/models/model_80000.pth"
    )

    # cars
    data = DataBunch(file_path="data", category_of_choice="cars", image_size=64, batch_size=16, is_camera_used=False)
    learner = Learner(
        None,
        data,
        UnsupervisedPart(image_size=64, voxel_size=32, number_of_point_cloud_points=4000),
        UnsupervisedLoss(),
        learning_rate=1e-4,
    )

    learner.fit(
        number_of_steps=50_000,
        start=0,
        evaluation_frequency_steps=10000,
        visualization_frequency_steps=1000,
        p=(0.2095, 1.0),
        sigma=(2.58, 0.2),
        restore="./cars_unsupervised/models/model_60000.pth"
    )
