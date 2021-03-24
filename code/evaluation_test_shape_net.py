from data.shape_net_dataset import DataBunch
from models.unsupervised_part import UnsupervisedPart
from training_test_shape_net import parameter_adjustment
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T
import pandas as pd


def training_pyplot(projections, poses):
    plt.figure(figsize=(10, 10))

    for i in range(5):
        plt.subplot(5, 5, i * 5 + 1)

        if i == 0:
            plt.title("Pose", **{'fontname': 'DejaVu Serif', 'size': 12})

        plt.imshow(T.to_pil_image(poses[i]))
        plt.axis(False)

        for j in range(4):
            plt.subplot(5, 5, i * 5 + j + 2)

            if i == 0:
                plt.title(f"Canditate {j+1}", **{'fontname': 'DejaVu Serif', 'size': 12})

            plt.imshow(T.to_pil_image(projections[i * 4 + j]), cmap='gray')
            plt.axis(False)

    plt.show()


def evaluation_pyplot(projections, poses, masks):
    plt.figure(figsize=(6, 10))

    for i in range(5):
        plt.subplot(5, 3, i * 3 + 1)

        if i == 0:
            plt.title("Image", **{'fontname': 'DejaVu Serif', 'size': 12})

        plt.imshow(T.to_pil_image(poses[i]))
        plt.axis(False)

        plt.subplot(5, 3, i * 3 + 2)

        if i == 0:
            plt.title("Mask", **{'fontname': 'DejaVu Serif', 'size': 12})

        plt.imshow(T.to_pil_image(masks.detach()[i]), cmap='gray')
        plt.axis(False)

        plt.subplot(5, 3, i * 3 + 3)

        if i == 0:
            plt.title("Student Projection", **{'fontname': 'DejaVu Serif', 'size': 12})

        plt.imshow(projections[i].detach(), cmap='gray')
        plt.axis(False)

    plt.show()


if __name__ == "__main__":
    # evaluation
    data = DataBunch(file_path="data", batch_size=2, is_camera_used=False)
    model = UnsupervisedPart()

    checkpoint = torch.load("chairs_unsupervised/models/model_91000.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    parameter_adjustment(model, 91 / 130)

    images, poses, masks = data.train_ds[1]

    model.eval()
    projection, student = model(images[4].unsqueeze(0), poses)
    projection = projection.cpu().detach()

    evaluation_pyplot(projections=projection, poses=poses, masks=masks)

    # train
    model.train()
    proj, ensemble, student = model(images[4].unsqueeze(0), poses)
    proj = proj.detach()

    training_pyplot(projections=projection, poses=poses)

    full_loss = pd.read_csv("chairs_unsupervised/train_full_loss.csv")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(full_loss.Step, full_loss.Value, c="blue", alpha=0.4)
    plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue")
    plt.ylabel("Unsupervised loss", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xlabel("Step", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.yticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.legend(prop={"size": 12})
    plt.show()

    #################################################################################################################
    data = DataBunch(file_path="data", category_of_choice="planes", image_size=64, batch_size=2, is_camera_used=False)
    model = UnsupervisedPart(image_size=64, voxel_size=32, number_of_point_cloud_points=2000)

    checkpoint = torch.load("planes_unsupervised/models/model_30000.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.point_cloud_drop_out.p = 1.0

    images, poses, masks = data.train_ds[1002]

    model.eval()
    projections, student = model(images[3].unsqueeze(0), poses)

    evaluation_pyplot(projections=projection, poses=poses, masks=masks)

    model.train()
    projections, ensemble, student = model(images[3].unsqueeze(0), poses)

    training_pyplot(projections=projection, poses=poses)

    full_loss = pd.read_csv("planes_unsupervised/full_loss.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(full_loss.Step, full_loss.Value, alpha=0.4, c="blue")
    plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue")
    plt.ylabel("Unsupervised loss", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xlabel("Step", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.yticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.legend(prop={"size": 12})
    plt.show()

    #################################################################################################################
    data = DataBunch(file_path="data", category_of_choice="cars", image_size=64, batch_size=2, is_camera_used=False)
    model = UnsupervisedPart(image_size=64, voxel_size=32, number_of_point_cloud_points=2000)

    checkpoint = torch.load("cars_unsupervised/models/model_30000.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.point_cloud_drop_out.p = 1.0

    images, poses, masks = data.train_ds[1002]

    model.eval()
    projections, student = model(images[3].unsqueeze(0), poses)

    evaluation_pyplot(projections=projection, poses=poses, masks=masks)

    model.train()
    projections, ensemble, student = model(images[3].unsqueeze(0), poses)

    training_pyplot(projections=projection, poses=poses)

    full_loss = pd.read_csv("cars_unsupervised/full_loss.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(full_loss.Step, full_loss.Value, alpha=0.4, c="blue")
    plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue")
    plt.ylabel("Unsupervised loss", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xlabel("Step", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.yticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.legend(prop={"size": 12})
    plt.show()

    #################################################################################################################
    # full loss
    full_loss = pd.read_csv("full_loss.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(full_loss.Step, full_loss.Value, alpha=0.4, c="blue")
    plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue")
    plt.ylabel("Unsupervised loss", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xlabel("Step", **{'fontname': 'DejaVu Serif', 'size': 12})
    plt.xticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.yticks(**{'fontname': 'DejaVu Serif', 'size': 12})
    plt.legend(prop={"size": 12})
    plt.show()
