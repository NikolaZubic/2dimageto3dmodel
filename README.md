# An Effective Loss Function for Generating 3D Models from Single 2D Image without Rendering

### [Papers with code](https://paperswithcode.com/paper/an-effective-loss-function-for-generating-3d) | [Paper](https://arxiv.org/abs/2103.03390)

[Nikola Zubić](https://www.linkedin.com/in/nikola-zubi%C4%87-50458b18b/) &nbsp; [Pietro Lio](https://www.cl.cam.ac.uk/~pl219/) &nbsp;

University of Novi Sad &nbsp; University of Cambridge

[AIAI 2021](http://www.aiai2021.eu/)

![](https://raw.githubusercontent.com/NikolaZubic/2dimageto3dmodel/main/images/cub_birds/birds_dataset_test.png)
![](https://raw.githubusercontent.com/NikolaZubic/2dimageto3dmodel/main/images/pascal_3d/pretrained_weights_p3d.png)

## Citation
Besides AIAI 2021, our paper is in a Springer's book entitled "Artificial Intelligence Applications and Innovations": [link](https://link.springer.com/chapter/10.1007%2F978-3-030-79150-6_25)
<br><br>
Please, cite our paper if you find this code useful for your research.
```
@article{zubic2021effective,
  title={An Effective Loss Function for Generating 3D Models from Single 2D Image without Rendering},
  author={Zubi{\'c}, Nikola and Li{\`o}, Pietro},
  journal={arXiv preprint arXiv:2103.03390},
  year={2021}
}
```

## Prerequisites
- Download code:<br>
  Git clone the code with the following command:
  ```
  git clone https://github.com/NikolaZubic/2dimageto3dmodel.git
  ```
- Open the project with Conda Environment (Python 3.7)

- Install packages:
  ```
  conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
  ```
  Then git clone Kaolin library in the root (2dimageto3dmodel) folder with the following [commit](https://github.com/NVIDIAGameWorks/kaolin/tree/e7e513173bd4159ae45be6b3e156a3ad156a3eb9) and run the following commands:
  ```
  cd kaolin
  git checkout e7e513173b
  python setup.py install
  pip install --no-dependencies nuscenes-devkit opencv-python-headless scikit-learn joblib pyquaternion cachetools
  pip install packaging
  ```
## Run the program
Run the following commands from the root/code/ (2dimageto3dmodel/code/) directory:<br>
```
python main.py --dataset cub --batch_size 16 --weights pretrained_weights_cub --save_results
```
for the CUB Birds Dataset.
<br><br>
```
python main.py --dataset p3d --batch_size 16 --weights pretrained_weights_p3d --save_results
```
for the Pascal 3D+ Dataset.<br><br>

The results will be saved at `2dimageto3dmodel/code/results/` path.

## Continue training
To continue the training process:<br>
Run the following commands (without ```--save_results```) from the root/code/ (2dimageto3dmodel/code/) directory:<br>
```
python main.py --dataset cub --batch_size 16 --weights pretrained_weights_cub
```
for the CUB Birds Dataset.
<br><br>
```
python main.py --dataset p3d --batch_size 16 --weights pretrained_weights_p3d
```
for the Pascal 3D+ Dataset.<br><br>

## Generation of Pseudo-ground-truths
In these reconstruction steps, we need a trained mesh estimation model. We can use the pre-trained model (already provided) or train it from scratch. The Pseudo-ground-truth data for CUB birds is generated in the following way:
```
python run_reconstruction.py --name pretrained_reconstruction_cub --dataset cub --batch_size 10 --generate_pseudogt
```
For Pascal 3D+ dataset:
```
python run_reconstruction.py --name pretrained_reconstruction_p3d --dataset p3d --optimize_z0 --batch_size 10 --generate_pseudogt
```
Through this, we replace a cache directory, which contains pre-computed statistics for the evaluation of Frechet Inception Distances, poses and images metadata, and the Pseudo-ground-truths for each image.

## Mesh generator training from scratch
Set up the Pseudo-ground-truth data as described in the section above, then execute the following command:
```
python main.py --name cub_512x512_class --conditional_class --dataset cub --gpu_ids 0,1,2,3 --batch_size 32 --epochs 1000 --tensorboard
```
Here, we train a CUB birds model, conditioned on class labels, for 1000 epochs. Every 20 epochs, we have FID evaluations (which can be changed with `--evaluate_freq`). Usage of different numbers of GPUs can produce slightly different results. Tensorboard allows us to export the results in Tensorboard's log directory `tensorboard_gan`.

After training, we can find the best model's checkpoint with the following command:
```
python main.py --name cub_512x512_class --conditional_class --dataset cub --gpu_ids 0,1,2,3 --batch_size 64 --evaluate --which_epoch best
```

## Mesh estimation model training
Use the following two commands for training from scratch:
```
python run_reconstruction.py --name pretrained_reconstruction_cub --dataset cub --batch_size 50 --tensorboard
python run_reconstruction.py --name pretrained_reconstruction_p3d --dataset p3d --optimize_z0 --batch_size 50 --tensorboard
```
Tensorboard log files are saved in `tensorboard_recon`.

## License
MIT

## Acknowledgment
This idea has been built based on the architecture of [Insafutdinov & Dosovitskiy](https://github.com/eldar/differentiable-point-clouds).<br>
[Poisson Surface Reconstruction](https://github.com/mmolero/pypoisson) was used for Point Cloud to 3D Mesh transformation.<br>
The GAN architecture (used for texture mapping) is a mixture of [Xian's TextureGAN](https://github.com/janesjanes/Pytorch-TextureGAN) and [Li's GAN](https://arxiv.org/abs/2101.10165).
