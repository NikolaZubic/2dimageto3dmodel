# An Effective Loss Function for Generating 3D Models from Single 2D Image without Rendering

### [Papers with code](https://paperswithcode.com/paper/an-effective-loss-function-for-generating-3d) | [Paper](https://arxiv.org/abs/2103.03390)

[Nikola Zubić](https://www.linkedin.com/in/nikola-zubi%C4%87-50458b18b/) &nbsp; [Pietro Lio](https://www.cl.cam.ac.uk/~pl219/) &nbsp;

University of Novi Sad &nbsp; University of Cambridge

[AIAI 2021](http://www.aiai2021.eu/)

![](https://raw.githubusercontent.com/NikolaZubic/2dimageto3dmodel/main/images/cub_birds/birds_dataset_test.png)
![](https://raw.githubusercontent.com/NikolaZubic/2dimageto3dmodel/main/images/pascal_3d/pretrained_weights_p3d.png)

## Citation
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

## License
MIT

## Acknowledgement
This idea has been built based on the architecture of [Insafutdinov & Dosovitskiy](https://github.com/eldar/differentiable-point-clouds).<br>
[Poisson Surface Reconstruction](https://github.com/mmolero/pypoisson) was used for Point Cloud to 3D Mesh transformation.<br>
The GAN architecture (used for texture mapping) is a mixture of [Xian's TextureGAN](https://github.com/janesjanes/Pytorch-TextureGAN) and [Li's GAN](https://arxiv.org/abs/2101.10165).
