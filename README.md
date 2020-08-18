# Learning Gradient Fields for Shape Generation

This repository contains a PyTorch implementation of the paper:

[*Learning Gradient Fields for Shape Generation*](http://www.cs.cornell.edu/~ruojin/ShapeGF/)
[[Project page]](http://www.cs.cornell.edu/~ruojin/ShapeGF/)
[[Arxiv]](https://arxiv.org/abs/2008.06520)
[[Short-video]](https://www.youtube.com/watch?v=HQTbtFzDYAU)
[[Long-video]](https://www.youtube.com/watch?v=xCCdnzt7NPA)

[Ruojin Cai*](http://www.cs.cornell.edu/~ruojin/), 
[Guandao Yang*](https://www.guandaoyang.com/), 
[Hadar Averbuch-Elor](http://www.cs.cornell.edu/~hadarelor/), 
[Zekun Hao](http://www.cs.cornell.edu/~zekun/), 
[Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/), 
[Noah Snavely](http://www.cs.cornell.edu/~snavely/), 
[Bharath Hariharan](http://home.bharathh.info/)
_(* Equal contribution)_

ECCV 2020 (*Spotlight*)

<p float="left">
    <img src="assets/ShapeGF.gif" height="256"/>
</p>

## Introduction
In this work, we propose a novel technique to generate shapes from point cloud data. A point cloud can be viewed as samples from a distribution of 3D points whose density is concentrated near the surface of the shape. Point cloud generation thus amounts to moving randomly sampled points to high-density areas. We generate point clouds by performing stochastic gradient ascent on an unnormalized probability density, thereby moving sampled points toward the high-likelihood regions. Our model directly predicts the gradient of the log density field and can be trained with a simple objective adapted from score-based generative models. We show that our method can reach state-of-the-art performance for point cloud auto-encoding and generation, while also allowing for extraction of a high-quality implicit surface.

## Dependencies
```bash
# Create conda environment with torch 1.2.0 and CUDA 10.0
conda env create -f environment.yml
conda activate ShapeGF

# Compile the evaluation metrics
cd evaluation/pytorch_structural_losses/
make clean
make all
```

## Dataset

Please follow the instruction from PointFlow to set-up the dataset: [link](https://github.com/stevenygd/PointFlow). 

## Pretrained Model 

Pretrained model will be available in the following google drive: [link](https://drive.google.com/drive/folders/1VBtAKSQBKaKoOeTzORbrWcnPoBk9Wl4-?usp=sharing).
To use the pretrained models, download the `pretrained` folder and put it under the project root directory.

#### Testing the pretrained auto-encoding model:
The following commands test the performance of the pre-trained models in the point cloud auto-encoding task.
The command outputs the CD and EMD on the test/validation sets.
```bash
# Usage:
# python test.py <config> --pretrained <checkpoint_filename>

python test.py configs/recon/airplane/airplane_recon_add.yaml \
    --pretrained pretrained/recon/airplane_recon_add.pt
python test.py configs/recon/car/car_recon_add.yaml \
    --pretrained pretrained/recon/car_recon_add.pt
python test.py configs/recon/chair/chair_recon_add.yaml \
    --pretrained pretrained/recon/chair_recon_add.pt
```

The pretrained model's auto-encoding performance is as follows:
| Dataset  | Metrics  | Ours  | Oracle |
|----------|----------|-------|--------|
| Airplane | CD x1e4  | 0.966 |  0.837 |
|          | EMD x1e2 | 2.632 |  2.062 |
| Chair    | CD x1e4  | 5.660 |  3.201 |
|          | EMD x1e2 | 4.976 |  3.297 |
| Car      | CD x1e4  | 5.306 |  3.904 |
|          | EMD x1e2 | 4.380 |  3.251 |

#### Testing the pretrained generation model:
The following commands test the performance of the pre-trained models in the point cloud generation task.
The command outputs the JSD, MMD-(CD/EMD), COV-(CD/EMD), and 1NN-(CD/EMD).

```bash
# Usage:
# python test.py <config> --pretrained <checkpoint_filename>

python test.py configs/gen/airplane_gen_add.yaml \
    --pretrained pretrained/gen/airplane_gen_add.pt
python test.py configs/gen/car/car_gen_add.yaml \
    --pretrained pretrained/gen/car_gen_add.pt
python test.py configs/gen/chair/chair_gen_add.yaml \
    --pretrained pretrained/gen/chair_gen_add.pt
```


## Training
#### Single GPU Training
```bash
# Usage:
python train.py <config>
```

#### Multi GPU Training

Our code also provides single-node multi GPU training using pytorch's Distributed Data Parallel.
The script will run on all GPUs visible to the function.
The usage and examples are as follows:
```bash
# Usage
python train_multi_gpus.py <config> 

# To specify the total batch size, use --batch_size
python train_multi_gpus.py <config> --batch_size <#gpu x batch_size/GPU>
```

#### Stage-1: Auto-encoding
In this stage, we create a conditional generator that models the distribution of 3D points conditioned on the latent vector.
The commands used to train our auto-encoding model for a single-shape, single ShapeNet category, and the whole ShapeNet are:
```bash
# Single shape
python train.py configs/recon/single_shapes/dress.yaml  # the dress in the teaser
python train.py configs/recon/single_shapes/torus.yaml  # the torus in the teaser

# Single category
python train.py configs/recon/airplane/airplane_recon_add.yaml  # airplane
python train.py configs/recon/airplane/chair_recon_add.yaml     # chair
python train.py configs/recon/airplane/car_recon_add.yaml       # car 

# Whole shape-net
python train_multi_gpus.py configs/recon/shapenet/shapenet_recon.yaml  # ShapeNet
```

#### Stage-2: Generation
In the second stage, we train a l-GAN to model the distribution of shapes - which are captured by the latent vector of the auto-encoder described in the first stage.
The commands used to train l-GAN for a single ShapeNet category using the default pretrained model (in the `<root>/pretrained` directory) are:
```bash
python train.py configs/gen/airplane_gen_add.yaml  # airplane
python train.py configs/gen/chair_gen_add.yaml     # chair
python train.py configs/gen/car_gen_add.yaml       # car 
``` 

# Cite 
Please cite our work if you find it useful: 
```bibtex
@inproceedings{ShapeGF,
 title={Learning Gradient Fields for Shape Generation},
 author={Cai, Ruojin and Yang, Guandao and Averbuch-Elor, Hadar and Hao, Zekun and Belongie, Serge and Snavely, Noah and Hariharan, Bharath},
 booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
 year={2020}
}
```
#### Acknowledgment
This work was supported in part by grants from Magic Leap and Facebook AI, and the Zuckerman STEM leadership program.

