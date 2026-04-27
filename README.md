# DCMF-AD
## Environment
The code was tested with the following environment:
```bash
Python 3.9
CUDA 11.8
PyTorch 2.5.1+cu118
Torchvision 0.20.1+cu118

Dataset
This project uses the MVTec 3D-AD dataset.
The dataset can be downloaded from the official MVTec website:
https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad
After downloading, organize the dataset as follows:
datasets/
└── mvtec_3d_ad/
    ├── bagel/
    ├── cable_gland/
    ├── carrot/
    ├── cookie/
    ├── dowel/
    ├── foam/
    ├── peach/
    ├── potato/
    ├── rope/
    └── tire/

Running
Training
python train.py --dataset_path datasets/mvtec_3d_ad --category bagel
Testing
python test.py --dataset_path datasets/mvtec_3d_ad --category bagel

To run other categories, replace bagel with one of the following:

bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire
