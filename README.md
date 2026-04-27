# DCMF-AD

Official implementation of **DCMF-AD: Unsupervised Industrial Surface Anomaly Detection via Cross-Modal Contrastive Learning and Adaptive Feature Fusion**.

DCMF-AD is an unsupervised multimodal industrial anomaly detection framework. It uses RGB images and 3D point-cloud/depth data for industrial surface defect detection.

The framework mainly contains:

- Bidirectional 2D↔3D cross-modal feature mapping
- Deep feature contrastive learning (DFCL)
- Learnable adaptive fusion layer (LAFL)
- Lightweight local detection head (LDH)

# 1. Environment

The code was tested with the following environment:

```bash
Python 3.9
CUDA 11.8
PyTorch 2.5.1+cu118
Torchvision 0.20.1+cu118

Install dependencies:

pip install -r requirements.txt

A recommended requirements.txt is:

torch==2.5.1+cu118
torchvision==0.20.1+cu118
numpy
opencv-python
scikit-learn
scipy
matplotlib
tqdm
Pillow
wandb

If PyTorch cannot be installed directly through pip, please install the CUDA version from the official PyTorch website.

2. Dataset

This project uses the MVTec 3D-AD dataset.

Dataset download link:
https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad
After downloading, organize the dataset as follows:
datasets/
└── mvtec_3d_anomaly_detection/
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

Each category should contain the official training and testing folders from MVTec 3D-AD.

Supported categories:

bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire
3. Project Structure

Recommended file structure:

DCMF-AD/
├── train.py
├── test.py
├── models/
│   ├── dataset.py
│   ├── features.py
│   ├── feature_transfer_nets.py
│   ├── full_models.py
│   ├── pointnet2_utils.py
│   └── ...
├── utils/
│   ├── general_utils.py
│   ├── metrics_utils.py
│   └── ...
├── results/
├── checkpoints_CFM_mvtec/
├── requirements.txt
└── README.md
4. Training

Run the following command to train the model on one category:

python train.py \
  --dataset_path ./datasets/mvtec_3d_anomaly_detection \
  --checkpoint_savepath ./checkpoints_CFM_mvtec \
  --class_name foam \
  --epochs_no 50 \
  --batch_size 4

Example for another category:

python train.py \
  --dataset_path ./datasets/mvtec_3d_anomaly_detection \
  --checkpoint_savepath ./checkpoints_CFM_mvtec \
  --class_name bagel \
  --epochs_no 50 \
  --batch_size 4

During training, the model progressively optimizes:

2D→3D feature mapping
3D→2D feature mapping
Deep feature contrastive loss
Adaptive fusion layer
Lightweight detection head

The training checkpoints will be saved to:

checkpoints_CFM_mvtec/
└── foam/
    ├── CFM_2Dto3D_foam_50ep_4bs.pth
    ├── CFM_3Dto2D_foam_50ep_4bs.pth
    ├── FusionLayer_foam_50ep_4bs.pth
    ├── JointDet_foam_50ep_4bs.pth
    ├── training_steps_foam_50ep_4bs.csv
    └── training_epochs_foam_50ep_4bs.csv
5. Testing

After training, run inference with:

python test.py \
  --dataset_path ./datasets/mvtec_3d_anomaly_detection \
  --checkpoint_folder ./checkpoints_CFM_mvtec \
  --class_name foam \
  --epochs_no 50 \
  --batch_size 4

The testing script loads the trained CFM modules, FusionLayer, and JointDet. It then generates anomaly maps and computes evaluation metrics.

6. Output Results
Quantitative results

The quantitative results are saved in:

results/quantitatives_mvtec/

Example output file:

results/quantitatives_mvtec/foam_50ep_4bs.md

The reported metrics include:

AUPRO@30%
AUPRO@10%
AUPRO@5%
AUPRO@1%
P-AUROC
I-AUROC
Qualitative results

If qualitative visualization is enabled, results are saved in:

results/qualitatives_mvtec/

The visualization includes:

RGB image
Ground-truth mask
Depth map
2D cosine anomaly response
3D cosine anomaly response
Fused anomaly map
Final detection result
7. Important Notes
Make sure the dataset path is correct.
If the following error occurs:
ValueError: num_samples should be a positive integer value, but got num_samples=0

it usually means that the dataset path or folder structure is incorrect.

The checkpoint path used for testing must be the same as the path used during training.
The class_name, epochs_no, and batch_size used during testing should match the training configuration.

For example, if training uses:

--class_name foam --epochs_no 50 --batch_size 4

then testing should also use:

--class_name foam --epochs_no 50 --batch_size 4

Otherwise, the checkpoint file names may not match.

8. Citation

If this repository is useful for your research, please cite our paper:

@article{dcmfad2026,
  title={Unsupervised Industrial Surface Anomaly Detection via Cross-Modal Contrastive Learning and Adaptive Feature Fusion},
  author={Ding, Shengnan and Liu, Fuguo and Shi, Yufeng},
  journal={},
  year={2026}
}
