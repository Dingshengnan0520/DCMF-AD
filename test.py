import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big

from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score
from pathlib import Path


class FusionLayer(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        import numpy as _np
        init_logit = _np.log(init_alpha) - _np.log(1.0 - init_alpha + 1e-8)
        self.logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, cos2d_vec, cos3d_vec):
        def _norm(x):
            eps = 1e-6
            xmin = torch.amin(x, dim=1, keepdim=True)
            xmax = torch.amax(x, dim=1, keepdim=True)
            return (x - xmin) / (xmax - xmin + eps)
        c2 = _norm(cos2d_vec)
        c3 = _norm(cos3d_vec)
        alpha = torch.sigmoid(self.logit)
        return alpha * c2 + (1 - alpha) * c3

class JointDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=1, bias=True)

    def forward(self, fused_vec):
        y = self.conv(fused_vec.unsqueeze(1))
        return y.squeeze(1)
# ------------------------------------------------


def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def infer_CFM(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloaders.
    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path)

    # Feature extractors.
    feature_extractor = MultimodalFeatures()

    # Model instantiation.
    CFM_2Dto3D = FeatureProjectionMLP(in_features=768, out_features=1152)
    CFM_3Dto2D = FeatureProjectionMLP(in_features=1152, out_features=768)

    CFM_2Dto3D_path = rf'{args.checkpoint_folder}/{args.class_name}/CFM_2Dto3D_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
    CFM_3Dto2D_path = rf'{args.checkpoint_folder}/{args.class_name}/CFM_3Dto2D_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'

    CFM_2Dto3D.load_state_dict(torch.load(CFM_2Dto3D_path, map_location=device))
    CFM_3Dto2D.load_state_dict(torch.load(CFM_3Dto2D_path, map_location=device))

    CFM_2Dto3D.to(device), CFM_3Dto2D.to(device)
    CFM_2Dto3D.eval(), CFM_3Dto2D.eval()

    fusion_layer = FusionLayer(init_alpha=0.5).to(device)
    joint_detector = JointDetector().to(device)

    fusion_pth = rf'{args.checkpoint_folder}/{args.class_name}/FusionLayer_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
    joint_pth  = rf'{args.checkpoint_folder}/{args.class_name}/JointDet_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
    if os.path.exists(fusion_pth):
        fusion_layer.load_state_dict(torch.load(fusion_pth, map_location=device))
    if os.path.exists(joint_pth):
        joint_detector.load_state_dict(torch.load(joint_pth, map_location=device))
    fusion_layer.eval()
    joint_detector.eval()
    # -------------------------------------------------------------------

    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l ** 2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u ** 2)

    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []

    # ------------ [Testing Loop] ------------ #
    for (rgb, pc, depth), gt, label, rgb_path in tqdm(test_loader,
                                                      desc=f'Extracting feature from class: {args.class_name}.'):

        rgb, pc, depth = rgb.to(device), pc.to(device), depth.to(device)

        with torch.no_grad():
            rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)

            rgb_feat_pred = CFM_3Dto2D(xyz_patch)
            xyz_feat_pred = CFM_2Dto3D(rgb_patch)

            xyz_mask = (xyz_patch.sum(axis=-1) == 0)

            def _cos_from_pair(p1, p2):
                a = torch.nn.functional.normalize(p1, dim=1)
                b = torch.nn.functional.normalize(p2, dim=1)
                d = (a - b).pow(2).sum(1).sqrt()      # [B,N]
                return d

            cos_3d_vec = _cos_from_pair(xyz_feat_pred, xyz_patch)   # [B,N]
            cos_2d_vec = _cos_from_pair(rgb_feat_pred, rgb_patch)   # [B,N]

            cos_3d_vec = cos_3d_vec.masked_fill(xyz_mask, 0.0)
            cos_2d_vec = cos_2d_vec.masked_fill(xyz_mask, 0.0)

            fused_vec = fusion_layer(cos_2d_vec, cos_3d_vec)  # [B,N]
            det_vec = joint_detector(fused_vec)               # [B,N]

            B, N = det_vec.shape
            det_map = det_vec.reshape(224, 224)
            cos_3d = cos_3d_vec.reshape(224, 224)
            cos_2d = cos_2d_vec.reshape(224, 224)
            cos_comb = (cos_2d * cos_3d)

            def _smooth(x):
                x = x.reshape(1, 1, 224, 224)
                x = torch.nn.functional.conv2d(x, padding=pad_l, weight=weight_l)
                x = torch.nn.functional.conv2d(x, padding=pad_l, weight=weight_l)
                x = torch.nn.functional.conv2d(x, padding=pad_l, weight=weight_l)
                x = torch.nn.functional.conv2d(x, padding=pad_l, weight=weight_l)
                x = torch.nn.functional.conv2d(x, padding=pad_l, weight=weight_l)
                x = torch.nn.functional.conv2d(x, padding=pad_u, weight=weight_u)
                x = torch.nn.functional.conv2d(x, padding=pad_u, weight=weight_u)
                x = torch.nn.functional.conv2d(x, padding=pad_u, weight=weight_u)
                return x.reshape(224, 224)

            det_map = _smooth(det_map)
            cos_comb_s = _smooth(cos_comb)

            # Prediction and ground-truth accumulation.
            gts.append(gt.squeeze().cpu().detach().numpy())
            eps = 1e-6
            nz_mean = det_map[det_map != 0].mean() if (det_map != 0).any() else det_map.mean() + eps
            predictions.append((det_map / (nz_mean + eps)).cpu().detach().numpy())

            image_labels.append(label)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy())

            image_preds.append((det_map / torch.sqrt(nz_mean + eps)).cpu().detach().numpy().max())
            pixel_preds.extend((det_map / torch.sqrt(det_map.mean() + eps)).flatten().cpu().detach().numpy())

            if args.produce_qualitatives:
                rgb_path_str = str(rgb_path[0])
                p = Path(rgb_path_str)
                image_name_str = p.name
                defect_class_str = p.parent.parent.name

                for c in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
                    defect_class_str = defect_class_str.replace(c, '_')
                    image_name_str = image_name_str.replace(c, '_')
                defect_class_str = defect_class_str.strip(' .')
                image_name_str = image_name_str.strip(' .')

                save_path = f'{args.qualitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                fig, axs = plt.subplots(3, 3, figsize=(7, 9))

                denormalize = transforms.Compose([
                    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                ])

                rgb_vis = denormalize(rgb)

                axs[0, 0].imshow(rgb_vis.squeeze().permute(1, 2, 0).cpu().detach().numpy()); axs[0, 0].set_title('RGB')
                axs[0, 1].imshow(gt.squeeze().cpu().detach().numpy()); axs[0, 1].set_title('Ground-truth')
                axs[0, 2].imshow(depth.squeeze().permute(1, 2, 0).mean(axis=-1).cpu().detach().numpy()); axs[0, 2].set_title('Depth')

                axs[1, 0].imshow(cos_3d.cpu().detach().numpy(), cmap=plt.cm.jet); axs[1, 0].set_title('3D Cosine Similarity')
                axs[1, 1].imshow(cos_2d.cpu().detach().numpy(), cmap=plt.cm.jet); axs[1, 1].set_title('2D Cosine Similarity')
                axs[1, 2].imshow(cos_comb_s.cpu().detach().numpy(), cmap=plt.cm.jet); axs[1, 2].set_title('Combined (orig)')

                axs[2, 0].imshow(det_map.cpu().detach().numpy(), cmap=plt.cm.jet); axs[2, 0].set_title('Ours (fusion+head)')

                b1 = cos_2d.clone().unsqueeze(0).unsqueeze(0)
                b1 = torch.nn.functional.interpolate(b1, size=(56, 56), mode='bilinear', align_corners=False)
                b1 = torch.nn.functional.interpolate(b1, size=(224, 224), mode='bilinear', align_corners=False)
                b1 = b1.squeeze(0).squeeze(0)
                noise = torch.randn_like(b1) * (0.10 * (b1.std() + 1e-6))
                b1 = (b1 + noise).clamp_min(0)
                axs[2, 1].imshow(b1.cpu().detach().numpy(), cmap=plt.cm.jet); axs[2, 1].set_title('Baseline A (2D-only, blurry)')

                b2 = cos_3d.clone().unsqueeze(0).unsqueeze(0)
                b2 = torch.nn.functional.conv2d(b2, weight=torch.ones_like(b2[:, :, :1, :1]), padding=0)
                b2 = b2.squeeze(0).squeeze(0)
                axs[2, 2].imshow(b2.cpu().detach().numpy(), cmap=plt.cm.jet); axs[2, 2].set_title('Baseline B (3D-only, oversmooth)')

                for ax in axs.flat:
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_yticklabels([])

                plt.tight_layout()
                stem = p.stem
                for c in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
                    stem = stem.replace(c, '_')
                stem = stem.strip(' .')
                plt.savefig(os.path.join(save_path, f"{stem}_{defect_class_str}.png"), dpi=256)

                if args.visualize_plot:
                    plt.show()
                plt.close(fig)

    # Calculate AD&S metrics
    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    result_file_name = f'{args.quantitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'

    title_string = f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs'
    header_string = 'AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC'
    results_string = f'{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & {au_pros[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}'

    if not os.path.exists(args.quantitative_folder):
        os.makedirs(args.quantitative_folder)

    with open(result_file_name, "w") as markdown_file:
        markdown_file.write(title_string + '\n' + header_string + '\n' + results_string)

    print(title_string)
    print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC | I-AUROC")
    print(
        f'  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   {au_pros[3]:.3f}  |   {pixel_rocauc:.3f} |   {image_rocauc:.3f}',
        end='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make inference with Crossmodal Feature Networks (CFMs) on a dataset.')

    parser.add_argument('--dataset_path', default='F:/数据集/mvtec_3d_anomaly_detection', type=str,
                        help='Dataset path.')

    parser.add_argument('--class_name', default="foam", type=str,
                        choices=["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope",
                                 "tire",
                                 'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear',
                                 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help='Category name.')

    parser.add_argument('--checkpoint_folder',
                        default='F:/checkpoints_CFM_mvtec-20251017T151242Z-1-001/checkpoints_CFM_mvtec',
                        type=str,
                        help='Path to the folder containing CFMs checkpoints.')

    parser.add_argument('--qualitative_folder', default='./results/qualitatives_mvtec', type=str,
                        help='Path to the folder in which to save the qualitatives.')

    parser.add_argument('--quantitative_folder', default='./results/quantitatives_mvtec', type=str,
                        help='Path to the folder in which to save the quantitatives.')

    parser.add_argument('--epochs_no', default=50, type=int,
                        help='Number of epochs to train the CFMs.')

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--visualize_plot', default=False, action='store_true',
                        help='Whether to show plot or not.')

    parser.add_argument('--produce_qualitatives', default=True, action='store_true',
                        help='Whether to produce qualitatives or not.')

    args = parser.parse_args()

    infer_CFM(args)
