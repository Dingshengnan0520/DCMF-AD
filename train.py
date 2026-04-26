import argparse
import os
import csv  
import torch
import torch.nn as nn 
import torch.nn.functional as F
import wandb
import numpy as np
from itertools import chain
from tqdm import tqdm, trange

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big


def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

# ---------- 新增模块：自适应融合与轻量检测头 ----------
class FusionLayer(nn.Module):
    """
    自适应融合：alpha∈(0,1)，输出 = alpha*norm(c2d) + (1-alpha)*norm(c3d)
    这里输入/输出均在 patch 级（B, N）。
    """
    def __init__(self, init_alpha=0.5):
        super().__init__()
        init_logit = np.log(init_alpha) - np.log(1.0 - init_alpha + 1e-8)
        self.logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, cos2d_vec, cos3d_vec):
        # 归一化到 [0,1]，避免量纲差异
        def _norm(x):
            eps = 1e-6
            xmin = torch.amin(x, dim=1, keepdim=True)
            xmax = torch.amax(x, dim=1, keepdim=True)
            return (x - xmin) / (xmax - xmin + eps)
        c2 = _norm(cos2d_vec)
        c3 = _norm(cos3d_vec)
        alpha = torch.sigmoid(self.logit)
        return alpha * c2 + (1 - alpha) * c3  # [B, N]

class JointDetector(nn.Module):
    """
    轻量检测头：对融合后的 patch 向量做一个 1D 1x1 卷积（可学习标度与偏置）。
    输入/输出： [B, N] -> [B, N]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=1, bias=True)

    def forward(self, fused_vec):
        y = self.conv(fused_vec.unsqueeze(1))  # [B,1,N]
        return y.squeeze(1)  # [B,N]
# ---------------------------------------------------

# --------- deep feature contrastive (batch-wise InfoNCE) ----------
def _mean_pool_valid(x, invalid_mask):
    """
    x: [B, N, D]  (N = #patches)
    invalid_mask: [B, N]  (True means invalid)
    return: pooled [B, D]
    """
    B, N, D = x.shape
    valid = ~invalid_mask  # True for valid tokens
    # avoid empty valid set
    valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    masked = x * valid.unsqueeze(-1)  # zero out invalid
    pooled = masked.sum(dim=1) / valid_counts
    return F.normalize(pooled, dim=-1)

def _contrastive_logits(a, b, tau=0.07):
    """
    a, b: [B, D] normalized
    return: logits [B, B] = a @ b^T / tau
    """
    return (a @ b.t()) / tau

def deep_feature_contrastive_loss(pred, target, invalid_mask, tau=0.07):
    """
    pred:   [B, N, D] (mapped features)
    target: [B, N, D] (gt features of the other modality)
    invalid_mask: [B, N] True for invalid tokens (e.g., xyz all zero)
    """
    if pred.size(0) < 2:  # batch=1 时不做对比，返回 0
        return pred.new_zeros(())
    a = _mean_pool_valid(pred, invalid_mask)     # [B, D]
    p = _mean_pool_valid(target, invalid_mask)   # [B, D]
    logits_ap = _contrastive_logits(a, p)        # [B, B]
    logits_pa = _contrastive_logits(p, a)        # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    loss = F.cross_entropy(logits_ap, labels) + F.cross_entropy(logits_pa, labels)
    return loss * 0.5
# ------------------------------------------------------------------


def train_CFM(args):

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    # 直接禁用 WandB，避免登录错误
    wandb.init(mode="disabled")
    print("[wandb] Disabled logging mode, running offline.")

    # Dataloader.
    train_loader = get_data_loader(
        "train",
        class_name=args.class_name,
        img_size=224,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Feature extractors.
    feature_extractor = MultimodalFeatures()

    # Model instantiation.
    CFM_2Dto3D = FeatureProjectionMLP(in_features=768,  out_features=1152)  # 2D -> 3D space
    CFM_3Dto2D = FeatureProjectionMLP(in_features=1152, out_features=768)   # 3D -> 2D space

    # ============ 新增：实例化融合层与检测头（并参与训练） ============
    fusion_layer = FusionLayer(init_alpha=0.5).to(device)
    joint_detector = JointDetector().to(device)
    # =============================================================

    # 多参数组，给融合/检测头更小学习率以保持稳定
    optimizer = torch.optim.Adam([
        {'params': chain(CFM_2Dto3D.parameters(), CFM_3Dto2D.parameters()), 'lr': 1e-4},
        {'params': fusion_layer.parameters(), 'lr': 5e-5},
        {'params': joint_detector.parameters(), 'lr': 5e-5},
    ])

    CFM_2Dto3D.to(device)
    CFM_3Dto2D.to(device)

    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # 对比损失温度与权重（温和叠加，保持原训练主导）
    tau = 0.07
    contrast_w = 0.10
    # 新增：融合/检测头的弱监督自一致损失权重（极小，避免偏离原训练）
    aux_w = 0.01

    # ================== CSV 日志准备（保持你的原列名不变） ==================
    save_dir = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(save_dir, exist_ok=True)
    step_csv = os.path.join(save_dir, f'training_steps_{model_name}.csv')
    epoch_csv = os.path.join(save_dir, f'training_epochs_{model_name}.csv')

    if not os.path.exists(step_csv):
        with open(step_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'step',
                        'loss_3Dto2D', 'loss_2Dto3D', 'contrast_loss', 'total_loss',
                        'cos_sim_3Dto2D', 'cos_sim_2Dto3D'])
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch',
                        'mean_loss_3Dto2D', 'mean_loss_2Dto3D', 'mean_contrast_loss', 'mean_total_loss',
                        'mean_cos_sim_3Dto2D', 'mean_cos_sim_2Dto3D'])
    # ======================================================================

    for epoch in trange(args.epochs_no, desc='Training Feature Transfer Net.'):

        epoch_cos_sim_3Dto2D, epoch_cos_sim_2Dto3D = [], []
        epoch_loss_3Dto2D, epoch_loss_2Dto3D, epoch_loss_contrast, epoch_loss_total = [], [], [], []

        step_idx = 0

        # ------------ [Trainig Loop] ------------ #
        # * Return (rgb_img, organized_pc, depth_map_3channel), globl_label
        for (rgb, pc, _), _ in tqdm(train_loader, desc=f'Extracting feature from class: {args.class_name}.'):
            step_idx += 1
            rgb, pc = rgb.to(device), pc.to(device)

            # Make CFMs trainable.
            CFM_2Dto3D.train()
            CFM_3Dto2D.train()
            fusion_layer.train()
            joint_detector.train()

            # 保持你原来的批处理策略
            if args.batch_size == 1:
                rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
            else:
                rgb_patches, xyz_patches = [], []
                for i in range(rgb.shape[0]):
                    rp, xp = feature_extractor.get_features_maps(
                        rgb[i].unsqueeze(0), pc[i].unsqueeze(0)
                    )
                    rgb_patches.append(rp)
                    xyz_patches.append(xp)
                rgb_patch = torch.stack(rgb_patches, dim=0)
                xyz_patch = torch.stack(xyz_patches, dim=0)

            # 预测
            rgb_feat_pred = CFM_3Dto2D(xyz_patch)  # 3D->2D 预测
            xyz_feat_pred = CFM_2Dto3D(rgb_patch)  # 2D->3D 预测

            # 损失
            xyz_mask = (xyz_patch.sum(axis=-1) == 0)  
            loss_2Dto3D = 1 - metric(xyz_feat_pred[~xyz_mask], xyz_patch[~xyz_mask]).mean()
            loss_3Dto2D = 1 - metric(rgb_feat_pred[~xyz_mask], rgb_patch[~xyz_mask]).mean()

            cos_sim_3Dto2D, cos_sim_2Dto3D = 1 - loss_3Dto2D.detach().cpu(), 1 - loss_2Dto3D.detach().cpu()
            epoch_cos_sim_3Dto2D.append(cos_sim_3Dto2D)
            epoch_cos_sim_2Dto3D.append(cos_sim_2Dto3D)

            # -------- 深层特征对比 ----------
            loss_contrast_3d2d = deep_feature_contrastive_loss(
                pred=rgb_feat_pred, target=rgb_patch, invalid_mask=xyz_mask, tau=tau
            )
            loss_contrast_2d3d = deep_feature_contrastive_loss(
                pred=xyz_feat_pred, target=xyz_patch, invalid_mask=xyz_mask, tau=tau
            )
            loss_contrast = (loss_contrast_3d2d + loss_contrast_2d3d) * 0.5
            # -------------------------------------------------

            def _cos_from_pair(p1, p2):  # p1/p2: [B,N,D] -> [B,N]
                a = F.normalize(p1, dim=-1)
                b = F.normalize(p2, dim=-1)
                d = (a - b).pow(2).sum(-1).sqrt()
                return d

            cos_3d_vec = _cos_from_pair(xyz_feat_pred, xyz_patch)   # [B,N]
            cos_2d_vec = _cos_from_pair(rgb_feat_pred, rgb_patch)   # [B,N]
            # mask 无效 token
            cos_3d_vec = cos_3d_vec.masked_fill(xyz_mask, 0.0)
            cos_2d_vec = cos_2d_vec.masked_fill(xyz_mask, 0.0)

            # 自适应融合 + 轻量检测头
            fused_vec = fusion_layer(cos_2d_vec, cos_3d_vec)        # [B,N]
            det_vec = joint_detector(fused_vec)                     # [B,N]
            comb_target = (cos_2d_vec * cos_3d_vec).detach()        # [B,N]
            valid = ~xyz_mask
            if valid.any():
                aux_loss = F.mse_loss(det_vec[valid], comb_target[valid])
            else:
                aux_loss = det_vec.new_zeros(())

            # -------------------------------------------------------

            if any([
                torch.isnan(loss_3Dto2D), torch.isinf(loss_3Dto2D),
                torch.isnan(loss_2Dto3D), torch.isinf(loss_2Dto3D),
                torch.isnan(loss_contrast), torch.isinf(loss_contrast),
                torch.isnan(aux_loss), torch.isinf(aux_loss)
            ]):
                print("NaN or Inf detected, exiting.")
                exit()

            total_loss = loss_3Dto2D + loss_2Dto3D + contrast_w * loss_contrast + aux_w * aux_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ====== 逐 step 记录到 CSV（列名保持不变） ======
            with open(step_csv, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    epoch + 1, step_idx,
                    float(loss_3Dto2D.detach().cpu().item()),
                    float(loss_2Dto3D.detach().cpu().item()),
                    float(loss_contrast.detach().cpu().item()),
                    float(total_loss.detach().cpu().item()),
                    float(cos_sim_3Dto2D.item()),
                    float(cos_sim_2Dto3D.item())
                ])
            # 累计到 epoch 统计
            epoch_loss_3Dto2D.append(loss_3Dto2D.detach().cpu())
            epoch_loss_2Dto3D.append(loss_2Dto3D.detach().cpu())
            epoch_loss_contrast.append(loss_contrast.detach().cpu())
            epoch_loss_total.append(total_loss.detach().cpu())
            # ====================================

        # Global epoch log（控制台输出）
        mean_3Dto2D = torch.stack(epoch_cos_sim_3Dto2D).mean().item()
        mean_2Dto3D = torch.stack(epoch_cos_sim_2Dto3D).mean().item()
        print(f"[Epoch {epoch+1}] Mean CosSim 3D→2D: {mean_3Dto2D:.4f} | 2D→3D: {mean_2Dto3D:.4f}")

        # ====== 逐 epoch 统计写入 CSV（列名保持不变） ======
        with open(epoch_csv, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                epoch + 1,
                float(torch.stack(epoch_loss_3Dto2D).mean().item()),
                float(torch.stack(epoch_loss_2Dto3D).mean().item()),
                float(torch.stack(epoch_loss_contrast).mean().item()),
                float(torch.stack(epoch_loss_total).mean().item()),
                float(mean_3Dto2D),
                float(mean_2Dto3D),
            ])
        # =======================================

    # ---------------- Save Models ----------------
    directory = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(directory, exist_ok=True)

    torch.save(CFM_2Dto3D.state_dict(), os.path.join(directory, f'CFM_2Dto3D_{model_name}.pth'))
    torch.save(CFM_3Dto2D.state_dict(), os.path.join(directory, f'CFM_3Dto2D_{model_name}.pth'))
    # 新增：保存融合层与检测头参数
    torch.save(fusion_layer.state_dict(), os.path.join(directory, f'FusionLayer_{model_name}.pth'))
    torch.save(joint_detector.state_dict(), os.path.join(directory, f'JointDet_{model_name}.pth'))

    print(f"[✓] Training complete. Models saved to: {directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Crossmodal Feature Networks (CFMs) on a dataset.')

    parser.add_argument('--dataset_path', default='F:/数据集/mvtec_3d_anomaly_detection', type=str,
                        help='Dataset path.')

    parser.add_argument('--checkpoint_savepath', default='F:/论文/SCI代码/checkpoints_CFM_mvtec-20251017T151242Z-1-001/checkpoints_CFM_mvtec', type=str,
                        help='Where to save the model checkpoints.')

    parser.add_argument('--class_name', default="foam", type=str, choices=[
        "bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
        'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle',
        'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'
    ], help='Category name.')

    parser.add_argument('--epochs_no', default=50, type=int,
                        help='Number of epochs to train the CFMs.')

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch dimension. Usually 16 is around the max.')

    args = parser.parse_args()
    train_CFM(args)
