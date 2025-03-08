import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

########################################
# 1) Import the radar classes you need
########################################
from RaDE_net8 import (
    RadarDepthEstimationModel,
    RadarModel,
    axis_aligned_iou_3d,
    IOU_THRESHOLDS,       # <--- so we can use IOU_THRESHOLDS in training
    compute_iou,          # <--- so we can compute multi-threshold IoU
    log_print
)

# ---------------------------------
# RadarDataset: return global_index
# ---------------------------------
from torch.utils.data import Dataset

class RadarDataset(Dataset):
    def __init__(self, input_file, label_file, 
                 train_indices_file, val_indices_file, 
                 test_indices_file=None, 
                 split="train"):
        try:
            self.inputs = np.load(input_file, allow_pickle=True)
            self.labels = np.load(label_file, allow_pickle=True)
            self.train_indices = np.load(train_indices_file, allow_pickle=True)
            self.val_indices   = np.load(val_indices_file, allow_pickle=True)
            self.split = split
            if test_indices_file is not None:
                self.test_indices = np.load(test_indices_file, allow_pickle=True)
            else:
                self.test_indices = None

        except Exception as e:
            raise IOError(f"Error loading data files: {e}")

    def __len__(self):
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "val":
            return len(self.val_indices)
        elif self.split == "test":
            return len(self.test_indices) if self.test_indices is not None else 0
        else:
            raise ValueError("Invalid split value.")

    def __getitem__(self, idx):
        """
        Return:
          - 'full_points': the radar points/features for this sample
          - 'dynamics': velocity/range/bearing
          - 'bbox_gt': bounding box ground truth
          - 'global_index': the row in the original inputs/labels arrays
        """
        if self.split == "train":
            ID = self.train_indices[idx]
        elif self.split == "val":
            ID = self.val_indices[idx]
        elif self.split == "test":
            ID = self.test_indices[idx]
        else:
            raise ValueError("Invalid split value.")

        # Retrieve raw data
        points_np = self.inputs[ID]
        labels_np = self.labels[ID]

        # Convert to torch
        points = torch.tensor(points_np, dtype=torch.float32)
        labels = torch.tensor(labels_np, dtype=torch.float32)  # shape (1,7)

        # Subfields
        full_points = points
        dynamics = points[:, [3, 4, 5]]  # (velocity, range, bearing)
        bbox_gt = labels[0, :7]         # the 7 bounding-box params

        return {
            'full_points': full_points,
            'dynamics': dynamics,
            'bbox_gt': bbox_gt,
            'global_index': ID
        }

########################################
# 2) Fusion MLP
########################################
class FusionMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super(FusionMLP, self).__init__()
        # input: 7D box + 1D conf from vantage A, plus 7D box + 1D conf from vantage B => 16
        self.mlp = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU()
        )
        self.box_out = nn.Linear(hidden_size, 7)   # fused bounding box
        self.conf_out = nn.Linear(hidden_size, 1)  # fused confidence (optional)

    def forward(self, boxA, confA, boxB, confB):
        # print(f"boxA shape: {boxA.shape}")  # Debugging
        # print(f"confA shape: {confA.shape}")  # Debugging
        # print(f"boxB shape: {boxB.shape}")  # Debugging
        # print(f"confB shape: {confB.shape}")  # Debugging

        # Ensure batch sizes match before concatenation
        assert boxA.shape[0] == confA.shape[0] == boxB.shape[0] == confB.shape[0], "Batch size mismatch!"



        x = torch.cat([boxA, confA, boxB, confB], dim=-1)  # shape: (batch,16)
        h = self.mlp(x)                                   # (batch,hidden_size)
        fused_box = self.box_out(h)                       # (batch,7)
        fused_conf = torch.sigmoid(self.conf_out(h))      # (batch,1)
        return fused_box, fused_conf

########################################
# 3) Transform side boxes to rear coords
########################################
def transform_box_side_to_rear(box_side, offsets):
    # box_side: (batch,7) -> (w,h,l,x,y,z,theta)
    w   = box_side[:,0]
    h   = box_side[:,1]
    l   = box_side[:,2]
    x_s = box_side[:,3]
    y_s = box_side[:,4]
    z_s = box_side[:,5]
    th_s= box_side[:,6]

    dx = offsets[:,0] if offsets.ndim > 1 else offsets[0]
    dy = offsets[:,1]if offsets.ndim > 1 else offsets[1]

    x_r = x_s + dx
    y_r = y_s + dy

    # keep w,h,l,z,theta
    box_rear = torch.stack([w,h,l,x_r,y_r,z_s,th_s], dim=1)
    return box_rear

########################################
# 4) Combined L1 + IoU late-fusion loss
########################################
def late_fusion_loss(pred_box, gt_box, w_bbox=0.7, w_iou=0.3):
    """
    Example: L = w_bbox * SmoothL1 + w_iou * (1 - IoU)
    """
    bbox_loss = F.smooth_l1_loss(pred_box, gt_box)
    ious = axis_aligned_iou_3d(pred_box, gt_box)
    iou_loss_val = 1.0 - ious.mean()

    total = w_bbox * bbox_loss + w_iou * iou_loss_val
    return total, bbox_loss.item(), iou_loss_val.item()

########################################
# 5A) validate_late_fusion
########################################
def validate_late_fusion(
    fusion_mlp,
    rear_model,
    side_model,
    val_rear_loader,
    val_side_loader,
    side_offsets,
    device
):
    """
    Similar to your baseline's validate_model():
    - no gradient updates
    - compute total loss, IoU stats (using multiple thresholds)
    - return average loss + iou dict
    """
    fusion_mlp.eval()
    rear_model.eval()
    side_model.eval()

    total_loss = 0.0
    batch_count = 0

    # We'll track IoU for multiple thresholds
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

    with torch.no_grad():
        for batch_rear, batch_side in zip(val_rear_loader, val_side_loader):
            # 1) Data
            full_pts_rear = batch_rear['full_points'].to(device)
            dyn_rear      = batch_rear['dynamics'].to(device)
            bbox_gt_rear  = batch_rear['bbox_gt'].to(device)
            idx_side      = batch_side['global_index']  # shape=(batch_size,)

            full_pts_side = batch_side['full_points'].to(device)
            dyn_side      = batch_side['dynamics'].to(device)

            # 2) Offsets
            offset_np   = side_offsets[idx_side]
            # offset_side = torch.from_numpy(offset_np).float().to(device).unsqueeze(0)  # Adds batch dim if needed

            offset_side = torch.from_numpy(offset_np).float().to(device)
            # print("offset_side shape:", offset_side.shape)
            # print("offset_side values:", offset_side)

            # 3) vantage predictions (frozen)
            boxA, confA = rear_model(full_pts_rear, dyn_rear)
            boxB_side, confB_side = side_model(full_pts_side, dyn_side)

            # 4) transform side -> rear
            boxB_rear = transform_box_side_to_rear(boxB_side, offset_side)

            # 5) fuse
            fused_box, fused_conf = fusion_mlp(boxA, confA, boxB_rear, confB_side)

            # 6) compute val loss
            loss, _, _ = late_fusion_loss(fused_box, bbox_gt_rear)
            total_loss += loss.item()

            # 7) iou stats for thresholds
            iou_vals = compute_iou(fused_box, bbox_gt_rear, thresholds=IOU_THRESHOLDS)
            for t in IOU_THRESHOLDS:
                iou_results[t] += iou_vals[t]

            batch_count += 1

    avg_loss = total_loss / batch_count if batch_count>0 else 0
    avg_iou  = {t: iou_results[t]/batch_count for t in IOU_THRESHOLDS}
    return avg_loss, avg_iou


########################################
# 5B) train_late_fusion
########################################
def train_late_fusion(
    rear_model, 
    side_model,
    fusion_mlp,
    rear_loader,
    side_loader,
    val_rear_loader,
    val_side_loader,
    side_offsets,
    optimizer,
    scheduler,
    epochs,
    device,
    save_path,
    log_file
):
    """
    For each epoch:
      - train on the train set (frozen vantage models)
      - validate on the val set
      - if val_loss is best, save checkpoint
      - scheduler step
    """
    from RaDE_net8 import compute_iou  # or you can import at the top

    log_print("=== LATE FUSION TRAIN START ===", log_file)
    log_print(f"Epochs: {epochs}", log_file)
    log_print(f"Using device: {device}", log_file)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --------------------
        # (A) Training Phase
        # --------------------
        train_start = time.time()
        fusion_mlp.train()
        rear_model.eval()  # vantage models frozen
        side_model.eval()

        total_loss = 0.0
        batch_count = 0

        # for logging IoU at multiple thresholds
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

        for batch_rear, batch_side in zip(rear_loader, side_loader):
            # 1) data
            full_pts_rear = batch_rear['full_points'].to(device)
            dyn_rear      = batch_rear['dynamics'].to(device)
            bbox_gt_rear  = batch_rear['bbox_gt'].to(device)
            idx_side      = batch_side['global_index']

            full_pts_side = batch_side['full_points'].to(device)
            dyn_side      = batch_side['dynamics'].to(device)

            # 2) offsets for side->rear
            offset_np = side_offsets[idx_side]
            offset_side = torch.from_numpy(offset_np).float().to(device)

            with torch.no_grad():
                boxA, confA = rear_model(full_pts_rear, dyn_rear)
                boxB_side, confB_side = side_model(full_pts_side, dyn_side)

            boxB_rear = transform_box_side_to_rear(boxB_side, offset_side)

            # fuse
            fused_box, fused_conf = fusion_mlp(boxA, confA, boxB_rear, confB_side)

            # compute loss
            loss, _, _ = late_fusion_loss(fused_box, bbox_gt_rear)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # IoU logging
            iou_vals = compute_iou(fused_box, bbox_gt_rear, thresholds=IOU_THRESHOLDS)
            for t in IOU_THRESHOLDS:
                iou_results[t] += iou_vals[t]

        train_end = time.time()
        train_time = train_end - train_start
        avg_train_loss = total_loss / batch_count if batch_count>0 else 0
        avg_train_iou  = {t: iou_results[t]/batch_count for t in IOU_THRESHOLDS}

        # --------------------
        # (B) Validation Phase
        # --------------------
        val_start = time.time()
        val_loss, val_iou = validate_late_fusion(
            fusion_mlp=fusion_mlp,
            rear_model=rear_model,
            side_model=side_model,
            val_rear_loader=val_rear_loader,
            val_side_loader=val_side_loader,
            side_offsets=side_offsets,
            device=device
        )
        val_end = time.time()
        val_time = val_end - val_start

        # if best => save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'fusion_model_state_dict': fusion_mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val_loss
            }, save_path)
            log_print(f"Saved best fusion checkpoint with val_loss={best_val_loss:.4f}", log_file)

        if scheduler is not None:
            scheduler.step(val_loss)

        total_epoch_time = train_time + val_time
        msg = (f"Epoch [{epoch+1}/{epochs}], "
               f"Train Loss={avg_train_loss:.4f}, "
               f"Val Loss={val_loss:.4f}, "
               f"Train IoU={{ {', '.join([f'{t}: {avg_train_iou[t]:.4f}' for t in avg_train_iou])} }}, "
               f"Val IoU={val_iou}, "
               f"Train Time={train_time:.2f}s, Val Time={val_time:.2f}s, "
               f"Total={total_epoch_time:.2f}s")
        log_print(msg, log_file)

    log_print("=== LATE FUSION TRAIN END ===\n", log_file)

########################################
# 5C) test_late_fusion
########################################
def test_late_fusion(
    rear_model, side_model, fusion_mlp,
    rear_loader, side_loader,
    side_offsets,
    device
):
    """
    Final test: no backprop, compute bounding-box + IoU loss
    """
    fusion_mlp.eval()
    rear_model.eval()
    side_model.eval()

    total_loss = 0.0
    total_l1   = 0.0
    total_iou  = 0.0
    count      = 0

    with torch.no_grad():
        for batch_rear, batch_side in zip(rear_loader, side_loader):
            full_points_rear = batch_rear['full_points'].to(device)
            dynamics_rear    = batch_rear['dynamics'].to(device)
            bbox_gt_rear     = batch_rear['bbox_gt'].to(device)
            idx_side         = batch_side['global_index']

            full_points_side = batch_side['full_points'].to(device)
            dynamics_side    = batch_side['dynamics'].to(device)

            offset_np = side_offsets[idx_side]
            offset_side = torch.from_numpy(offset_np).float().to(device)

            boxA, confA = rear_model(full_points_rear, dynamics_rear)
            boxB_side, confB_side = side_model(full_points_side, dynamics_side)

            boxB_rear = transform_box_side_to_rear(boxB_side, offset_side)
            fused_box, fused_conf = fusion_mlp(boxA, confA, boxB_rear, confB_side)

            # Use the same late_fusion_loss for consistency
            loss_val, l1_val, iou_val = late_fusion_loss(fused_box, bbox_gt_rear)
            total_loss += loss_val
            total_l1   += l1_val
            total_iou  += iou_val
            count      += 1

    avg_loss = total_loss / count if count>0 else 0
    avg_l1   = total_l1 / count if count>0 else 0
    avg_iou  = total_iou / count if count>0 else 0
    return avg_loss, avg_l1, avg_iou


########################################
# 6) Main
########################################
def main():
    parser = argparse.ArgumentParser(description="Late Fusion with Train+Val+Test Example")

    # 1) Rear dataset
    parser.add_argument("--rear_input_file", type=str, default="data_late_fusion/rear/inputs.npy")
    parser.add_argument("--rear_label_file", type=str, default="data_late_fusion/rear/labels.npy")
    parser.add_argument("--rear_train_indices", type=str, default="data_late_fusion/rear/train_indices.npy")
    parser.add_argument("--rear_val_indices", type=str, default="data_late_fusion/rear/val_indices.npy")
    parser.add_argument("--rear_test_indices", type=str, default="data_late_fusion/rear/test_indices.npy")

    # 2) Side dataset
    parser.add_argument("--side_input_file", type=str, default="data_late_fusion/side/inputs.npy")
    parser.add_argument("--side_label_file", type=str, default="data_late_fusion/side/labels.npy")
    parser.add_argument("--side_train_indices", type=str, default="data_late_fusion/side/train_indices.npy")
    parser.add_argument("--side_val_indices", type=str, default="data_late_fusion/side/val_indices.npy")
    parser.add_argument("--side_test_indices", type=str, default="data_late_fusion/side/test_indices.npy")
    parser.add_argument("--side_gps_offsets", type=str, default="data_late_fusion/side/gps_offsets.npy")

    # 3) Pre-trained vantage model ckpts
    parser.add_argument("--rear_ckpt", type=str, default="models/best_rear_model.pth")
    parser.add_argument("--side_ckpt", type=str, default="models/best_side_model.pth")

    # 4) Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_fusion", type=str, default="fusion_mlp.pth")
    parser.add_argument("--log_file", type=str, default="fusion_log.txt")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(sd=42):
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sd)
    set_seed(42)

    # =========================================
    # A) Create train/val/test Datasets
    # =========================================
    # Train
    rear_train_dataset = RadarDataset(
        input_file=args.rear_input_file,
        label_file=args.rear_label_file,
        train_indices_file=args.rear_train_indices,
        val_indices_file=args.rear_val_indices,
        test_indices_file=args.rear_test_indices,
        split="train"
    )
    side_train_dataset = RadarDataset(
        input_file=args.side_input_file,
        label_file=args.side_label_file,
        train_indices_file=args.side_train_indices,
        val_indices_file=args.side_val_indices,
        test_indices_file=args.side_test_indices,
        split="train"
    )
    rear_train_loader = DataLoader(rear_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    side_train_loader = DataLoader(side_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    for rear_batch, side_batch in zip(rear_train_loader, side_train_loader):

        # Pick the correct key from the batch (e.g., 'full_points' or another main key)
        key = 'full_points'  # Change this if needed
        assert rear_batch[key].shape[0] == side_batch[key].shape[0], "Batch size mismatch!"

    # Val
    rear_val_dataset = RadarDataset(
        input_file=args.rear_input_file,
        label_file=args.rear_label_file,
        train_indices_file=args.rear_train_indices,
        val_indices_file=args.rear_val_indices,
        test_indices_file=args.rear_test_indices,
        split="val"
    )
    side_val_dataset = RadarDataset(
        input_file=args.side_input_file,
        label_file=args.side_label_file,
        train_indices_file=args.side_train_indices,
        val_indices_file=args.side_val_indices,
        test_indices_file=args.side_test_indices,
        split="val"
    )
    rear_val_loader = DataLoader(rear_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    side_val_loader = DataLoader(side_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Test
    rear_test_dataset = RadarDataset(
        input_file=args.rear_input_file,
        label_file=args.rear_label_file,
        train_indices_file=args.rear_train_indices,
        val_indices_file=args.rear_val_indices,
        test_indices_file=args.rear_test_indices,
        split="test"
    )
    side_test_dataset = RadarDataset(
        input_file=args.side_input_file,
        label_file=args.side_label_file,
        train_indices_file=args.side_train_indices,
        val_indices_file=args.side_val_indices,
        test_indices_file=args.side_test_indices,
        split="test"
    )
    rear_test_loader = DataLoader(rear_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    side_test_loader = DataLoader(side_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    # =========================================
    # B) Load side offsets
    # =========================================
    side_offsets = np.load(args.side_gps_offsets, allow_pickle=True)

    # =========================================
    # C) Create vantage models, load ckpts
    # =========================================
    radar_model_rear = RadarModel(input_dim=3, dynamic_dim=3, hidden_dim=256)
    rear_model = RadarDepthEstimationModel(radar_model=radar_model_rear, hidden_dim=256)
    ckpt_rear = torch.load(args.rear_ckpt, map_location=device)
    rear_model.load_state_dict(ckpt_rear["model_state_dict"])
    rear_model.to(device)
    rear_model.eval()
    for p in rear_model.parameters():
        p.requires_grad = False

    radar_model_side = RadarModel(input_dim=3, dynamic_dim=3, hidden_dim=256)
    side_model = RadarDepthEstimationModel(radar_model=radar_model_side, hidden_dim=256)
    ckpt_side = torch.load(args.side_ckpt, map_location=device)
    side_model.load_state_dict(ckpt_side["model_state_dict"])
    side_model.to(device)
    side_model.eval()
    for p in side_model.parameters():
        p.requires_grad = False

    # =========================================
    # D) Fusion MLP
    # =========================================
    fusion_mlp = FusionMLP(hidden_size=64).to(device)

    # Create optimizer for fusion MLP only
    optimizer = torch.optim.Adam(fusion_mlp.parameters(), lr=args.lr, weight_decay=1e-3)
    # Optional: a scheduler like your baseline
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=0.0001,
        cooldown=0,
        min_lr=1e-6,
        verbose=True
    )
    # print(f"Log file path: {args.log_file}")  # Debugging line
    # print(f"Directory path: {os.path.dirname(args.log_file)}")  # Debugging line
    # os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_f = open(args.log_file, "w")

    # =========================================
    # E) Train with Validation each epoch
    # =========================================
    train_late_fusion(
        rear_model=rear_model,
        side_model=side_model,
        fusion_mlp=fusion_mlp,
        rear_loader=rear_train_loader,
        side_loader=side_train_loader,
        val_rear_loader=rear_val_loader,      # pass val loaders
        val_side_loader=side_val_loader,
        side_offsets=side_offsets,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        save_path=args.save_fusion,
        log_file=log_f
    )

    # =========================================
    # F) Test with the best fusion checkpoint
    # =========================================
    best_ckpt = torch.load(args.save_fusion, map_location=device)
    fusion_mlp.load_state_dict(best_ckpt["fusion_model_state_dict"])
    fusion_mlp.eval()

    val_loss, val_l1, val_iou = test_late_fusion(
        rear_model=rear_model,
        side_model=side_model,
        fusion_mlp=fusion_mlp,
        rear_loader=rear_val_loader,
        side_loader=side_val_loader,
        side_offsets=side_offsets,
        device=device
    )
    log_print(f"\n[Validation after training] Loss={val_loss:.4f}, L1={val_l1:.4f}, IoU={val_iou:.4f}", log_f)

    test_loss, test_l1, test_iou = test_late_fusion(
        rear_model=rear_model,
        side_model=side_model,
        fusion_mlp=fusion_mlp,
        rear_loader=rear_test_loader,
        side_loader=side_test_loader,
        side_offsets=side_offsets,
        device=device
    )
    log_print(f"[Test] Loss={test_loss:.4f}, L1={test_l1:.4f}, IoU={test_iou:.4f}", log_f)

    log_f.close()


if __name__ == "__main__":
    main()
