import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import random
import argparse
import os
from typing import List, Dict
from datetime import datetime

# ---------------------------
# 1) RadarDataset Class
# ---------------------------
class RadarDataset(Dataset):
    def __init__(self, input_file, label_file, train_indices_file=None, val_indices_file=None, isValidation=False):
        """
        Load radar sensing data and split into train/validation sets.
        Args:
            input_file (str): Path to the inputs .npy file.
            label_file (str): Path to the labels .npy file.
            train_indices_file (str, optional): Path to train indices .npy file.
            val_indices_file (str, optional): Path to validation indices .npy file.
            isValidation (bool): Flag to determine train or validation split.
        """
        try:
            self.inputs = np.load(input_file, allow_pickle=True)
            self.labels = np.load(label_file, allow_pickle=True)
            
            # Load indices conditionally
            if train_indices_file is not None:
                self.train_indices = np.load(train_indices_file, allow_pickle=True)
            else:
                self.train_indices = None
            
            if val_indices_file is not None:
                self.val_indices = np.load(val_indices_file, allow_pickle=True)
            else:
                self.val_indices = None
            
            self.isValidation = isValidation

            # Log dataset details
            if self.isValidation:
                print(f"Loaded validation dataset with {len(self.val_indices)} samples.")
            else:
                print(f"Loaded training dataset with {len(self.train_indices)} samples.")
        except Exception as e:
            raise IOError(f"Error loading data files: {e}")

    def __len__(self):
        if self.isValidation:
            return len(self.val_indices)
        else:
            return len(self.train_indices)

    def __getitem__(self, idx):
        ID = self.val_indices[idx] if self.isValidation else self.train_indices[idx]
        points = torch.tensor(self.inputs[ID], dtype=torch.float32)
        labels = torch.tensor(self.labels[ID], dtype=torch.float32)

        return {
            'points': points[:, [0, 1, 2]],   # X, Y, Z coordinates
            'dynamics': points[:, [3, 4, 5]], # Velocity, Range, Bearing
            'range': points[:, 4],           # Range
            'bearing': points[:, 5],         # Bearing
            'intensity': points[:, 6],       # Intensity
            'bbox_gt': labels[0, :7],        # (w, h, l, x, y, z, theta)
            'depth_gt': labels[0, 5]         # Z position (depth GT)
        }


# ---------------------------
# 2) Self-Attention Module
# ---------------------------
class SelfAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, v)


# ---------------------------
# 3) Depth Estimation Subnet
# ---------------------------
class DepthEstimationSubnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # nn.Sigmoid()  # Outputs confidence score between 0 and 1
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------
# 4) Bounding Box Decoder
# ---------------------------
class BoundingBoxDecoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # (w, h, l, x, y, z, theta)
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------
# 5) Core Radar Model
# ---------------------------
class RadarModel(nn.Module):
    def __init__(self, input_dim=3, dynamic_dim=3, hidden_dim=128):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        self.attention = SelfAttentionModule(hidden_dim * 2)

    def forward(self, points, dynamics):
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, points.shape[-1])
        dynamics_flat = dynamics.view(-1, dynamics.shape[-1])

        # Encode features
        point_features = self.point_encoder(points_flat)
        dynamic_features = self.dynamic_encoder(dynamics_flat)

        # Concatenate encoded features
        combined_features = torch.cat([point_features, dynamic_features], dim=1)

        # Reshape back for attention
        combined_features = combined_features.view(batch_size, num_points, -1)
        attended_features = self.attention(combined_features)

        return attended_features


# ---------------------------
# 6) Full Radar Depth Estimation Model
# ---------------------------
class RadarDepthEstimationModel(nn.Module):
    def __init__(self, radar_model, hidden_dim=128):
        super().__init__()
        self.radar_model = radar_model
        self.depth_subnet = DepthEstimationSubnet(hidden_dim * 2)
        self.decoder = BoundingBoxDecoder(hidden_dim * 2)

    def forward(self, points, dynamics):
        attended_features = self.radar_model(points, dynamics)
        # Global average pooling over the points dimension
        pooled_features = attended_features.mean(dim=1)
        depth_confidence = self.depth_subnet(pooled_features)
        bbox_prediction = self.decoder(pooled_features)
        return bbox_prediction, depth_confidence


# ---------------------------
# 7) Attention-Based Fusion Layer
# ---------------------------
class AttentionFusionLayer(nn.Module):
    def __init__(self, input_dim=16):
        """
        A lightweight feedforward attention layer for fusing predictions.
        Expecting 7 (bbox_pred) + 1 (depth_conf) = 8 dims per model => total 16.
        """
        super(AttentionFusionLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)  # Normalize weights to sum to 1
        )

    def forward(self, features):
        return self.fc(features)


# ---------------------------
# 8) Axis-Aligned 3D IoU Utility (ignores theta)
# ---------------------------
def axis_aligned_iou_3d(bboxes1, bboxes2):
    w1, h1, l1, cx1, cy1, cz1, _ = [bboxes1[:, i] for i in range(7)]
    w2, h2, l2, cx2, cy2, cz2, _ = [bboxes2[:, i] for i in range(7)]

    x1_min = cx1 - (w1 / 2.0)
    x1_max = cx1 + (w1 / 2.0)
    y1_min = cy1 - (h1 / 2.0)
    y1_max = cy1 + (h1 / 2.0)
    z1_min = cz1 - (l1 / 2.0)
    z1_max = cz1 + (l1 / 2.0)

    x2_min = cx2 - (w2 / 2.0)
    x2_max = cx2 + (w2 / 2.0)
    y2_min = cy2 - (h2 / 2.0)
    y2_max = cy2 + (h2 / 2.0)
    z2_min = cz2 - (l2 / 2.0)
    z2_max = cz2 + (l2 / 2.0)

    inter_x = torch.clamp(torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min), min=0)
    inter_y = torch.clamp(torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min), min=0)
    inter_z = torch.clamp(torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min), min=0)

    intersection = inter_x * inter_y * inter_z
    vol1 = w1 * h1 * l1
    vol2 = w2 * h2 * l2
    union = vol1 + vol2 - intersection
    iou = intersection / (union + 1e-6)
    return iou


# ---------------------------
# 9) IoU thresholds
# ---------------------------
IOU_THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


# ---------------------------
# 10) IoU Computation
# ---------------------------
def compute_iou(pred_boxes, gt_boxes, thresholds=IOU_THRESHOLDS):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    iou_scores = {}
    for thresh in thresholds:
        iou_scores[thresh] = (ious > thresh).float().mean().item()
    return iou_scores


# ---------------------------
# 11) IoU Loss
# ---------------------------
def iou_loss_fn(pred_boxes, gt_boxes):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    return 1.0 - ious.mean()


# ---------------------------
# 12) Custom Loss (w_bbox, w_depth, w_iou)
# ---------------------------
def custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights):
    w_bbox, w_depth, w_iou = loss_weights
    
    # bounding-box (Smooth L1)
    bbox_gt = bbox_gt.view(-1, 7)
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_gt)

    # continuous depth (L1)
    depth_conf = depth_conf.view(-1)
    depth_gt = depth_gt.view(-1)
    # depth_loss = F.binary_cross_entropy(depth_conf, depth_gt)
    depth_loss = F.l1_loss(depth_conf.view(-1), depth_gt.view(-1))

    iou_loss_value = iou_loss_fn(bbox_pred, bbox_gt)

    return w_bbox * bbox_loss + w_depth * depth_loss + w_iou * iou_loss_value


def log_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")


# ---------------------------
# 13) train_fusion
# ---------------------------
def train_fusion(
    rear_model: nn.Module,
    side_model: nn.Module,
    fusion_layer: nn.Module,
    rear_train_loader: DataLoader,
    side_train_loader: DataLoader,
    rear_val_loader: DataLoader,
    side_val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_weights: tuple,
    log_file,
    device: torch.device,
    save_path: str
):
    """
    Train a fusion approach on two vantage points (rear + side).
    Each epoch:
      - zip() the rear and side loaders (train)
      - fuse
      - compute fused loss
      - backprop
    Then validation is done the same way.
    """
    def log_print_local(msg):
        print(msg)
        log_file.write(msg + "\n")

    log_print_local("=== TRAINING START (Fusion) ===")
    log_print_local(f"Batch size: {batch_size}")
    log_print_local(f"Learning rate: {learning_rate}")
    log_print_local(f"Epochs: {epochs}")
    log_print_local(f"Loss Weights (bbox, depth, iou): {loss_weights}")
    log_print_local(f"Using device: {device}\n")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        rear_model.train()
        side_model.train()
        fusion_layer.train()

        total_loss = 0.0
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

        # Train loop
        for (rear_batch, side_batch) in zip(rear_train_loader, side_train_loader):
            optimizer.zero_grad()

            # Move data
            points_rear = rear_batch['points'].to(device)
            dynamics_rear = rear_batch['dynamics'].to(device)
            bbox_gt_rear = rear_batch['bbox_gt'].to(device)
            depth_gt_rear = rear_batch['depth_gt'].to(device)

            points_side = side_batch['points'].to(device)
            dynamics_side = side_batch['dynamics'].to(device)
            bbox_gt_side = side_batch['bbox_gt'].to(device)
            depth_gt_side = side_batch['depth_gt'].to(device)

            # Forward pass
            rear_bbox_pred, rear_depth_conf = rear_model(points_rear, dynamics_rear)
            side_bbox_pred, side_depth_conf = side_model(points_side, dynamics_side)

            # Fusion
            features = torch.cat([rear_bbox_pred, rear_depth_conf,
                                  side_bbox_pred, side_depth_conf], dim=-1)
            weights = fusion_layer(features)

            bbox_fused  = weights[:,0:1]*rear_bbox_pred + weights[:,1:2]*side_bbox_pred
            depth_fused = weights[:,0:1]*rear_depth_conf + weights[:,1:2]*side_depth_conf

            # Use rear ground truth
            bbox_gt  = bbox_gt_rear
            depth_gt = depth_gt_rear

            # Loss
            loss = custom_loss(bbox_fused, bbox_gt, depth_fused, depth_gt, loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(rear_model.parameters())
                + list(side_model.parameters())
                + list(fusion_layer.parameters()), 
                max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()

            # Optional IoU
            iou_batch = compute_iou(bbox_fused, bbox_gt, thresholds=IOU_THRESHOLDS)
            for k in iou_results:
                iou_results[k] += iou_batch[k]

        avg_train_loss = total_loss / len(rear_train_loader)
        avg_train_iou = {k: iou_results[k]/len(rear_train_loader) for k in iou_results}

        # Validation
        val_loss, val_iou = validate_fusion(
            rear_model, side_model, fusion_layer,
            rear_val_loader, side_val_loader,
            loss_weights, device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'rear_model_state_dict': rear_model.state_dict(),
                'side_model_state_dict': side_model.state_dict(),
                'fusion_layer_state_dict': fusion_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val_loss
            }, save_path)
            log_print_local(f"Saved best fusion model (val_loss={best_val_loss:.4f})")

        scheduler.step(val_loss)

        end_time = time.time()
        msg = (f"Epoch {epoch+1}/{epochs}, "
               f"Train Loss: {avg_train_loss:.4f}, "
               f"Val Loss: {val_loss:.4f}, "
               f"Train IoU: {avg_train_iou}, "
               f"Val IoU: {val_iou}, "
               f"Time: {end_time - start_time:.2f}s")
        log_print_local(msg)

    log_print_local("=== TRAINING END (Fusion) ===\n")


def validate_fusion(
    rear_model: nn.Module,
    side_model: nn.Module,
    fusion_layer: nn.Module,
    rear_val_loader: DataLoader,
    side_val_loader: DataLoader,
    loss_weights: tuple,
    device: torch.device
):
    rear_model.eval()
    side_model.eval()
    fusion_layer.eval()

    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

    with torch.no_grad():
        for (rear_batch, side_batch) in zip(rear_val_loader, side_val_loader):
            points_rear   = rear_batch['points'].to(device)
            dynamics_rear = rear_batch['dynamics'].to(device)
            bbox_gt_rear  = rear_batch['bbox_gt'].to(device)
            depth_gt_rear = rear_batch['depth_gt'].to(device)

            points_side   = side_batch['points'].to(device)
            dynamics_side = side_batch['dynamics'].to(device)
            bbox_gt_side  = side_batch['bbox_gt'].to(device)
            depth_gt_side = side_batch['depth_gt'].to(device)

            rear_bbox_pred, rear_depth_conf = rear_model(points_rear, dynamics_rear)
            side_bbox_pred, side_depth_conf = side_model(points_side, dynamics_side)

            features = torch.cat([rear_bbox_pred, rear_depth_conf,
                                  side_bbox_pred, side_depth_conf], dim=-1)
            weights = fusion_layer(features)

            bbox_fused  = weights[:,0:1]*rear_bbox_pred  + weights[:,1:2]*side_bbox_pred
            depth_fused = weights[:,0:1]*rear_depth_conf + weights[:,1:2]*side_depth_conf

            # Use rear GT
            bbox_gt  = bbox_gt_rear
            depth_gt = depth_gt_rear

            loss = custom_loss(bbox_fused, bbox_gt, depth_fused, depth_gt, loss_weights)
            total_loss += loss.item()

            # IoU
            iou_batch = compute_iou(bbox_fused, bbox_gt, thresholds=IOU_THRESHOLDS)
            for k in iou_results:
                iou_results[k] += iou_batch[k]

    avg_val_loss = total_loss / len(rear_val_loader)
    avg_val_iou = {k: iou_results[k]/len(rear_val_loader) for k in iou_results}
    return avg_val_loss, avg_val_iou


# ---------------------------------------
# 14) Main Function (No test)
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Radar Depth Estimation with Fusion Layer (Train & Val only)')
    
    # Model checkpoint paths (optional)
    parser.add_argument('--rear_model_path', type=str, help='Path to load a pre-trained rear model (optional)')
    parser.add_argument('--side_model_path', type=str, help='Path to load a pre-trained side model (optional)')

    # REAR Data Files
    parser.add_argument('--rear_train_inputs', type=str, required=True, help='Path to REAR train inputs.npy')
    parser.add_argument('--rear_train_labels', type=str, required=True, help='Path to REAR train labels.npy')
    parser.add_argument('--rear_train_indices', type=str, required=True, help='Path to REAR train_indices.npy')

    parser.add_argument('--rear_val_inputs', type=str, required=True, help='Path to REAR val inputs.npy')
    parser.add_argument('--rear_val_labels', type=str, required=True, help='Path to REAR val labels.npy')
    parser.add_argument('--rear_val_indices', type=str, required=True, help='Path to REAR val_indices.npy')

    # SIDE Data Files
    parser.add_argument('--side_train_inputs', type=str, required=True, help='Path to SIDE train inputs.npy')
    parser.add_argument('--side_train_labels', type=str, required=True, help='Path to SIDE train labels.npy')
    parser.add_argument('--side_train_indices', type=str, required=True, help='Path to SIDE train_indices.npy')

    parser.add_argument('--side_val_inputs', type=str, required=True, help='Path to SIDE val inputs.npy')
    parser.add_argument('--side_val_labels', type=str, required=True, help='Path to SIDE val labels.npy')
    parser.add_argument('--side_val_indices', type=str, required=True, help='Path to SIDE val_indices.npy')

    # Logging + hyperparams
    parser.add_argument('--log_file', type=str, default="./results/fusion_log.txt", help='Path to log file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    # Set seeds
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed(42)

    # Ensure log dir
    log_dir = os.path.dirname(args.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # REAR Data (Train + Val)
    # =========================
    rear_train_dataset = RadarDataset(
        input_file=args.rear_train_inputs,
        label_file=args.rear_train_labels,
        train_indices_file=args.rear_train_indices,
        val_indices_file=None,
        isValidation=False
    )
    rear_train_loader = DataLoader(
        rear_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    rear_val_dataset = RadarDataset(
        input_file=args.rear_val_inputs,
        label_file=args.rear_val_labels,
        train_indices_file=None,
        val_indices_file=args.rear_val_indices,
        isValidation=True
    )
    rear_val_loader = DataLoader(
        rear_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # =========================
    # SIDE Data (Train + Val)
    # =========================
    side_train_dataset = RadarDataset(
        input_file=args.side_train_inputs,
        label_file=args.side_train_labels,
        train_indices_file=args.side_train_indices,
        val_indices_file=None,
        isValidation=False
    )
    side_train_loader = DataLoader(
        side_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    side_val_dataset = RadarDataset(
        input_file=args.side_val_inputs,
        label_file=args.side_val_labels,
        train_indices_file=None,
        val_indices_file=args.side_val_indices,
        isValidation=True
    )
    side_val_loader = DataLoader(
        side_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    print("All datasets loaded. (Train + Val only)")

    # Initialize Models
    rear_model = RadarDepthEstimationModel(radar_model=RadarModel(input_dim=3, hidden_dim=128)).to(device)
    side_model = RadarDepthEstimationModel(radar_model=RadarModel(input_dim=3, hidden_dim=128)).to(device)

    # (Optional) Load pre-trained single-view checkpoints if desired
    # e.g. if args.rear_model_path is not None:
    #   ckpt = torch.load(args.rear_model_path)
    #   rear_model.load_state_dict(ckpt['model_state_dict'])

    # Initialize Fusion Layer
    fusion_layer = AttentionFusionLayer(input_dim=16).to(device)  # (7 + 1)*2

    # Combine params
    all_params = list(rear_model.parameters()) \
               + list(side_model.parameters()) \
               + list(fusion_layer.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Train + Val (Fusion)
    with open(args.log_file, "w") as log_f:
        train_fusion(
            rear_model=rear_model,
            side_model=side_model,
            fusion_layer=fusion_layer,
            rear_train_loader=rear_train_loader,
            side_train_loader=side_train_loader,
            rear_val_loader=rear_val_loader,
            side_val_loader=side_val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=1e-4,
            loss_weights=(3.0,1.0,1.0),  # (w_bbox, w_depth, w_iou)
            log_file=log_f,
            device=device,
            save_path="best_fusion.pth"
        )

    print("Training + Validation complete. No test set in this script.")

if __name__ == "__main__":
    main()
