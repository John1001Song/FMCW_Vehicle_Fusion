import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import time
import argparse
from typing import List, Dict

# ---------------------------
# 1) RadarDataset Class
# ---------------------------
class RadarDataset(Dataset):
    def __init__(self, input_file, label_file, train_indices_file, val_indices_file, test_indices_file=None, split="train"):
        try:
            self.inputs = np.load(input_file, allow_pickle=True)
            self.labels = np.load(label_file, allow_pickle=True)
            self.train_indices = np.load(train_indices_file, allow_pickle=True)
            self.val_indices = np.load(val_indices_file, allow_pickle=True)
            self.split = split
            if test_indices_file is not None:
                self.test_indices = np.load(test_indices_file, allow_pickle=True)
            else:
                self.test_indices = None

            print(f"Loaded {len(self.inputs)} inputs, {len(self.labels)} labels")
            print(f"Train samples: {len(self.train_indices)}, Validation samples: {len(self.val_indices)}", end="")
            if self.test_indices is not None:
                print(f", Test samples: {len(self.test_indices)}")
            else:
                print("")
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
            raise ValueError("Invalid split value. Choose from 'train', 'val', or 'test'.")

    def __getitem__(self, idx):
        if self.split == "train":
            ID = self.train_indices[idx]
        elif self.split == "val":
            ID = self.val_indices[idx]
        elif self.split == "test":
            ID = self.test_indices[idx]
        else:
            raise ValueError("Invalid split value. Choose from 'train', 'val', or 'test'.")
        points = torch.tensor(self.inputs[ID], dtype=torch.float32)
        labels = torch.tensor(self.labels[ID], dtype=torch.float32)
        return {
            'full_points': points,           # shape: (num_points, feature_dim)
            'points': points[:, [0, 1, 2]],    # X, Y, Z coordinates
            'dynamics': points[:, [3, 4, 5]],  # Velocity, Range, Bearing
            'range': points[:, 4],
            'bearing': points[:, 5],
            'intensity': points[:, 6],
            'bbox_gt': labels[0, :7],
            'depth_gt': labels[0, 5]
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
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs confidence score between 0 and 1
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
# 5) Core Radar Model with Multi-Branch Feature Extraction
# ---------------------------
class RadarModel(nn.Module):
    def __init__(self, input_dim=3, dynamic_dim=3, hidden_dim=256):
        super().__init__()
        # --- Position Branch: processes 3D coordinates ---
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        # --- Dynamics Branch: processes velocity, range, bearing ---
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        # Fusion of position and dynamics features.
        self.fusion = nn.Linear(512, hidden_dim)
        self.attention = SelfAttentionModule(hidden_dim)
        # --- Intensity Branch ---
        self.intensity_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.intensity_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)  # Computes attention weights per point.
        )

    def forward(self, full_points, dynamics):
        batch_size, num_points, feat_dim = full_points.shape

        # Process position features (first 3 dimensions)
        position = full_points[:, :, :3]  # (batch, num_points, 3)
        position_flat = position.view(-1, 3)
        p_features = self.point_encoder(position_flat)  # (batch*num_points, 256)

        # Process dynamics features (provided separately)
        dynamics_flat = dynamics.view(-1, dynamics.shape[-1])
        d_features = self.dynamic_encoder(dynamics_flat)  # (batch*num_points, 256)

        # Concatenate and fuse.
        combined = torch.cat([p_features, d_features], dim=1)  # (batch*num_points, 512)
        fused = self.fusion(combined)  # (batch*num_points, hidden_dim) where hidden_dim=256
        fused = fused.view(batch_size, num_points, -1)  # (batch, num_points, 256)
        attended_features = self.attention(fused)  # (batch, num_points, 256)
        pooled_main = attended_features.mean(dim=1)  # (batch, 256)

        # --- Intensity Branch ---
        intensity = full_points[:, :, 6].unsqueeze(-1)  # (batch, num_points, 1)
        intensity_flat = intensity.view(-1, 1)  # (batch*num_points, 1)
        intensity_feat = self.intensity_branch(intensity_flat)  # (batch*num_points, 256)
        intensity_feat = intensity_feat.view(batch_size, num_points, -1)  # (batch, num_points, 256)
        intensity_flat_for_att = intensity_feat.view(-1, 256)
        int_weights = self.intensity_attention(intensity_flat_for_att)  # (batch*num_points, 1)
        int_weights = int_weights.view(batch_size, num_points, 1)
        weighted_intensity = intensity_feat * int_weights  # (batch, num_points, 256)
        pooled_intensity = weighted_intensity.mean(dim=1)  # (batch, 256)

        # Concatenate pooled main and intensity features.
        combined_features = torch.cat([pooled_main, pooled_intensity], dim=1)  # (batch, 512)
        return combined_features

# ---------------------------
# 6) Full Radar Depth Estimation Model with Optional Temporal Fusion
# ---------------------------
class RadarDepthEstimationModel(nn.Module):
    def __init__(self, radar_model, hidden_dim=256):
        super().__init__()
        self.radar_model = radar_model
        # Fused feature dimension is 512.
        self.depth_subnet = DepthEstimationSubnet(512)
        self.decoder = BoundingBoxDecoder(512)
    
    def forward(self, full_points, dynamics):
        # If input is per-frame (3D tensor), no further pooling is needed.
        if full_points.dim() == 3:
            fused_features = self.radar_model(full_points, dynamics)  # (batch, 512)
            depth_conf = self.depth_subnet(fused_features)
            bbox_pred = self.decoder(fused_features)
            return bbox_pred, depth_conf
        # If input is sequential (4D tensor), process frame-by-frame and then fuse temporally.
        elif full_points.dim() == 4:
            batch_size, seq_len, num_points, _ = full_points.shape
            fused_features_seq = []
            for t in range(seq_len):
                fused = self.radar_model(full_points[:, t, :, :], dynamics[:, t, :, :])
                fused_features_seq.append(fused.unsqueeze(1))
            fused_features_seq = torch.cat(fused_features_seq, dim=1)  # (batch, seq_len, 512)
            # Here you can apply a temporal transformer module (with CLS token) if desired.
            # For now, we simply average over the sequence.
            sequence_rep = fused_features_seq.mean(dim=1)  # (batch, 512)
            depth_conf = self.depth_subnet(sequence_rep)
            bbox_pred = self.decoder(sequence_rep)
            return bbox_pred, depth_conf
        else:
            raise ValueError("Input tensor must be 3D or 4D.")

# ---------------------------
# 7) Temporal Transformer Module (using CLS token)
# ---------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, token_dim, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=50):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, token_dim)
        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, token_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, token_dim)
        x = x + self.pos_embedding[:, :seq_len+1, :]
        x = self.dropout(x)
        # Transformer expects (seq_len+1, batch, token_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        cls_out = x[0]  # (batch, token_dim)
        return cls_out

# Optionally attach the temporal transformer to the RadarDepthEstimationModel if sequential inputs are used.
# For example:
# model.temporal_transformer = TemporalTransformer(token_dim=512, n_layers=2, n_heads=4)

# ---------------------------------------
# Axis-Aligned 3D IoU Utility and Loss Functions
# ---------------------------------------
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
    return intersection / (union + 1e-6)

IOU_THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

def compute_iou(pred_boxes, gt_boxes, thresholds=IOU_THRESHOLDS):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    return {thresh: (ious > thresh).float().mean().item() for thresh in thresholds}

def iou_loss_fn(pred_boxes, gt_boxes):
    return 1.0 - axis_aligned_iou_3d(pred_boxes, gt_boxes).mean()

def custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights):
    # loss_weights: (w_bbox, w_depth, w_iou, w_intensity)
    w_bbox, w_depth, w_iou, w_intensity = loss_weights
    bbox_gt = bbox_gt.view(-1, 7)
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_gt)
    depth_conf = depth_conf.view(-1)
    depth_gt = depth_gt.view(-1)
    depth_loss = F.binary_cross_entropy(depth_conf, depth_gt)
    iou_loss_value = iou_loss_fn(bbox_pred, bbox_gt)
    intensity_loss = 0.0  # (Placeholder; intensity loss could be added later)
    return w_bbox * bbox_loss + w_depth * depth_loss + w_iou * iou_loss_value + w_intensity * intensity_loss

# ---------------------------------------
# Helper: Logging function
# ---------------------------------------
def log_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

# ---------------------------------------
# Training, Validation, and Testing Functions with Timing
# ---------------------------------------
def train_model(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                epochs: int, loss_weights: tuple, log_file, device: torch.device, save_path: str):
    log_print("=== TRAINING START ===", log_file)
    log_print(f"Epochs: {epochs}", log_file)
    log_print(f"Loss Weights (bbox, depth, IoU, intensity): {loss_weights}", log_file)
    log_print(f"Using device: {device}", log_file)
    log_print("", log_file)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_start = time.time()
        total_loss = 0.0
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            full_points = batch['full_points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            bbox_pred, depth_conf = model(full_points, dynamics)
            loss = custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
        train_end = time.time()
        train_time = train_end - train_start

        val_start = time.time()
        val_loss, val_iou = validate_model(model, val_dataloader, loss_weights, device)
        val_end = time.time()
        val_time = val_end - val_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val_loss
            }, save_path)
            log_print(f"Saved best model with validation loss: {best_val_loss:.4f}", log_file)

        scheduler.step(val_loss)
        total_epoch_time = train_time + val_time
        msg = (f"Epoch {epoch+1}/{epochs}, "
               f"Train Loss: {total_loss / len(train_dataloader):.4f}, "
               f"Val Loss: {val_loss:.4f}, "
               f"Train IoU: {{ {', '.join([f'{k}: {iou_results[k] / len(train_dataloader):.4f}' for k in iou_results])} }}, "
               f"Val IoU: {val_iou}, "
               f"Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s, Total Epoch Time: {total_epoch_time:.2f}s")
        log_print(msg, log_file)
    log_print("=== TRAINING END ===\n", log_file)

def validate_model(model: nn.Module, val_dataloader: DataLoader, loss_weights: tuple, device: torch.device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
    with torch.no_grad():
        for batch in val_dataloader:
            full_points = batch['full_points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            bbox_pred, depth_conf = model(full_points, dynamics)
            loss = custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights)
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
    avg_loss = total_loss / len(val_dataloader)
    avg_iou = {k: iou_results[k] / len(val_dataloader) for k in iou_results}
    return avg_loss, avg_iou

def test_model(model: nn.Module, dataloader: DataLoader, log_file, loss_weights: tuple, device: torch.device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
    with torch.no_grad():
        for batch in dataloader:
            full_points = batch['full_points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            bbox_pred, depth_conf = model(full_points, dynamics)
            loss = custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights)
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
    avg_loss = total_loss / len(dataloader)
    avg_iou = {key: iou_results[key] / len(dataloader) for key in iou_results}
    msg = f"Test Loss: {avg_loss:.4f}, IoU: {avg_iou}"
    log_print(msg, log_file)

# ---------------------------------------
# Main Function
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Radar Depth Estimation Training Script with Temporal Fusion')
    parser.add_argument('--input_file', type=str, required=True, help='Path to inputs.npy')
    parser.add_argument('--label_file', type=str, required=True, help='Path to labels.npy')
    parser.add_argument('--train_indices_file', type=str, required=True, help='Path to train_indices.npy')
    parser.add_argument('--val_indices_file', type=str, required=True, help='Path to val_indices.npy')
    parser.add_argument('--test_indices_file', type=str, required=True, help='Path to test_indices.npy')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--loss_weights', type=float, nargs=4, default=[0.70, 0.10, 0.10, 0.10],
                        help='Loss weights for bbox, depth, IoU, and intensity respectively')
    parser.add_argument('--log_file', type=str, default="./results/train_test_log.txt", help='Path to log file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--save_model_dir', type=str, default="./results/", help='Directory to save the trained model')
    args = parser.parse_args()

    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = RadarDataset(args.input_file, args.label_file, args.train_indices_file,
                                  args.val_indices_file, args.test_indices_file, split="train")
    val_dataset = RadarDataset(args.input_file, args.label_file, args.train_indices_file,
                                args.val_indices_file, args.test_indices_file, split="val")
    test_dataset = RadarDataset(args.input_file, args.label_file, args.train_indices_file,
                                 args.val_indices_file, args.test_indices_file, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # Instantiate the radar model with extended branches.
    radar_model = RadarModel(input_dim=3, dynamic_dim=3, hidden_dim=256)
    model = RadarDepthEstimationModel(radar_model=radar_model, hidden_dim=256)
    # Optionally, if sequential inputs are used, attach a temporal transformer:
    # model.temporal_transformer = TemporalTransformer(token_dim=512, n_layers=2, n_heads=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
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

    import os
    log_dir = os.path.dirname(args.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    with open(args.log_file, "a") as log_file:
        log_print(f"Using device: {device}", log_file)
        save_path = os.path.join(args.save_model_dir, "best_model.pth")
        train_model(model, train_loader, val_loader, optimizer, scheduler, args.epochs,
                    tuple(args.loss_weights), log_file, device, save_path)
        test_model(model, test_loader, log_file, tuple(args.loss_weights), device)

if __name__ == "__main__":
    main()
