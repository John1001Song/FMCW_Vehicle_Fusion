import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import random
import argparse
from typing import List, Dict

# ---------------------------
# 1) RadarDataset Class (unchanged)
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
            return 0 if self.test_indices is None else len(self.test_indices)
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
            'points': points[:, [0, 1, 2]],   # X, Y, Z coordinates
            'dynamics': points[:, [3, 4, 5]],   # Velocity, Range, Bearing
            'range': points[:, 4],
            'bearing': points[:, 5],
            'intensity': points[:, 6],
            'bbox_gt': labels[0, :7],
            'depth_gt': labels[0, 5]
        }

# ---------------------------
# 2) Self-Attention Module (unchanged)
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
# 3) Depth Estimation Subnet (unchanged)
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
            nn.Sigmoid()  # Outputs a value between 0 and 1
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------------
# 3a) Intensity Estimation Subnet (new)
# ---------------------------
class IntensityEstimationSubnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict a continuous intensity value
            # Optionally, you could add an activation if your target is normalized.
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------------
# 4) Bounding Box Decoder (unchanged)
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
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------------
# 5) Core Radar Model (modified with increased depth and attention on features)
# ---------------------------
class RadarModel(nn.Module):
    def __init__(self, input_dim=3, dynamic_dim=3, hidden_dim=256):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fusion = nn.Linear(512, hidden_dim)
        self.attention = SelfAttentionModule(hidden_dim)

    def forward(self, points, dynamics):
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, points.shape[-1])
        dynamics_flat = dynamics.view(-1, dynamics.shape[-1])
        p_features = self.point_encoder(points_flat)
        d_features = self.dynamic_encoder(dynamics_flat)
        combined = torch.cat([p_features, d_features], dim=1)
        fused = self.fusion(combined)
        fused = fused.view(batch_size, num_points, -1)
        attended_features = self.attention(fused)
        return attended_features

# ---------------------------
# 6) Full Radar Depth Estimation Model (modified)
# ---------------------------
class RadarDepthEstimationModel(nn.Module):
    def __init__(self, radar_model, hidden_dim=128):
        super().__init__()
        self.radar_model = radar_model
        self.depth_subnet = DepthEstimationSubnet(hidden_dim)
        self.decoder = BoundingBoxDecoder(hidden_dim)
        # New intensity subnet to predict intensity from pooled features.
        self.intensity_subnet = IntensityEstimationSubnet(hidden_dim)
    
    def forward(self, points, dynamics):
        features = self.radar_model(points, dynamics)
        pooled = features.mean(dim=1)  # Global average pooling over points.
        depth_conf = self.depth_subnet(pooled)
        bbox_pred = self.decoder(pooled)
        intensity_pred = self.intensity_subnet(pooled)
        return bbox_pred, depth_conf, intensity_pred

# ---------------------------------------
# 7) Axis-Aligned 3D IoU Utility (unchanged)
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
    iou = intersection / (union + 1e-6)
    return iou

# ---------------------------------------
# 8) Updated IoU thresholds (unchanged)
# ---------------------------------------
IOU_THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# ---------------------------------------
# 9) Updated IoU Computation (unchanged)
# ---------------------------------------
def compute_iou(pred_boxes, gt_boxes, thresholds=IOU_THRESHOLDS):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    iou_scores = {}
    for thresh in thresholds:
        iou_scores[thresh] = (ious > thresh).float().mean().item()
    return iou_scores

# ---------------------------------------
# 10) IoU Loss Function (unchanged)
# ---------------------------------------
def iou_loss_fn(pred_boxes, gt_boxes):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    return 1.0 - ious.mean()

# ---------------------------------------
# 11) Custom Loss Function with Four Weights (modified)
# ---------------------------------------
def custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, intensity_pred, intensity_gt, loss_weights):
    """
    loss_weights: (w_bbox, w_depth, w_iou, w_intensity)
    """
    w_bbox, w_depth, w_iou, w_intensity = loss_weights
    bbox_gt = bbox_gt.view(-1, 7)
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_gt)
    
    depth_conf = depth_conf.view(-1)
    depth_gt = depth_gt.view(-1)
    depth_loss = F.binary_cross_entropy(depth_conf, depth_gt)
    
    intensity_loss = F.mse_loss(intensity_pred.view(-1), intensity_gt.view(-1))
    
    iou_loss_value = iou_loss_fn(bbox_pred, bbox_gt)
    return w_bbox * bbox_loss + w_depth * depth_loss + w_iou * iou_loss_value + w_intensity * intensity_loss

# ---------------------------------------
# HELPER: Logging function (unchanged)
# ---------------------------------------
def log_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

# ---------------------------------------
# 12) Training Function (modified to account for intensity prediction)
# ---------------------------------------
def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_weights: tuple,
    log_file,
    device: torch.device,
    save_path: str  # Path to save the best model
):
    log_print("=== TRAINING START ===", log_file)
    log_print(f"Batch size: {batch_size}", log_file)
    log_print(f"Learning rate: {learning_rate}", log_file)
    log_print(f"Epochs: {epochs}", log_file)
    log_print(f"Loss Weights (bbox, depth, iou, intensity): {loss_weights}", log_file)
    log_print(f"Using device: {device}", log_file)
    log_print("", log_file)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            points = batch['points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            # Here, we compute the ground truth intensity as the average of the intensity channel.
            # You may precompute this in your dataset if available.
            intensity_gt = batch['intensity'].mean(dim=1).unsqueeze(1).to(device)
            bbox_pred, depth_conf, intensity_pred = model(points, dynamics)
            loss = custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, intensity_pred, intensity_gt, loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_iou = {key: iou_results[key] / len(train_dataloader) for key in iou_results}
        val_loss, val_iou = validate_model(model, val_dataloader, loss_weights, device)
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
        end_time = time.time()
        msg = (f"Epoch {epoch+1}/{epochs}, "
               f"Train Loss: {avg_train_loss:.4f}, "
               f"Val Loss: {val_loss:.4f}, "
               f"Train IoU: {avg_train_iou}, "
               f"Val IoU: {val_iou}, "
               f"Eval_time: {end_time - start_time:.2f}s")
        log_print(msg, log_file)
    log_print("=== TRAINING END ===\n", log_file)

def validate_model(model: nn.Module, val_dataloader: DataLoader, loss_weights: tuple, device: torch.device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
    with torch.no_grad():
        for batch in val_dataloader:
            points = batch['points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            intensity_gt = batch['intensity'].mean(dim=1).unsqueeze(1).to(device)
            bbox_pred, depth_conf, intensity_pred = model(points, dynamics)
            loss = custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, intensity_pred, intensity_gt, loss_weights)
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
    avg_loss = total_loss / len(val_dataloader)
    avg_iou = {key: iou_results[key] / len(val_dataloader) for key in iou_results}
    return avg_loss, avg_iou

def test_model(model: nn.Module, dataloader: DataLoader, log_file, loss_weights: tuple, device: torch.device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
    with torch.no_grad():
        for batch in dataloader:
            points = batch['points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            intensity_gt = batch['intensity'].mean(dim=1).unsqueeze(1).to(device)
            bbox_pred, depth_conf, intensity_pred = model(points, dynamics)
            loss = custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, intensity_pred, intensity_gt, loss_weights)
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
    avg_loss = total_loss / len(dataloader)
    avg_iou = {key: iou_results[key] / len(dataloader) for key in iou_results}
    msg = (f"Test Loss: {avg_loss:.4f}, IoU: {avg_iou}")
    log_print(msg, log_file)

# ---------------------------------------
# 14) Main Function
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Radar Depth Estimation Training Script')
    parser.add_argument('--input_file', type=str, required=True, help='Path to inputs.npy')
    parser.add_argument('--label_file', type=str, required=True, help='Path to labels.npy')
    parser.add_argument('--train_indices_file', type=str, required=True, help='Path to train_indices.npy')
    parser.add_argument('--val_indices_file', type=str, required=True, help='Path to val_indices.npy')
    parser.add_argument('--test_indices_file', type=str, required=True, help='Path to test_indices.npy')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    # Now expect 4 loss weights: for bbox, depth, IoU, and intensity.
    parser.add_argument('--loss_weights', type=float, nargs=4, default=[0.60, 0.10, 0.30, 0.05],
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

    input_file = args.input_file
    label_file = args.label_file
    train_indices_file = args.train_indices_file
    val_indices_file = args.val_indices_file
    test_indices_file = args.test_indices_file

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    loss_weights = tuple(args.loss_weights)

    train_dataset = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, test_indices_file, split="train")
    val_dataset = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, test_indices_file, split="val")
    test_dataset = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, test_indices_file, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    radar_model = RadarModel(input_dim=3, dynamic_dim=3, hidden_dim=128)
    model = RadarDepthEstimationModel(radar_model=radar_model, hidden_dim=128)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
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
        
        train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_weights=loss_weights,
            log_file=log_file,
            device=device,
            save_path=save_path
        )

        test_model(model, test_loader, log_file, loss_weights, device)

if __name__ == "__main__":
    main()
