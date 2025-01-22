import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import random
import argparse
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
    def __init__(self, input_dim=16):  # Updated default input_dim to 16
        """
        A lightweight feedforward attention layer for fusing predictions.
        Args:
            input_dim (int): The dimensionality of the input features.
        """
        super(AttentionFusionLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # First layer matches the concatenated input size
            nn.ReLU(),
            nn.Linear(128, 2),         # Two weights (w1, w2) for the models
            nn.Softmax(dim=-1)         # Normalize weights to sum to 1
        )

    def forward(self, features):
        """
        Forward pass for attention-based fusion.
        Args:
            features (Tensor): Concatenated features from model 1 and model 2 (batch_size, input_dim).
        Returns:
            Tensor: Attention weights for model 1 and model 2 (batch_size, 2).
        """
        weights = self.fc(features)  # Compute weights
        return weights


# ============================================
# 7) Axis-Aligned 3D IoU Utility (ignores theta)
# ============================================
def axis_aligned_iou_3d(bboxes1, bboxes2):
    """
    Compute IoU for axis-aligned 3D boxes ignoring orientation.
    Boxes are expected in the format: (w, h, l, x, y, z, theta)
    We'll ignore 'theta' and treat the box as axis-aligned.
    Returns IoU per sample as a 1D tensor of shape (batch_size,).
    """
    w1, h1, l1, cx1, cy1, cz1, _ = [bboxes1[:, i] for i in range(7)]
    w2, h2, l2, cx2, cy2, cz2, _ = [bboxes2[:, i] for i in range(7)]

    # min/max corners for bboxes1
    x1_min = cx1 - (w1 / 2.0)
    x1_max = cx1 + (w1 / 2.0)
    y1_min = cy1 - (h1 / 2.0)
    y1_max = cy1 + (h1 / 2.0)
    z1_min = cz1 - (l1 / 2.0)
    z1_max = cz1 + (l1 / 2.0)

    # min/max corners for bboxes2
    x2_min = cx2 - (w2 / 2.0)
    x2_max = cx2 + (w2 / 2.0)
    y2_min = cy2 - (h2 / 2.0)
    y2_max = cy2 + (h2 / 2.0)
    z2_min = cz2 - (l2 / 2.0)
    z2_max = cz2 + (l2 / 2.0)

    # Compute intersection along each axis
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
# 8) Updated IoU thresholds
# ---------------------------------------
# IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
IOU_THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# ---------------------------------------
# 9) Updated IoU Computation
# ---------------------------------------
def compute_iou(pred_boxes, gt_boxes, thresholds=IOU_THRESHOLDS):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    iou_scores = {}
    for thresh in thresholds:
        iou_scores[thresh] = (ious > thresh).float().mean().item()
    return iou_scores


# ---------------------------------------
# 10) IoU Loss Function
# ---------------------------------------
def iou_loss_fn(pred_boxes, gt_boxes):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    return 1.0 - ious.mean()


# ---------------------------------------
# 11) Custom Loss Function with Weights
# ---------------------------------------
def custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights):
    """
    Args:
        bbox_pred (Tensor): (batch_size, 7)
        bbox_gt   (Tensor): (batch_size, 7)
        depth_conf (Tensor): (batch_size, 1) confidence
        depth_gt  (Tensor): (batch_size,) ground truth depth or label
        loss_weights (tuple/list): (w_bbox, w_depth, w_iou)

    Returns:
        Scalar total loss
    """
    # Extract weights
    w_bbox, w_depth, w_iou = loss_weights

    # Ensure bbox_gt has the correct shape
    bbox_gt = bbox_gt.view(-1, 7)  # (batch_size, 7)

    # Smooth L1 Loss for bounding box regression
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_gt)

    # Ensure depth_conf and depth_gt have the same shape
    depth_conf = depth_conf.view(-1)  # Flatten to (batch_size,)
    depth_gt = depth_gt.view(-1)      # Flatten to (batch_size,)

    # BCE Loss for depth confidence
    depth_loss = F.binary_cross_entropy(depth_conf, depth_gt)

    # IoU Loss
    iou_loss_value = iou_loss_fn(bbox_pred, bbox_gt)

    # Weighted sum
    return w_bbox * bbox_loss + w_depth * depth_loss + w_iou * iou_loss_value


# ---------------------------------------
# HELPER: Logging function
# ---------------------------------------
def log_print(message, file_handle):
    """
    Prints the message to stdout and writes it to a file.
    """
    print(message)
    file_handle.write(message + "\n")


# ---------------------------------------
# 12) Training Function
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
    log_print(f"Loss Weights (bbox, depth, iou): {loss_weights}", log_file)
    log_print(f"Using device: {device}", log_file)
    log_print("", log_file)

    # Initialize the best validation loss
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

        # ---------- TRAINING LOOP ----------
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move data to the appropriate device
            points = batch['points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)

            bbox_pred, depth_conf = model(points, dynamics)
            loss = custom_loss(
                bbox_pred=bbox_pred,
                bbox_gt=bbox_gt,
                depth_conf=depth_conf,
                depth_gt=depth_gt,
                loss_weights=loss_weights
            )
            loss.backward()
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            # Compute IoU across thresholds for training set (optional)
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

        # Average metrics for training
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_iou = {key: iou_results[key] / len(train_dataloader) for key in iou_results}

        # ---------- VALIDATION LOOP ----------
        val_loss, val_iou = validate_model(model, val_dataloader, loss_weights, device)

        # Check if validation loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': best_val_loss
                },
                save_path
            )
            log_print(f"Saved best model with validation loss: {best_val_loss:.4f}", log_file)

        # Step the ReduceLROnPlateau scheduler **with the validation loss**
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
    """
    Validate the model on val_dataloader
    and return the average validation loss and IoU metrics.
    """
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

    with torch.no_grad():
        for batch in val_dataloader:
            # Move data to the appropriate device
            points = batch['points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)

            bbox_pred, depth_conf = model(points, dynamics)
            loss = custom_loss(
                bbox_pred=bbox_pred,
                bbox_gt=bbox_gt,
                depth_conf=depth_conf,
                depth_gt=depth_gt,
                loss_weights=loss_weights
            )
            total_loss += loss.item()
            # IoU
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
            # Move data to the appropriate device
            points = batch['points'].to(device)
            dynamics = batch['dynamics'].to(device)
            bbox_gt = batch['bbox_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)

            bbox_pred, depth_conf = model(points, dynamics)
            loss = custom_loss(
                bbox_pred=bbox_pred,
                bbox_gt=bbox_gt,
                depth_conf=depth_conf,
                depth_gt=depth_gt,
                loss_weights=loss_weights
            )
            total_loss += loss.item()
            # IoU
            iou_batch = compute_iou(bbox_pred, bbox_gt, thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

    avg_loss = total_loss / len(dataloader)
    avg_iou = {key: iou_results[key] / len(dataloader) for key in iou_results}
    msg = (f"Test Loss: {avg_loss:.4f}, IoU: {avg_iou}")
    log_print(msg, log_file)


def test_with_fusion(
    model1: nn.Module,
    model2: nn.Module,
    fusion_layer: nn.Module,
    test_loader: List[DataLoader],  # [rear_test_loader, side_test_loader]
    device: torch.device
):
    """
    Test two models with an attention-based fusion layer.
    Args:
        model1 (nn.Module): Trained rear-view model.
        model2 (nn.Module): Trained side-view model.
        fusion_layer (nn.Module): Attention-based fusion layer.
        test_loader (List[DataLoader]): [rear_test_loader, side_test_loader].
        device (torch.device): Device for computation (CPU or CUDA).
    Returns:
        tuple: (iou_scores, avg_depth_error, results)
    """
    model1.eval()
    model2.eval()
    fusion_layer.eval()

    rear_loader, side_loader = test_loader
    iou_scores = {thresh: [] for thresh in IOU_THRESHOLDS}  # IoU for each threshold
    depth_errors = []
    results = []

    with torch.no_grad():
        for rear_batch, side_batch in zip(rear_loader, side_loader):
            # Move rear inputs to device
            rear_points = rear_batch['points'].to(device)
            rear_dynamics = rear_batch['dynamics'].to(device)
            rear_bbox_gt = rear_batch['bbox_gt'].to(device)  # Ground truth for bounding boxes
            rear_depth_gt = rear_batch['depth_gt'].to(device)  # Ground truth for depth

            # Move side inputs to device
            side_points = side_batch['points'].to(device)
            side_dynamics = side_batch['dynamics'].to(device)

            # Forward pass through models
            rear_bbox_pred, rear_depth_conf = model1(rear_points, rear_dynamics)
            side_bbox_pred, side_depth_conf = model2(side_points, side_dynamics)

            # Concatenate predictions
            features = torch.cat(
                [rear_bbox_pred, rear_depth_conf, side_bbox_pred, side_depth_conf],
                dim=-1
            )  # Expected shape: (batch_size, 16)

            # Compute fusion weights
            weights = fusion_layer(features)  # Output shape: (batch_size, 2)

            # Weighted fusion
            bbox_fused = weights[:, 0:1] * rear_bbox_pred + weights[:, 1:2] * side_bbox_pred
            depth_fused = weights[:, 0:1] * rear_depth_conf + weights[:, 1:2] * side_depth_conf

            # 1. Compute IoU for fused bounding boxes across thresholds
            iou_batch = compute_iou(bbox_fused, rear_bbox_gt, thresholds=IOU_THRESHOLDS)
            for thresh, score in iou_batch.items():
                iou_scores[thresh].append(score)

            # 2. Compute depth error
            depth_error = torch.abs(depth_fused.squeeze() - rear_depth_gt)  # MAE
            depth_errors.append(depth_error.mean().item())  # Average depth error for the batch

            # Append results (bbox_fused and depth_fused for each sample)
            results.append({
                "bbox_fused": bbox_fused.cpu().numpy(),
                "depth_fused": depth_fused.cpu().numpy(),
                "iou_scores": iou_batch,  # IoU for current batch across thresholds
                "depth_error": depth_error.cpu().numpy()
            })

    # Average IoU for each threshold
    avg_iou_scores = {thresh: sum(scores) / len(scores) for thresh, scores in iou_scores.items()}

    # Average depth error across all batches
    avg_depth_error = sum(depth_errors) / len(depth_errors)

    return avg_iou_scores, avg_depth_error, results



# ---------------------------------------
# 14) Main Function
# ---------------------------------------
def main():
    # ---------------------------
    # Parse Command-Line Arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description='Radar Depth Estimation Testing with Fusion Layer')
    parser.add_argument('--rear_model_path', type=str, required=True, help='Path to the trained rear model .pth file')
    parser.add_argument('--side_model_path', type=str, required=True, help='Path to the trained side model .pth file')
    parser.add_argument('--rear_input_file', type=str, required=True, help='Path to rear test inputs.npy')
    parser.add_argument('--rear_label_file', type=str, required=True, help='Path to rear test labels.npy')
    parser.add_argument('--rear_test_indices_file', type=str, required=True, help='Path to rear test indices.npy')
    parser.add_argument('--side_input_file', type=str, required=True, help='Path to side test inputs.npy')
    parser.add_argument('--side_label_file', type=str, required=True, help='Path to side test labels.npy')
    parser.add_argument('--side_test_indices_file', type=str, required=True, help='Path to side test indices.npy')
    parser.add_argument('--log_file', type=str, default="./results/fusion_test_log.txt", help='Path to log file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    args = parser.parse_args()

    # ---------------------------
    # Set Random Seeds for Reproducibility
    # ---------------------------
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)

    # ---------------------------
    # Ensure Log Directory Exists
    # ---------------------------
    import os
    log_dir = os.path.dirname(args.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ---------------------------
    # Device Configuration
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # Datasets for Testing
    # ---------------------------
    # Rear view dataset
    rear_test_dataset = RadarDataset(
        input_file=args.rear_input_file,
        label_file=args.rear_label_file,
        train_indices_file=None,  # Not used in testing
        val_indices_file=args.rear_test_indices_file,
        isValidation=True
    )
    rear_test_loader = DataLoader(
        rear_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Side view dataset
    side_test_dataset = RadarDataset(
        input_file=args.side_input_file,
        label_file=args.side_label_file,
        train_indices_file=None,  # Not used in testing
        val_indices_file=args.side_test_indices_file,
        isValidation=True
    )
    side_test_loader = DataLoader(
        side_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # ---------------------------
    # Load Pretrained Models
    # ---------------------------
    rear_model = RadarDepthEstimationModel(radar_model=RadarModel(input_dim=3, hidden_dim=128)).to(device)
    side_model = RadarDepthEstimationModel(radar_model=RadarModel(input_dim=3, hidden_dim=128)).to(device)

    rear_model.load_state_dict(torch.load(args.rear_model_path)['model_state_dict'])
    side_model.load_state_dict(torch.load(args.side_model_path)['model_state_dict'])

    # ---------------------------
    # Initialize Fusion Layer
    # ---------------------------
    fusion_layer = AttentionFusionLayer(input_dim=16).to(device)  # 7 (bbox_pred) + 1 (depth_conf) per model

    # ---------------------------
    # Test with Fusion
    # ---------------------------
    with open(args.log_file, "a") as log_file:
        # Add a timestamp to distinguish the current evaluation
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_print(f"\n=== Evaluation Start: {current_time} ===", log_file)
        
        log_print(f"Using device: {device}", log_file)
        log_print(f"Testing with rear model: {args.rear_model_path}", log_file)
        log_print(f"Testing with side model: {args.side_model_path}", log_file)

        avg_iou_scores, avg_depth_error, results = test_with_fusion(
            model1=rear_model,
            model2=side_model,
            fusion_layer=fusion_layer,
            test_loader=[rear_test_loader, side_test_loader],
            device=device
        )

        # Log IoU scores for each threshold
        log_print(f"Average IoU scores across thresholds:", log_file)
        for thresh, avg_iou in avg_iou_scores.items():
            log_print(f"  Threshold {thresh:.1f}: {avg_iou:.4f}", log_file)

        # Log the average depth error
        log_print(f"Average Depth Error (MAE): {avg_depth_error:.4f} meters", log_file)

        # Optionally log detailed results for debugging or analysis
        for idx, result in enumerate(results):
            log_print(f"Batch {idx + 1}: Fused IoU Scores: {result['iou_scores']}, Depth Error: {result['depth_error']}", log_file)

        # Mark the end of the evaluation with another timestamp
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_print(f"=== Evaluation End: {end_time} ===\n", log_file)



if __name__ == "__main__":
    main()
