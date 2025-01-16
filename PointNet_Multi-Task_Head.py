import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import os
from pprint import pformat  # Import pprint for better formatting

# ---------------------------
# 1) RadarDataset Class
# ---------------------------
class RadarDataset(Dataset):
    def __init__(self, input_file, label_file, train_indices_file, val_indices_file, isValidation=False):
        """
        Load radar sensing data and split into train/validation sets
        Args:
            input_file (str): Path to the inputs .npy file
            label_file (str): Path to the labels .npy file
            train_indices_file (str): Path to train indices .npy file
            val_indices_file (str): Path to validation indices .npy file
            isValidation (bool): Flag to determine train or validation split
        """
        self.inputs = np.load(input_file, allow_pickle=True)
        self.labels = np.load(label_file, allow_pickle=True)
        self.train_indices = np.load(train_indices_file, allow_pickle=True)
        self.val_indices = np.load(val_indices_file, allow_pickle=True)
        self.isValidation = isValidation

        # Print dataset details
        print(f"Loaded {len(self.inputs)} inputs, {len(self.labels)} labels")
        print(f"Training samples: {len(self.train_indices)}, Validation samples: {len(self.val_indices)}")

    def __len__(self):
        return len(self.val_indices) if self.isValidation else len(self.train_indices)

    def __getitem__(self, idx):
        ID = self.val_indices[idx] if self.isValidation else self.train_indices[idx]
        points = torch.tensor(self.inputs[ID], dtype=torch.float32)
        labels = torch.tensor(self.labels[ID], dtype=torch.float32)

        # Ensure labels have the expected shape
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        # Data Augmentation: Random Rotation (only for training)
        if not self.isValidation:
            angle = np.random.uniform(-np.pi, np.pi)
            rotation_matrix = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,             0,              1]
            ], dtype=torch.float32)
            
            points_xyz = points[:, :3]
            points_xyz = torch.matmul(points_xyz, rotation_matrix)
            points[:, :3] = points_xyz

        # Safely access target_cls
        if labels.shape[1] > 7:
            target_cls = labels[0, 7]
        else:
            # Option A: Set to 1 (assuming all are cars)
            # Option B: Derive based on some condition
            target_cls = torch.tensor(1.0, dtype=torch.float32)  # Example: All cars

        return {
            'points': points[:, [0, 1, 2]],    # X, Y, Z coordinates
            'dynamics': points[:, [3, 4, 5]], # Velocity, Range, Bearing
            'range': points[:, 4],            # Range
            'bearing': points[:, 5],          # Bearing
            'intensity': points[:, 6],        # Intensity
            'bbox_gt': labels[0, :7],         # (w, h, l, x, y, z, theta)
            'depth_gt': labels[0, 5],         # Z position (depth GT)
            'target_cls': target_cls          # Classification label
        }

# ---------------------------
# 2) PointNet++ Backbone
# ---------------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(xyz.shape[0], 1, points.shape[1], points.shape[2])
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # Use Farthest Point Sampling (FPS) to sample new_xyz
            new_xyz = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            # Grouping with radius search
            new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), new_xyz)
            new_xyz = new_xyz.transpose(1, 2).contiguous()
            idx, _ = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = pointnet2_utils.grouping_operation(xyz.transpose(1, 2).contiguous(), idx)
            grouped_xyz -= new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
            if points is not None:
                grouped_points = pointnet2_utils.grouping_operation(points.transpose(1, 2).contiguous(), idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=3)
            else:
                new_points = grouped_xyz
            new_points = new_points.permute(0, 3, 1, 2)

        # MLP layers
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 3)[0]
        return new_xyz, new_points

class PointNetPlusPlusBackbone(nn.Module):
    def __init__(self, input_dim=6, num_classes=2):
        super(PointNetPlusPlusBackbone, self).__init__()
        # Input channels: xyz (3) + dynamics (3) = 6
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=input_dim, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=128, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 1024], group_all=True)

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, N, 3]
            points: [B, N, D]
        Output:
            global_feature: [B, 1024]
        """
        l1_xyz, l1_points = self.sa1(xyz, points)   # [B, 1024, 3], [B, 128, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 256, 3], [B, 256, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 64, 3], [B, 512, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B, 1, 3], [B, 1024, 1]
        global_feature = l4_points.view(-1, 1024)
        return global_feature

# ---------------------------
# 3) Multi-Task Heads
# ---------------------------
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)  # Output shape: [B, 1]

class BoundingBoxRegressionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(BoundingBoxRegressionHead, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 7)  # (w, h, l, x, y, z, theta)
        )

    def forward(self, x):
        return self.regressor(x)  # Output shape: [B, 7]

# ---------------------------
# 4) PointNet++ with Multi-Task Heads
# ---------------------------
class PointNetPlusPlusMultiTaskModel(nn.Module):
    def __init__(self, input_dim=6, num_classes=2):
        super(PointNetPlusPlusMultiTaskModel, self).__init__()
        self.backbone = PointNetPlusPlusBackbone(input_dim=input_dim)
        self.classification_head = ClassificationHead(input_dim=1024, hidden_dim=512)
        self.regression_head = BoundingBoxRegressionHead(input_dim=1024, hidden_dim=512)

    def forward(self, xyz, dynamics):
        """
        Input:
            xyz: [B, N, 3]
            dynamics: [B, N, 3]
        Output:
            classification: [B, 1]
            bbox_pred: [B, 7]
        """
        # Combine xyz and dynamics as input features
        features = torch.cat([xyz, dynamics], dim=-1)  # [B, N, 6]
        global_feature = self.backbone(xyz, features)  # [B, 1024]
        classification = self.classification_head(global_feature)  # [B, 1]
        bbox_pred = self.regression_head(global_feature)  # [B, 7]
        return classification, bbox_pred

# ============================================
# 5) Axis-Aligned 3D IoU Utility (ignores theta)
# ============================================
def axis_aligned_iou_3d(bboxes1, bboxes2):
    """
    Compute axis-aligned IoU for 3D bounding boxes.
    Args:
        bboxes1: [B, 7] tensor (w, h, l, x, y, z, theta)
        bboxes2: [B, 7] tensor (w, h, l, x, y, z, theta)
    Returns:
        iou: [B] tensor
    """
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
# 6) Updated IoU thresholds
# ---------------------------------------
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---------------------------------------
# 7) Updated IoU Computation
# ---------------------------------------
def compute_iou(pred_boxes, gt_boxes, thresholds=IOU_THRESHOLDS):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    iou_scores = {}
    for thresh in thresholds:
        iou_scores[thresh] = (ious > thresh).float().mean().item()
    return iou_scores

# ---------------------------------------
# 8) IoU Loss Function
# ---------------------------------------
def iou_loss(pred_boxes, gt_boxes):
    """
    Compute IoU loss between predicted and ground truth boxes.
    Args:
        pred_boxes: [B, 7] tensor
        gt_boxes: [B, 7] tensor
    Returns:
        loss: scalar tensor
    """
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    return 1.0 - ious.mean()

# ---------------------------------------
# 9) Custom Loss Function with Weights
# ---------------------------------------
def custom_loss(class_pred, bbox_pred, class_gt, bbox_gt, loss_weights):
    """
    Compute combined loss for classification and bounding box regression.
    Args:
        class_pred: [B, 1] tensor
        bbox_pred: [B, 7] tensor
        class_gt: [B] tensor
        bbox_gt: [B, 7] tensor
        loss_weights: tuple (w_class, w_bbox, w_iou)
    Returns:
        total_loss: scalar tensor
    """
    w_class, w_bbox, w_iou = loss_weights
    
    # Classification Loss: Binary Cross-Entropy
    class_gt = class_gt.unsqueeze(1)  # [B, 1]
    cls_loss = F.binary_cross_entropy(class_pred, class_gt)
    
    # Bounding Box Regression Loss: Smooth L1 Loss
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_gt)
    
    # IoU Loss
    iou = iou_loss(bbox_pred, bbox_gt)
    
    # Combine losses
    total_loss = w_class * cls_loss + w_bbox * bbox_loss + w_iou * iou
    return total_loss

# ---------------------------------------
# HELPER: Logging function
# ---------------------------------------
def log_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

# ---------------------------------------
# 10) Training Function
#    Enhanced to compute validation IoU and implement Early Stopping & Checkpointing
# ---------------------------------------
def train_model(
    model, train_dataloader, val_dataloader, optimizer, 
    scheduler, epochs, batch_size, learning_rate, loss_weights, log_file, device,
    patience=20  # Patience for Early Stopping
):
    model.train()
    log_print("=== TRAINING START ===", log_file)
    log_print(f"Batch size: {batch_size}", log_file)
    log_print(f"Learning rate: {learning_rate}", log_file)
    log_print(f"Epochs: {epochs}", log_file)
    log_print(f"Loss Weights (class, bbox, iou): {loss_weights}", log_file)
    log_print(f"Device: {device}", log_file)  # Log device information
    log_print("", log_file)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
        class_correct = 0
        class_total = 0

        model.train()
        for batch in train_dataloader:
            # Move all tensors in the batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            optimizer.zero_grad()

            # Forward pass
            class_pred, bbox_pred = model(batch['points'], batch['dynamics'])

            # Compute loss
            loss = custom_loss(
                class_pred=class_pred, 
                bbox_pred=bbox_pred, 
                class_gt=batch['target_cls'], 
                bbox_gt=batch['bbox_gt'], 
                loss_weights=loss_weights
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Compute IoU
            iou_batch = compute_iou(bbox_pred, batch['bbox_gt'], thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

            # Compute Classification Accuracy
            predicted = (class_pred > 0.5).float()
            class_total += batch['target_cls'].size(0)
            class_correct += (predicted.squeeze() == batch['target_cls']).sum().item()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_iou = {key: iou_results[key] / len(train_dataloader) for key in iou_results}
        train_accuracy = class_correct / class_total

        # Validation
        val_loss, val_iou, val_accuracy = validate_model(model, val_dataloader, loss_weights, device)
        scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss - 1e-4:  # Improvement threshold
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), "./results/best_model.pth")
            log_print(f"New best model saved at epoch {epoch+1}", log_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_print(f"Early stopping triggered after {epoch+1} epochs", log_file)
                break

        end_time = time.time()
        # **Modification Made:** Split the log message into multiple lines using newline characters
        msg = (
            f"Epoch {epoch+1},\n"
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f},\n"
            f"Train IoU: {pformat(avg_train_iou)},\n"  # Using pformat for better formatting
            f"Val IoU: {pformat(val_iou)},\n"
            f"Eval_time: {end_time - start_time:.2f}s\n"
        )
        log_print(msg, log_file)
    
    log_print("=== TRAINING END ===\n", log_file)

def validate_model(model, val_dataloader, loss_weights, device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
    class_correct = 0
    class_total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            # Move all tensors in the batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Forward pass
            class_pred, bbox_pred = model(batch['points'], batch['dynamics'])

            # Compute loss
            loss = custom_loss(
                class_pred=class_pred, 
                bbox_pred=bbox_pred, 
                class_gt=batch['target_cls'], 
                bbox_gt=batch['bbox_gt'], 
                loss_weights=loss_weights
            )
            total_loss += loss.item()

            # Compute IoU
            iou_batch = compute_iou(bbox_pred, batch['bbox_gt'], thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

            # Compute Classification Accuracy
            predicted = (class_pred > 0.5).float()
            class_total += batch['target_cls'].size(0)
            class_correct += (predicted.squeeze() == batch['target_cls']).sum().item()

    avg_loss = total_loss / len(val_dataloader)
    avg_iou = {key: iou_results[key] / len(val_dataloader) for key in iou_results}
    avg_accuracy = class_correct / class_total
    return avg_loss, avg_iou, avg_accuracy

# ---------------------------------------
# 11) Testing Function
# ---------------------------------------
def test_model(model, dataloader, log_file, loss_weights, device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}
    class_correct = 0
    class_total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move all tensors in the batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Forward pass
            class_pred, bbox_pred = model(batch['points'], batch['dynamics'])

            # Compute loss
            loss = custom_loss(
                class_pred=class_pred, 
                bbox_pred=bbox_pred, 
                class_gt=batch['target_cls'], 
                bbox_gt=batch['bbox_gt'], 
                loss_weights=loss_weights
            )
            total_loss += loss.item()

            # Compute IoU
            iou_batch = compute_iou(bbox_pred, batch['bbox_gt'], thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

            # Compute Classification Accuracy
            predicted = (class_pred > 0.5).float()
            class_total += batch['target_cls'].size(0)
            class_correct += (predicted.squeeze() == batch['target_cls']).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_iou = {key: iou_results[key] / len(dataloader) for key in iou_results}
    avg_accuracy = class_correct / class_total

    msg = (
        f"Validation/Test Loss: {avg_loss:.4f}, Test Acc: {avg_accuracy:.4f},\n"
        f"IoU: {pformat(avg_iou)}"
    )
    log_print(msg, log_file)

# ---------------------------------------
# 12) Main Function
# ---------------------------------------
def main():
    # Hyperparameters
    input_file = '/home/js/Documents/GitHub/FMCW_Vehicle_Fusion/input_files_Nov_16_17/inputs.npy'
    label_file = '/home/js/Documents/GitHub/FMCW_Vehicle_Fusion/input_files_Nov_16_17/labels.npy'
    train_indices_file = '/home/js/Documents/GitHub/FMCW_Vehicle_Fusion/input_files_Nov_16_17/train_indices.npy'
    val_indices_file = '/home/js/Documents/GitHub/FMCW_Vehicle_Fusion/input_files_Nov_16_17/val_indices.npy'

    batch_size = 8
    learning_rate = 0.0005
    epochs = 900
    loss_weights = (1.0, 1.0, 1.0)  # (class_weight, bbox_weight, iou_weight)

    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Dataset
    train_dataset = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, isValidation=False)
    val_dataset   = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, isValidation=True)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize PointNet++ Multi-Task Model
    model = PointNetPlusPlusMultiTaskModel(input_dim=6, num_classes=2)  # input_dim=6 (xyz + dynamics)

    # Initialize model weights
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(initialize_weights)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model is on {device}")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Define optimizer and scheduler
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

    # Training and Testing
    with open("./results/train_test_log.txt", "a") as log_file:
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
            device=device  # Pass device to training function
        )
        # Load the best model before testing
        model.load_state_dict(torch.load("./results/best_model.pth"))
        test_model(model, val_loader, log_file, loss_weights, device)  # Pass device to test function

if __name__ == "__main__":
    main()
