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

        return {
            'points': points[:, [0, 1, 2]],    # X, Y, Z coordinates
            'dynamics': points[:, [3, 4, 5]], # Velocity, Range, Bearing
            'range': points[:, 4],            # Range
            'bearing': points[:, 5],          # Bearing
            'intensity': points[:, 6],        # Intensity
            'bbox_gt': labels[0, :7],         # (w, h, l, x, y, z, theta)
            'depth_gt': labels[0, 5]          # Z position (depth GT)
        }

# ---------------------------
# 2) TransformerBlock Class
#    Increased number of heads and layers
# ---------------------------
class TransformerBlock(nn.Module):
    """
    An enhanced transformer-like block with:
    - Multi-head self-attention
    - Residual connections
    - Feed-forward MLP + another residual
    - LayerNorm and dropout for better training stability
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):  # Increased num_heads to 4
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # Expanded MLP capacity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: (batch_size, num_points, embed_dim)
        """
        # Multi-head attention (self-attention)
        attn_out, _ = self.mha(x, x, x)  # q, k, v = x
        # Residual + LayerNorm
        x = self.layernorm1(x + self.dropout(attn_out))

        # Feed-forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))
        return x

# ---------------------------
# 3) DepthEstimationSubnet Class
#    Expanded layers to handle increased hidden_dim
# ---------------------------
class DepthEstimationSubnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Increased layer sizes
            nn.ReLU(),
            nn.Dropout(0.2),              # Reduced dropout to retain more information
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs confidence score between 0 and 1
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------------
# 4) BoundingBoxDecoder Class
#    Expanded layers to handle increased hidden_dim
# ---------------------------
class BoundingBoxDecoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),  # Increased layer sizes
            nn.ReLU(),
            nn.Dropout(0.2),              # Reduced dropout
            nn.Linear(2048, 1024),
            nn.ReLU(),
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
# 5) RadarModel Class
#    Increased hidden_dim and transformer layers
# ---------------------------
class RadarModel(nn.Module):
    def __init__(
        self,
        input_dim=3,
        dynamic_dim=3,
        hidden_dim=512,            # Increased from 256 to 512
        num_heads=4,               # Increased from 1 to 4
        num_transformer_layers=4  # Increased from 2 to 4
    ):
        """
        - input_dim: dimension of (x, y, z)
        - dynamic_dim: dimension of (velocity, range, bearing)
        - hidden_dim: dimension for each encoder's output
        - num_heads: number of heads for multi-head attention
        - num_transformer_layers: how many TransformerBlock layers to stack
        """
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # Expanded encoder layers
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)  # Now 512
        )
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, 128),  # Expanded encoder layers
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)  # Now 512
        )

        # Combine point_features + dynamic_features => hidden_dim * 2 => 1024
        embed_dim = hidden_dim * 2

        # Enhanced multi-head transformer blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, dropout=0.1)
            for _ in range(num_transformer_layers)
        ])

    def forward(self, points, dynamics):
        batch_size, num_points, _ = points.shape

        # Flatten and encode
        points_flat = points.reshape(-1, points.shape[-1])       # (batch_size*num_points, 3)
        dynamics_flat = dynamics.reshape(-1, dynamics.shape[-1]) # (batch_size*num_points, 3)

        point_features = self.point_encoder(points_flat)         # => (batch_size*num_points, 512)
        dynamic_features = self.dynamic_encoder(dynamics_flat)   # => (batch_size*num_points, 512)

        # Combine
        combined = torch.cat([point_features, dynamic_features], dim=1)  # => (batch_size*num_points, 1024)
        combined = combined.view(batch_size, num_points, -1)             # => (batch_size, num_points, 1024)

        # Pass through multiple TransformerBlock layers
        x = combined
        for layer in self.transformer_layers:
            x = layer(x)

        return x  # final shape => (batch_size, num_points, 1024)

# ---------------------------
# 6) RadarDepthEstimationModel Class
#    Full Radar Depth Estimation Model
# ---------------------------
class RadarDepthEstimationModel(nn.Module):
    def __init__(self, radar_model, hidden_dim=512):
        super().__init__()
        # radar_model returns (batch_size, num_points, hidden_dim * 2) => 1024
        self.radar_model = radar_model
        self.depth_subnet = DepthEstimationSubnet(hidden_dim * 2)  # 1024
        self.decoder = BoundingBoxDecoder(hidden_dim * 2)          # 1024

    def forward(self, points, dynamics):
        attn_out = self.radar_model(points, dynamics)  # => (batch_size, num_points, 1024)
        # Global average pooling over points dimension
        pooled_features = attn_out.mean(dim=1)         # => (batch_size, 1024)

        depth_confidence = self.depth_subnet(pooled_features)  # => (batch_size, 1)
        bbox_prediction = self.decoder(pooled_features)        # => (batch_size, 7)
        return bbox_prediction, depth_confidence

# ============================================
# 7) Axis-Aligned 3D IoU Utility (ignores theta)
# ============================================
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
# 8) Updated IoU thresholds
# ---------------------------------------
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
def iou_loss(pred_boxes, gt_boxes):
    ious = axis_aligned_iou_3d(pred_boxes, gt_boxes)
    return 1.0 - ious.mean()

# ---------------------------------------
# 11) Custom Loss Function with Weights
# ---------------------------------------
def custom_loss(bbox_pred, bbox_gt, depth_conf, depth_gt, loss_weights):
    w_bbox, w_depth, w_iou = loss_weights
    
    bbox_gt = bbox_gt.view(-1, 7)
    bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_gt)
    
    # **Change Made:** Replace BCE loss with Smooth L1 Loss for continuous depth targets
    depth_loss = F.smooth_l1_loss(depth_conf.squeeze(), depth_gt)
    
    iou_loss_value = iou_loss(bbox_pred, bbox_gt)

    return w_bbox * bbox_loss + w_depth * depth_loss + w_iou * iou_loss_value

# ---------------------------------------
# HELPER: Logging function
# ---------------------------------------
def log_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

# ---------------------------------------
# 12) Training Function
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
    log_print(f"Loss Weights (bbox, depth, iou): {loss_weights}", log_file)
    log_print(f"Device: {device}", log_file)  # Log device information
    log_print("", log_file)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

        model.train()
        for batch in train_dataloader:
            # Move all tensors in the batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            optimizer.zero_grad()

            bbox_pred, depth_conf = model(batch['points'], batch['dynamics'])
            loss = custom_loss(
                bbox_pred=bbox_pred, 
                bbox_gt=batch['bbox_gt'], 
                depth_conf=depth_conf, 
                depth_gt=batch['depth_gt'], 
                loss_weights=loss_weights
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            iou_batch = compute_iou(bbox_pred, batch['bbox_gt'], thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_iou = {key: iou_results[key] / len(train_dataloader) for key in iou_results}

        # Validation
        val_loss, val_iou = validate_model(model, val_dataloader, loss_weights, device)
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
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f},\n"
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
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Move all tensors in the batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            bbox_pred, depth_conf = model(batch['points'], batch['dynamics'])
            loss = custom_loss(
                bbox_pred=bbox_pred,
                bbox_gt=batch['bbox_gt'],
                depth_conf=depth_conf,
                depth_gt=batch['depth_gt'],
                loss_weights=loss_weights
            )
            total_loss += loss.item()

            # Compute IoU for validation
            iou_batch = compute_iou(bbox_pred, batch['bbox_gt'], thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]
    
    avg_loss = total_loss / len(val_dataloader)
    avg_iou = {key: iou_results[key] / len(val_dataloader) for key in iou_results}
    return avg_loss, avg_iou

# ---------------------------------------
# 13) Testing Function
# ---------------------------------------
def test_model(model, dataloader, log_file, loss_weights, device):
    model.eval()
    total_loss = 0.0
    iou_results = {t: 0.0 for t in IOU_THRESHOLDS}

    with torch.no_grad():
        for batch in dataloader:
            # Move all tensors in the batch to the specified device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            bbox_pred, depth_conf = model(batch['points'], batch['dynamics'])
            loss = custom_loss(
                bbox_pred=bbox_pred,
                bbox_gt=batch['bbox_gt'],
                depth_conf=depth_conf,
                depth_gt=batch['depth_gt'],
                loss_weights=loss_weights
            )
            total_loss += loss.item()
            iou_batch = compute_iou(bbox_pred, batch['bbox_gt'], thresholds=IOU_THRESHOLDS)
            for key in iou_results:
                iou_results[key] += iou_batch[key]

    avg_iou = {key: iou_results[key] / len(dataloader) for key in iou_results}
    msg = (
        f"Validation/Test Loss: {total_loss / len(dataloader):.4f},\n"
        f"IoU: {pformat(avg_iou)}"
    )
    log_print(msg, log_file)

# ---------------------------------------
# 14) Main Function
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
    loss_weights = (0.60, 0.10, 0.30)  # (bbox_weight, depth_weight, iou_weight)

    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Dataset
    train_dataset = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, isValidation=False)
    val_dataset   = RadarDataset(input_file, label_file, train_indices_file, val_indices_file, isValidation=True)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)  # Added: num_workers and pin_memory
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  # Added: num_workers and pin_memory

    # Enhanced Radar Model w/ multi-head transformer, hidden_dim=512, more layers
    radar_model = RadarModel(
        input_dim=3,
        dynamic_dim=3,
        hidden_dim=512,      # increased from 256 to 512
        num_heads=4,         # increased from 1 to 4
        num_transformer_layers=4  # increased from 2 to 4
    )
    # Now RadarDepthEstimationModel also uses hidden_dim=512
    model = RadarDepthEstimationModel(radar_model=radar_model, hidden_dim=512)

    # Initialize model weights
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
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
