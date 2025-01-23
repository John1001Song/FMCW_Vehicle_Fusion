import torch
from torchviz import make_dot
from torchsummary import summary

# Import models from middle_fusion
from middle_fusion import RadarModel, RadarDepthEstimationModel

# Define constants for the model
INPUT_DIM = 3  # For points
DYNAMIC_DIM = 3  # For dynamics
HIDDEN_DIM = 128

def visualize_high_level_model():
    # Initialize the models
    radar_model = RadarModel(input_dim=INPUT_DIM, dynamic_dim=DYNAMIC_DIM, hidden_dim=HIDDEN_DIM)
    model = RadarDepthEstimationModel(radar_model=radar_model, hidden_dim=HIDDEN_DIM)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print a high-level textual summary
    print("High-Level Model Summary:")
    print("RadarDepthEstimationModel:")
    print(f"  ├── RadarModel: {radar_model}")
    print(f"  ├── DepthEstimationSubnet: {model.depth_subnet}")
    print(f"  └── BoundingBoxDecoder: {model.decoder}")

    # Create dummy inputs for visualization
    dummy_points = torch.randn(1, 16, INPUT_DIM).to(device)  # (batch_size, num_points, input_dim)
    dummy_dynamics = torch.randn(1, 16, DYNAMIC_DIM).to(device)  # (batch_size, num_points, dynamic_dim)

    # Forward pass to generate outputs
    bbox_pred, depth_conf = model(dummy_points, dummy_dynamics)

    # Pass high-level outputs for visualization
    outputs = (bbox_pred, depth_conf)
    graph = make_dot(outputs, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)

    # Save the high-level visualization to a file
    graph.render("high_level_radar_model_architecture", format="png", cleanup=True)

    print("High-level model architecture visualization saved as 'high_level_radar_model_architecture.png'.")

if __name__ == "__main__":
    visualize_high_level_model()
