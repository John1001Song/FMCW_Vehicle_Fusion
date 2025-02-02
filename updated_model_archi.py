import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_model_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Function to draw a box with text
    def draw_box(x, y, text, width=2.5, height=1.0, fontsize=10, bold=False):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                      edgecolor="black", facecolor="lightblue", linewidth=2)
        ax.add_patch(rect)
        fontweight = "bold" if bold else "normal"
        ax.text(x + width / 2, y + height / 2, text, fontsize=fontsize, fontweight=fontweight,
                ha="center", va="center")

    # Input Data
    draw_box(1, 8, "Radar Input Data\n(X, Y, Z, Vel, Range, Bearing)", bold=True)

    # Multi-Scale Feature Extraction
    draw_box(3.5, 7, "Multi-Scale Feature Extraction\n(1D Conv: 1x1, 3x3, 5x5)")

    # Multi-Head Attention
    draw_box(6, 6, "Multi-Head Attention\n(4 Heads)")

    # Attention Pooling
    draw_box(7, 5, "Improved Attention Pooling\n(Weighted Sum)")

    # Two Output Paths
    draw_box(8.5, 4, "Bounding Box Decoder\n(MLP: 512 → 256 → 7)")
    draw_box(8.5, 2.5, "Depth Estimation Subnet\n(MLP: 512 → 256 → 1)")

    # Final Outputs
    draw_box(10, 4, "3D Bounding Box Output\n(w, h, l, x, y, z, θ)", width=2)
    draw_box(10, 2.5, "Depth Confidence Output\n(Sigmoid Activation)", width=2)

    # Arrows to show flow
    arrowprops = dict(arrowstyle="->", linewidth=2, color="black")
    ax.annotate("", xy=(3.5, 7.5), xytext=(2.5, 8), arrowprops=arrowprops)  # Input → Feature Extraction
    ax.annotate("", xy=(6, 6.5), xytext=(5.5, 7), arrowprops=arrowprops)  # Feature Extraction → Attention
    ax.annotate("", xy=(7, 5.5), xytext=(6.5, 6), arrowprops=arrowprops)  # Attention → Pooling
    ax.annotate("", xy=(8.5, 4.5), xytext=(7.5, 5), arrowprops=arrowprops)  # Pooling → Bounding Box Decoder
    ax.annotate("", xy=(8.5, 3), xytext=(7.5, 5), arrowprops=arrowprops)  # Pooling → Depth Subnet
    ax.annotate("", xy=(10, 4.5), xytext=(9.5, 4), arrowprops=arrowprops)  # BBox Decoder → Output
    ax.annotate("", xy=(10, 3), xytext=(9.5, 2.5), arrowprops=arrowprops)  # Depth Subnet → Depth Output

    # Title
    plt.title("Radar-Based 3D Object Detection Model Architecture", fontsize=14, fontweight="bold")

    # Save and Show
    plt.savefig("radar_model_architecture_updated.png", dpi=300, bbox_inches='tight')
    plt.show()

# Generate the diagram
draw_model_architecture()
