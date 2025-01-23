import re
import matplotlib.pyplot as plt
import pandas as pd

# File paths
log_files = {
    "rear_original": "log_rear.txt",
    "rear_updated": "log_rear_jv.txt",
    "side_original": "log_side.txt",
    "side_updated": "log_side_jv.txt",
    "combined_original": "log_combined.txt",
    "combined_updated": "log_combined_jv.txt"
}

# Metrics to extract
metrics = ["Val Loss", "IoU @ 0.1", "IoU @ 0.5", "IoU @ 0.9"]

def extract_metrics(file_path):
    """Extract validation loss and IoU metrics from a log file."""
    metrics_data = {metric: [] for metric in metrics}
    epoch_pattern = re.compile(r"Epoch \d+/")
    
    with open(file_path, "r") as f:
        for line in f:
            # Extract Val Loss
            if "Val Loss:" in line:
                val_loss = float(re.search(r"Val Loss: ([0-9.]+)", line).group(1))
                metrics_data["Val Loss"].append(val_loss)
            
            # Extract IoU
            if "Val IoU:" in line:
                iou_values = re.findall(r"0\.\d+", line)
                metrics_data["IoU @ 0.1"].append(float(iou_values[-1]))
                metrics_data["IoU @ 0.5"].append(float(iou_values[4]))
                metrics_data["IoU @ 0.9"].append(float(iou_values[0]))

    return metrics_data

def plot_and_save_metric(metric_name, data, title, filename):
    """Plot a given metric across datasets and save it as a PNG file."""
    plt.figure(figsize=(10, 6))
    for model_version, values in data.items():
        plt.plot(range(1, len(values) + 1), values, label=model_version)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

# Read and process all log files
data_per_metric = {metric: {} for metric in metrics}

for key, file_path in log_files.items():
    dataset, version = key.split("_")
    metrics_data = extract_metrics(file_path)
    for metric, values in metrics_data.items():
        model_key = f"{dataset}_{version}"
        data_per_metric[metric][model_key] = values

# Plot and save each metric
for metric in metrics:
    filename = f"comparison_{metric.replace(' ', '_').lower()}.png"
    plot_and_save_metric(
        metric_name=metric,
        data=data_per_metric[metric],
        title=f"Comparison of {metric} Across Datasets and Models",
        filename=filename
    )
