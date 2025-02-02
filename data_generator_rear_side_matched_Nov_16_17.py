import numpy as np
import copy
import os
import json
import pandas as pd
from collections import defaultdict

"""
Generate *separate* rear + side datasets, each saved in its own folder,
but ensure they have the same sample ordering and the same train/val splits.

After this script, you'll have:

  input_files_rear_Nov_16_17/
    - inputs.npy    (shape: (N, 70, 8))
    - labels.npy    (shape: (N, 2, 7))
    - train_indices.npy
    - val_indices.npy

  input_files_side_Nov_16_17/
    - inputs.npy    (shape: (N, 70, 8))
    - labels.npy    (shape: (N, 2, 7))
    - train_indices.npy
    - val_indices.npy

where N is the number of frames that exist in both vantage points
(rear_dict ∩ side_dict).
"""


# ----------------------------------------------------------------
# 1) Helpers to load bounding box from JSON
# ----------------------------------------------------------------

def get_bbox_from_json_rear(frame_idx, scene_no, base_dir):
    """
    e.g. label file:  {base_dir}/scene{scene_no}/label/frame{frame_idx}.json
    returns (labels_padded, labels_orig)
    """
    labels_dir = f"{base_dir}/scene{scene_no}/label"
    json_path  = f"{labels_dir}/frame{frame_idx}.json"

    with open(json_path) as f:
        data = json.load(f)
    b = data['labels'][0]
    center = [b['center']['x'], b['center']['y'], b['center']['z']]
    l = b['size']['l']
    h = b['size']['h']
    w = b['size']['w']
    theta = b['bearing']

    single_label = np.array([w, h, l, center[0], center[1], center[2], theta]).reshape(1, -1)  # (1,7)
    labels_padded = np.pad(single_label, ((2 - single_label.shape[0], 0), (0, 0)), mode='edge')  # (2,7)
    return labels_padded, single_label


def get_bbox_from_json_side(frame_idx, scene_no, base_dir):
    """
    e.g. label file:  {base_dir}/scene{scene_no}/label_side/frame{frame_idx}_side.json
    returns (labels_padded, labels_orig)
    """
    labels_dir = f"{base_dir}/scene{scene_no}/label_side"
    json_path  = f"{labels_dir}/frame{frame_idx}_side.json"

    with open(json_path) as f:
        data = json.load(f)
    b = data['labels'][0]
    center = [b['center']['x'], b['center']['y'], b['center']['z']]
    l = b['size']['l']
    h = b['size']['h']
    w = b['size']['w']
    theta = b['bearing']

    single_label = np.array([w, h, l, center[0], center[1], center[2], theta]).reshape(1, -1)  # (1,7)
    labels_padded = np.pad(single_label, ((2 - single_label.shape[0], 0), (0, 0)), mode='edge')  # (2,7)
    return labels_padded, single_label


# ----------------------------------------------------------------
# 2) Collect REAR data in a dict: key = (day, scene_no, frame_idx)
# ----------------------------------------------------------------
def collect_rear_data():
    rear_dict = {}

    base_dirs = [
        "/home/jinyues/pointillism/pointillism/Nov_16_Processed_with_Combined",
        "/home/jinyues/pointillism/pointillism/Nov_17_Processed_with_Combined"
    ]

    subfolders_per_base_dir = {
        "/home/jinyues/pointillism/pointillism/Nov_16_Processed_with_Combined": ["Dell1", "Razr3", "Razr4", "Razr5", "Razr6"],
        "/home/jinyues/pointillism/pointillism/Nov_17_Processed_with_Combined": ["Dell1", "Dell2", "Dell3", "Dell4", "Dell5", "Dell6"]
    }

    def get_scene_indices(base_path):
        if "Nov_16" in base_path:
            return [1, 3, 4, 5, 6]
        else:
            return [1, 2, 3, 4, 5, 6]

    nPointsperFrame = 70
    minPointsperFrame = 1

    for base_dir in base_dirs:
        day_label = "Nov16" if "Nov_16" in base_dir else "Nov17"
        subfolders = subfolders_per_base_dir.get(base_dir, [])
        scene_list = get_scene_indices(base_dir)

        for scene_no in scene_list:
            test_dir = f"{base_dir}/scene{scene_no}"
            label_dir = f"{test_dir}/label"
            if not os.path.exists(label_dir):
                continue

            for subfolder in subfolders:
                csv_folder = f"{test_dir}/{subfolder}"
                if not os.path.exists(csv_folder):
                    continue

                csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
                frame_indices = sorted([
                    int(f.replace("frame","").replace(".csv","")) for f in csv_files if "frame" in f
                ])

                for frame_idx in frame_indices:
                    json_path = f"{label_dir}/frame{frame_idx}.json"
                    csv_path  = f"{csv_folder}/frame{frame_idx}.csv"

                    if not os.path.exists(json_path) or not os.path.exists(csv_path):
                        continue

                    # Labels
                    try:
                        labels_padded, labels_orig = get_bbox_from_json_rear(frame_idx, scene_no, base_dir)
                    except:
                        continue

                    # CSV
                    try:
                        df = pd.read_csv(csv_path)
                        frame_data = df.values[:, [2,3,4,5,6,7,8,9]]
                    except:
                        continue

                    # Enforce 70 points
                    n_points = frame_data.shape[0]
                    if n_points >= nPointsperFrame:
                        frame_points = frame_data[:nPointsperFrame]
                    elif n_points >= minPointsperFrame:
                        needed = nPointsperFrame - n_points
                        repeat_idx = np.random.choice(n_points, needed, replace=True)
                        repeated = frame_data[repeat_idx,:]
                        frame_points = np.concatenate([frame_data, repeated], axis=0)
                    else:
                        continue

                    key = (day_label, scene_no, frame_idx)
                    rear_dict[key] = {
                        "points": frame_points,   # (70,8)
                        "labels": labels_padded   # (2,7)
                    }
    return rear_dict


# ----------------------------------------------------------------
# 3) Collect SIDE data similarly
# ----------------------------------------------------------------
def collect_side_data():
    side_dict = {}

    base_dirs = [
        "/home/jinyues/pointillism/pointillism/Nov_16_Processed_with_Combined",
        "/home/jinyues/pointillism/pointillism/Nov_17_Processed_with_Combined"
    ]

    subfolders_per_base_dir = {
        "/home/jinyues/pointillism/pointillism/Nov_16_Processed_with_Combined": ["Razr1", "Dell3", "Dell4", "Dell5", "Dell6"],
        "/home/jinyues/pointillism/pointillism/Nov_17_Processed_with_Combined": ["Razr1", "Razr2", "Razr3", "Razr4", "Razr5", "Razr6"]
    }

    def get_scene_indices(base_path):
        if "Nov_16" in base_path:
            return [1, 3, 4, 5, 6]
        else:
            return [1, 2, 3, 4, 5, 6]

    nPointsperFrame = 70
    minPointsperFrame = 1

    for base_dir in base_dirs:
        day_label = "Nov16" if "Nov_16" in base_dir else "Nov17"
        subfolders = subfolders_per_base_dir.get(base_dir, [])
        scene_list = get_scene_indices(base_dir)

        for scene_no in scene_list:
            test_dir = f"{base_dir}/scene{scene_no}"
            label_side_dir = f"{test_dir}/label_side"
            if not os.path.exists(label_side_dir):
                continue

            for subfolder in subfolders:
                csv_folder = f"{test_dir}/{subfolder}"
                if not os.path.exists(csv_folder):
                    continue

                csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
                frame_indices = sorted([
                    int(f.replace("frame","").replace(".csv","")) for f in csv_files if "frame" in f
                ])

                for frame_idx in frame_indices:
                    json_path = f"{label_side_dir}/frame{frame_idx}_side.json"
                    csv_path  = f"{csv_folder}/frame{frame_idx}.csv"

                    if not os.path.exists(json_path) or not os.path.exists(csv_path):
                        continue

                    # Labels
                    try:
                        labels_padded, labels_orig = get_bbox_from_json_side(frame_idx, scene_no, base_dir)
                    except:
                        continue

                    # CSV
                    try:
                        df = pd.read_csv(csv_path)
                        frame_data = df.values[:, [2,3,4,5,6,7,8,9]]
                    except:
                        continue

                    # Enforce 70 points
                    n_points = frame_data.shape[0]
                    if n_points >= nPointsperFrame:
                        frame_points = frame_data[:nPointsperFrame]
                    elif n_points >= minPointsperFrame:
                        needed = nPointsperFrame - n_points
                        repeat_idx = np.random.choice(n_points, needed, replace=True)
                        repeated = frame_data[repeat_idx,:]
                        frame_points = np.concatenate([frame_data, repeated], axis=0)
                    else:
                        continue

                    key = (day_label, scene_no, frame_idx)
                    side_dict[key] = {
                        "points": frame_points,   # (70,8)
                        "labels": labels_padded   # (2,7)
                    }
    return side_dict


# ----------------------------------------------------------------
# 4) Main: merge, split, and save to separate folders
# ----------------------------------------------------------------
def main():
    rear_dict = collect_rear_data()
    side_dict = collect_side_data()

    print(f"Collected {len(rear_dict)} rear frames, {len(side_dict)} side frames.")

    # 4A) Find intersection of keys
    common_keys = list(set(rear_dict.keys()) & set(side_dict.keys()))
    print(f"Common frames that have BOTH rear and side: {len(common_keys)}")

    # We want to do a per-scene split => group by (day, scene_no)
    scene2keys = {}
    for key in common_keys:
        day_label, scene_no, frame_idx = key
        scene_id = (day_label, scene_no)
        if scene_id not in scene2keys:
            scene2keys[scene_id] = []
        scene2keys[scene_id].append(key)

    # We'll accumulate train_keys, val_keys from each scene
    train_keys = []
    val_keys   = []

    # 10% validation from each scene
    val_ratio = 0.1

    # 4B) Convert to a list, shuffle
    np.random.seed(42)
    for scene_id, keys_list in scene2keys.items():
        # shuffle
        np.random.shuffle(keys_list)
        N_scene = len(keys_list)
        val_count = int(val_ratio * N_scene)

        val_keys_scene = keys_list[:val_count]
        train_keys_scene = keys_list[val_count:]

        val_keys.extend(val_keys_scene)
        train_keys.extend(train_keys_scene)

    # Combine all
    all_keys = train_keys + val_keys
    N = len(all_keys)

    print(f"Total frames after grouping: {N}, train={len(train_keys)}, val={len(val_keys)}")

    # Prepare arrays for REAR and SIDE
    # We'll create arrays for REAR and SIDE in the final consistent order
    # The order is [train_keys..., val_keys...], so index i in the final array
    # belongs to either train or val depending on i in train_indices or val_indices
    rear_points  = np.zeros((N, 70, 8), dtype=np.float32)
    rear_labels  = np.zeros((N, 2, 7),  dtype=np.float32)
    side_points  = np.zeros((N, 70, 8), dtype=np.float32)
    side_labels  = np.zeros((N, 2, 7),  dtype=np.float32)

    # Fill arrays in the new consistent order
    meta_array = []
    for i, key in enumerate(all_keys):
        r_item = rear_dict[key]
        s_item = side_dict[key]
        rear_points[i] = r_item["points"]
        rear_labels[i] = r_item["labels"]
        side_points[i] = s_item["points"]
        side_labels[i] = s_item["labels"]
        meta_array.append([key[0], key[1], key[2]])
    # We'll build train_indices and val_indices accordingly
    train_indices = np.arange(len(train_keys))
    val_indices   = np.arange(len(train_keys), len(train_keys) + len(val_keys))
    meta_array = np.array(meta_array, dtype=object)


    # 4C) Output directories
    out_rear = "input_files_rear_Nov_16_17_v3"
    out_side = "input_files_side_Nov_16_17_v3"
    os.makedirs(out_rear, exist_ok=True)
    os.makedirs(out_side, exist_ok=True)

    # -- META --
    np.save(os.path.join(out_rear,"metadata.npy"), meta_array)

    # -- REAR --
    np.save(os.path.join(out_rear, "inputs.npy"),  rear_points)
    np.save(os.path.join(out_rear, "labels.npy"),  rear_labels)
    np.save(os.path.join(out_rear, "train_indices.npy"), train_indices)
    np.save(os.path.join(out_rear, "val_indices.npy"),   val_indices)

    # -- SIDE --
    np.save(os.path.join(out_side, "inputs.npy"),  side_points)
    np.save(os.path.join(out_side, "labels.npy"),  side_labels)
    np.save(os.path.join(out_side, "train_indices.npy"), train_indices)
    np.save(os.path.join(out_side, "val_indices.npy"),   val_indices)

    print(f"Saved REAR dataset to '{out_rear}', SIDE dataset to '{out_side}'.")
    print(f"Total matched frames: {N}")

# TODO: later,update this func to chceck output is correct or not 
def check_scene_split(metadata_file, train_indices_file, val_indices_file):
    """
    metadata_file: path to a .npy with shape (N,3), each row = [day_label, scene_no, frame_idx]
    train_indices_file: path to train_indices.npy
    val_indices_file:   path to val_indices.npy

    We'll group by (day_label, scene_no) and see how many train vs val.
    """
    # Load metadata and index arrays
    metadata = np.load(metadata_file)  # shape (N,3) or object dtype
    train_indices = np.load(train_indices_file)
    val_indices   = np.load(val_indices_file)

    # If day_label is a string ("Nov16"/"Nov17"), it's best if metadata is an object array, 
    # else you might store day as 0/1. Adjust as needed.
    
    # We'll build counters for train/val in each scene group
    scene2train = defaultdict(int)
    scene2val   = defaultdict(int)

    # For each index in train_indices, find which scene
    for idx in train_indices:
        # Suppose metadata[idx] = [day_label, scene_no, frame_idx]
        day_label, scene_no, frame_idx = metadata[idx]
        scene_id = (day_label, scene_no)
        scene2train[scene_id] += 1

    # For val
    for idx in val_indices:
        day_label, scene_no, frame_idx = metadata[idx]
        scene_id = (day_label, scene_no)
        scene2val[scene_id] += 1

    # Print results
    print("=== Scene-based split check ===")
    for scene_id in sorted(scene2train.keys() | scene2val.keys()):
        train_count = scene2train[scene_id]
        val_count   = scene2val[scene_id]
        total = train_count + val_count
        if total == 0:
            continue
        ratio_val = 100.0 * val_count / total
        ratio_train = 100.0 * train_count / total
        print(f"Scene {scene_id}: train={train_count}, val={val_count}, total={total}, "
              f"train%={ratio_train:.1f}, val%={ratio_val:.1f}")


if __name__ == "__main__":
    main()
