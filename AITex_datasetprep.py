import os
import shutil
from sklearn.model_selection import train_test_split
import glob
from collections import Counter

base_path = "D:/AITexProject/aitex_dataset"
out_base = "D:/AITexProject/aitex_dataset_processed"

def get_fabric_type(filepath):
    filename = os.path.basename(filepath)
    return filename.split('_')[-1].replace('.png', '')

def get_mask_path(image_path, mask_base_dir):
    # Calculates the corresponding mask path for a defect image
    filename = os.path.basename(image_path)
    # Masks follow the pattern: 'nnnn_ddd_ff.png' -> 'nnnn_ddd_ff_mask.png'
    mask_filename = filename.replace('.png', '_mask.png')
    return os.path.join(mask_base_dir, mask_filename)

# Data Collection
defect_paths = glob.glob(os.path.join(base_path, "Defect_images", "*.png"))
nodefect_paths = glob.glob(os.path.join(base_path, "NODefect_images", "**", "*.png"), recursive=True)
mask_base_dir = os.path.join(base_path, "Mask_images")

X = defect_paths + nodefect_paths
y = [1]*len(defect_paths) + [0]*len(nodefect_paths)
fabrics = [get_fabric_type(p) for p in X]

print("Original fabric distribution:", Counter(fabrics))

# Remove fabric 08 (1 sample)
filtered = [(x, label, fab) for x, label, fab in zip(X, y, fabrics) if fab != '08']
X = [item[0] for item in filtered]
y = [item[1] for item in filtered]
fabrics = [item[2] for item in filtered]

print(f"After filtering: {len(X)} images")
print("Fabric distribution:", Counter(fabrics))

# Check label+fabric combinations
combo_counts = Counter([f"{label}_{fab}" for label, fab in zip(y, fabrics)])
print("\nLabel+Fabric combinations:")
for combo, count in combo_counts.most_common():
    print(f"  {combo}: {count}")

# STRATIFY BY LABEL ONLY
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Only balance defect/no-defect
    random_state=42
)

# Create dirs and copy
for split in ["train", "val"]:
    os.makedirs(os.path.join(out_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_base, split, "masks"), exist_ok=True)

def copy_files(paths, labels, split):
    for path, label in zip(paths, labels):
        dest_img_path = os.path.join(out_base, split, "images", os.path.basename(path))
        shutil.copy(path, dest_img_path)

        image_filename = os.path.basename(path)
        dest_mask_folder = os.path.join(out_base, split, "masks")

        if label == 1:  # Defect Image
            mask_path = get_mask_path(path, mask_base_dir)
            dest_mask_path = os.path.join(dest_mask_folder, os.path.basename(mask_path))

            if os.path.exists(mask_path):
                # Copy the real defect mask
                shutil.copy(mask_path, dest_mask_path)
            else:
                print(f"WARNING: Mask not found for {image_filename}. Creating placeholder.")
                with open(os.path.join(dest_mask_folder, image_filename.replace('.png', '_MISSING_MASK.txt')),
                          'w') as f:
                    f.write(f"Mask missing for {image_filename}")

        else:  # NoDefect Image (label == 0)
            placeholder_filename = image_filename.replace('.png', '_EMPTY_MASK_PLACEHOLDER.txt')
            with open(os.path.join(dest_mask_folder, placeholder_filename), 'w') as f:
                f.write(f"Generate all-black mask for image {image_filename}")
copy_files(X_train, y_train, "train")
copy_files(X_val, y_val, "val")

print(f"\nTraining - Defect: {sum(y_train)}, NoDefect: {len(y_train)-sum(y_train)}")
print(f"Validation - Defect: {sum(y_val)}, NoDefect: {len(y_val)-sum(y_val)}")

# Check fabric distribution in splits
train_fabrics = [get_fabric_type(p) for p in X_train]
val_fabrics = [get_fabric_type(p) for p in X_val]
print("\nTrain fabrics:", Counter(train_fabrics))
print("Val fabrics:", Counter(val_fabrics))