import os
import shutil
from sklearn.model_selection import train_test_split
import glob
from collections import Counter

base_path = "D:/AITexProject/aitex_dataset"

def get_fabric_type(filepath):
    filename = os.path.basename(filepath)
    return filename.split('_')[-1].replace('.png', '')

# Collect paths
defect_paths = glob.glob(os.path.join(base_path, "Defect_images", "*.png"))
nodefect_paths = glob.glob(os.path.join(base_path, "NODefect_images", "**", "*.png"), recursive=True)

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
out_base = "D:/AITexProject/aitex_dataset_processed"
for split in ["train", "val"]:
    for cls in ["Defect", "NoDefect"]:
        os.makedirs(os.path.join(out_base, split, cls), exist_ok=True)

def copy_images(paths, labels, split):
    for path, label in zip(paths, labels):
        cls = "Defect" if label == 1 else "NoDefect"
        dest = os.path.join(out_base, split, cls, os.path.basename(path))
        shutil.copy(path, dest)

copy_images(X_train, y_train, "train")
copy_images(X_val, y_val, "val")

print(f"\nTraining - Defect: {sum(y_train)}, NoDefect: {len(y_train)-sum(y_train)}")
print(f"Validation - Defect: {sum(y_val)}, NoDefect: {len(y_val)-sum(y_val)}")

# Check fabric distribution in splits
train_fabrics = [get_fabric_type(p) for p in X_train]
val_fabrics = [get_fabric_type(p) for p in X_val]
print("\nTrain fabrics:", Counter(train_fabrics))
print("Val fabrics:", Counter(val_fabrics))