import os
import shutil
from sklearn.model_selection import train_test_split
import glob

base_path = "D:/AITexProject/aitex_dataset"
# Collect image paths
defect_paths = glob.glob(os.path.join(base_path, "Defect_images", "*.png"))
nodefect_paths = glob.glob(os.path.join(base_path, "NODefect_images", "**", "*.png"), recursive=True)

X = defect_paths + nodefect_paths
y = [1]*len(defect_paths) + [0]*len(nodefect_paths)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create new dirs
out_base = "D:/AITexProject/aitex_dataset_processed"
for split in ["train", "val"]:
    for cls in ["Defect", "NoDefect"]:
        os.makedirs(os.path.join(out_base, split, cls), exist_ok=True)

# Copy files
def copy_images(paths, labels, split):
    for path, label in zip(paths, labels):
        cls = "Defect" if label == 1 else "NoDefect"
        dest = os.path.join(out_base, split, cls, os.path.basename(path))
        shutil.copy(path, dest)

copy_images(X_train, y_train, "train")
copy_images(X_val, y_val, "val")

# Verification
print(f"Training - Defect: {sum(y_train)}, NoDefect: {len(y_train)-sum(y_train)}")
print(f"Validation - Defect: {sum(y_val)}, NoDefect: {len(y_val)-sum(y_val)}")