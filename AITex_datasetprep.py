import os
import shutil
from sklearn.model_selection import train_test_split
import glob
from collections import Counter
from PIL import Image, ImageOps
import numpy as np

# CONFIGURATION
base_path = "D:/AITexProject/aitex_dataset"
out_base = "D:/AITexProject/aitex_dataset_processed"
TARGET_SIZE = (4096, 256)

def get_fabric_type(filepath):
    filename = os.path.basename(filepath)
    return filename.split('_')[-1].replace('.png', '')

def get_mask_path(image_path, mask_base_dir):
    filename = os.path.basename(image_path)
    mask_filename = filename.replace('.png', '_mask.png')
    return os.path.join(mask_base_dir, mask_filename)


def safe_save_image(src_path, dest_path, target_size=TARGET_SIZE):
    """
    1. Converts everything to standard Grayscale ('L')
    2. Pads images if they are smaller than 4096x256
    """
    try:
        with Image.open(src_path) as img:
            # 1. Force Grayscale (removes Alpha channel issues)
            img = img.convert('L')

            # 2. Check Size
            if img.size == target_size:
                img.save(dest_path)
            else:
                print(f"Fixing size mismatch: {os.path.basename(src_path)} {img.size} -> {target_size}")

                # Create a black canvas
                new_img = Image.new('L', target_size, 0)

                # Paste the original image (align left)
                new_img.paste(img, (0, 0))

                # Save
                new_img.save(dest_path)

    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def copy_files(file_list, label_list, split):
    # Create directories
    dest_img_folder = os.path.join(out_base, split, "images")
    dest_mask_folder = os.path.join(out_base, split, "masks")
    os.makedirs(dest_img_folder, exist_ok=True)
    os.makedirs(dest_mask_folder, exist_ok=True)

    print(f"\nProcessing {split} set...")

    for path, label in zip(file_list, label_list):
        image_filename = os.path.basename(path)
        dest_img_path = os.path.join(dest_img_folder, image_filename)

        # Default assumption: Image is valid unless proven otherwise
        is_valid_defect = False

        if label == 1:  # Defect Image
            mask_path = get_mask_path(path, mask_base_dir)
            dest_mask_path = os.path.join(dest_mask_folder, os.path.basename(mask_path))

            if os.path.exists(mask_path):
                try:
                    with Image.open(mask_path) as m:
                        if np.array(m).max() > 0:
                            # It has white pixels. REAL defect.
                            is_valid_defect = True
                            safe_save_image(mask_path, dest_mask_path)
                        else:
                            # It exists but is all black. FAKE defect.
                            print(f"SKIPPING POISON: {image_filename} has an empty mask.")
                except Exception as e:
                    print(f"Error checking mask {mask_path}: {e}")

            # If it was not valid (either missing file OR empty content)
            if not is_valid_defect:
                # Mark as missing so the Data Loader skips it
                if not os.path.exists(mask_path):
                    print(f"WARNING: Mask file missing for {image_filename}")

                with open(os.path.join(dest_mask_folder, image_filename.replace('.png', '_MISSING_MASK.txt')),
                          'w') as f:
                    f.write(f"Mask invalid for {image_filename}")

            # Only save the image if we didn't mark it as broken?
            # Actually, we SAVE the image, but the _MISSING_MASK file tells the loader to ignore it.
            safe_save_image(path, dest_img_path)

        else:  # NoDefect Image (label == 0)
            safe_save_image(path, dest_img_path)
            # Create marker file for normal images
            placeholder_filename = image_filename.replace('.png', '_EMPTY_MASK_PLACEHOLDER.txt')
            with open(os.path.join(dest_mask_folder, placeholder_filename), 'w') as f:
                f.write(f"Generate all-black mask for image {image_filename}")

# 1. Find Files
defect_paths = glob.glob(os.path.join(base_path, "Defect_images", "*.png"))
nodefect_paths = glob.glob(os.path.join(base_path, "NODefect_images", "**", "*.png"), recursive=True)
mask_base_dir = os.path.join(base_path, "Mask_images")

print(f"Found {len(defect_paths)} defect images")
print(f"Found {len(nodefect_paths)} normal images")

# 2. Combine
X = defect_paths + nodefect_paths
y = [1] * len(defect_paths) + [0] * len(nodefect_paths)
fabrics = [get_fabric_type(p) for p in X]

# 3. Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Clear Old Output Directory
if os.path.exists(out_base):
    print("Cleaning old processed folder...")
    shutil.rmtree(out_base)

# 5. Run Processing
copy_files(X_train, y_train, "train")
copy_files(X_val, y_val, "val")

print(f"\nDataset processed to: {out_base}")
print(f"Training: {len(X_train)} images")
print(f"Validation: {len(X_val)} images")