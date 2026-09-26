import os
import shutil
import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import numpy as np

INPUT_BASE = "D:/AITexProject/aitex_dataset"
OUTPUT_BASE = "D:/AITexProject/aitex_raw_clean"

MIN_SAMPLES = 10


def get_fabric_code(filepath):
    """Extract fabric code from filename: 0092_030_01.png -> 01"""
    return os.path.basename(filepath).split('_')[-1].replace('.png', '')


def create_blank_mask(img_shape, save_path):
    """Create a blank mask (all zeros) for no-defect images"""
    blank_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    cv2.imwrite(save_path, blank_mask)


def check_mask_exists(img_path):
    """Check if mask exists for any image"""
    filename = os.path.basename(img_path)
    mask_filename = filename.replace(".png", "_mask.png")
    mask_path = os.path.join(INPUT_BASE, "Mask_images", mask_filename)
    return os.path.exists(mask_path), mask_path


def is_defect_image(filepath):
    """Check if image is from Defect_images folder"""
    # Normalize path separators for cross-platform compatibility
    normalized = filepath.replace('\\', '/')
    # Must contain 'Defect_images' but NOT 'NODefect_images'
    return "Defect_images" in normalized and "NODefect_images" not in normalized


def copy_files(file_list, split_name):
    """
    Copies images and their corresponding masks to the output directory
    """
    dest_img_dir = os.path.join(OUTPUT_BASE, split_name, "images")
    dest_mask_dir = os.path.join(OUTPUT_BASE, split_name, "masks")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    print(f"\nCopying {len(file_list)} files to {split_name}...")

    skipped_defects = 0
    created_blank = 0
    copied_defect_masks = 0
    preserved_nodefect_masks = 0

    for img_path in tqdm(file_list):
        filename = os.path.basename(img_path)
        mask_filename = filename.replace(".png", "_mask.png")
        mask_exists, mask_src_path = check_mask_exists(img_path)

        if is_defect_image(img_path):
            # DEFECT IMAGE - Must have real mask
            if mask_exists:
                # Mask exists - copy both image and mask
                shutil.copy(img_path, os.path.join(dest_img_dir, filename))
                shutil.copy(mask_src_path, os.path.join(dest_mask_dir, mask_filename))
                copied_defect_masks += 1
            else:
                # Mask missing - SKIP this image entirely
                skipped_defects += 1
                print(f"\nSkipping defect {filename} - mask not found")
        else:
            # NO-DEFECT IMAGE - Check if mask exists, otherwise create blank
            img = cv2.imread(img_path)
            if img is not None:
                shutil.copy(img_path, os.path.join(dest_img_dir, filename))
                dest_mask_path = os.path.join(dest_mask_dir, mask_filename)

                if mask_exists:
                    # Preserve existing mask for no-defect image
                    shutil.copy(mask_src_path, dest_mask_path)
                    preserved_nodefect_masks += 1
                else:
                    # Create blank mask only if none exists
                    create_blank_mask(img.shape, dest_mask_path)
                    created_blank += 1
            else:
                print(f"\nCannot read {filename} - skipping")

    print(f"\nCopied defect masks: {copied_defect_masks}")
    print(f"Preserved no-defect masks: {preserved_nodefect_masks}")
    print(f"Created blank masks (no-defect): {created_blank}")
    if skipped_defects > 0:
        print(f"WARNING: Skipped defects (missing mask): {skipped_defects}")


def main():
    if os.path.exists(OUTPUT_BASE):
        print(f"\nRemoving old output: {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)

    # Collect files
    defect_files = glob.glob(os.path.join(INPUT_BASE, "Defect_images", "*.png"))
    nodefect_files = glob.glob(os.path.join(INPUT_BASE, "NODefect_images", "**", "*.png"), recursive=True)

    print(f"Defect images: {len(defect_files)}")
    print(f"No-defect images: {len(nodefect_files)}")

    # FILTER OUT DEFECT IMAGES WITHOUT MASKS
    print(f"\nChecking for missing masks in defect images...")
    defect_files_with_masks = []
    missing_mask_count = 0

    for img_path in defect_files:
        mask_exists, _ = check_mask_exists(img_path)
        if mask_exists:
            defect_files_with_masks.append(img_path)
        else:
            missing_mask_count += 1
            filename = os.path.basename(img_path)
            print(f"  {filename} - mask not found (will be excluded)")

    print(f"\nValid defect images: {len(defect_files_with_masks)}")
    print(f"Excluded (missing mask): {missing_mask_count}")

    # Check no-defect mask status
    nodefect_with_masks = sum(1 for f in nodefect_files if check_mask_exists(f)[0])
    print(f"\nNo-defect images with existing masks: {nodefect_with_masks}")
    print(f"No-defect images needing blank masks: {len(nodefect_files) - nodefect_with_masks}")

    # Filter no-defect by fabric
    nodefect_fabrics = [get_fabric_code(f) for f in nodefect_files]
    fabric_counts = Counter(nodefect_fabrics)

    print(f"\nNo-defect fabric distribution:")
    for fabric, count in sorted(fabric_counts.items()):
        print(f"  Fabric {fabric}: {count}")

    valid_fabrics = {code for code, count in fabric_counts.items() if count >= MIN_SAMPLES}
    dropped_fabrics = set(fabric_counts.keys()) - valid_fabrics

    if dropped_fabrics:
        print(f"\nFiltering no-defect images:")
        print(f"  Keeping fabrics: {sorted(valid_fabrics)}")
        print(f"  Dropping fabrics (<{MIN_SAMPLES} samples): {sorted(dropped_fabrics)}")

    # Defects with masks + filtered no-defects
    filtered_nodefect = [f for f in nodefect_files if get_fabric_code(f) in valid_fabrics]
    all_files = defect_files_with_masks + filtered_nodefect

    print(f"\nFinal dataset composition:")
    print(f"  Defect (with masks): {len(defect_files_with_masks)}")
    print(f"  No-defect (filtered): {len(filtered_nodefect)}")
    print(f"  Total: {len(all_files)}")
    if len(defect_files_with_masks) > 0:
        print(f"  Class ratio (no-defect/defect): {len(filtered_nodefect) / len(defect_files_with_masks):.2f}:1")

    # Create labels for stratification using is_defect_image function
    labels = ['defect' if is_defect_image(f) else 'nodefect' for f in all_files]

    # Verify label counts
    defect_count = labels.count('defect')
    nodefect_count = labels.count('nodefect')
    print(f"\nLabel verification before split:")
    print(f"  Defect labels: {defect_count}")
    print(f"  No-defect labels: {nodefect_count}")

    # Stratified split
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_defect = train_labels.count('defect')
    train_nodefect = train_labels.count('nodefect')
    val_defect = val_labels.count('defect')
    val_nodefect = val_labels.count('nodefect')
    print(f"TRAIN SET: {len(train_files)} total")
    print(f"  Defect: {train_defect} ({train_defect / len(train_files) * 100:.1f}%)")
    print(f"  No-defect: {train_nodefect} ({train_nodefect / len(train_files) * 100:.1f}%)")

    print(f"\nVAL SET: {len(val_files)} total")
    print(f"  Defect: {val_defect} ({val_defect / len(val_files) * 100:.1f}%)")
    print(f"  No-defect: {val_nodefect} ({val_nodefect / len(val_files) * 100:.1f}%)")

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print(f"Output location: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()