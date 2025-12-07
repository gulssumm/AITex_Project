import os
import shutil
import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

INPUT_BASE = "D:/AITexProject/aitex_dataset"
OUTPUT_BASE = "D:/AITexProject/aitex_raw_clean"

MIN_SAMPLES = 10  # Fabrics with fewer than 10 samples (like '08') will be dropped

def get_fabric_code(filepath):
    # 0092_030_01.png
    return os.path.basename(filepath).split('_')[-1].replace('.png', '')


def copy_files(file_list, split_name):
    """
    Copies images and their corresponding masks to the output directory
    Creates a .txt placeholder if a mask is missing
    """
    dest_img_dir = os.path.join(OUTPUT_BASE, split_name, "images")
    dest_mask_dir = os.path.join(OUTPUT_BASE, split_name, "masks")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    print(f"Copying {len(file_list)} files to {split_name}")

    for img_path in tqdm(file_list):
        filename = os.path.basename(img_path)
        
        # Raw Copy
        shutil.copy(img_path, os.path.join(dest_img_dir, filename))

        if "Defect_images" in img_path:
            mask_filename = filename.replace(".png", "_mask.png")
            mask_src_path = os.path.join(INPUT_BASE, "Mask_images", mask_filename)

            if os.path.exists(mask_src_path):
                # Mask exists
                shutil.copy(mask_src_path, os.path.join(dest_mask_dir, mask_filename))
            else:
                # Mask missing
                txt_filename = mask_filename.replace(".png", ".txt")
                txt_path = os.path.join(dest_mask_dir, txt_filename)

                with open(txt_path, "w") as f:
                    f.write(f"MISSING_MASK for {filename}")


def main():
    if os.path.exists(OUTPUT_BASE):
        print(f"Removing old output at {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)

    defect_files = glob.glob(os.path.join(INPUT_BASE, "Defect_images", "*.png"))
    nodefect_files = glob.glob(os.path.join(INPUT_BASE, "NODefect_images", "**", "*.png"), recursive=True)
    all_files = defect_files + nodefect_files

    print(f"Total raw images found: {len(all_files)}")

    # Filter Out Rare Fabrics
    fabric_codes = [get_fabric_code(f) for f in all_files]
    counts = Counter(fabric_codes)

    valid_fabrics = {code for code, count in counts.items() if count >= MIN_SAMPLES}
    dropped_fabrics = set(counts.keys()) - valid_fabrics

    print(f"Valid Fabrics: {len(valid_fabrics)}")
    print(f"Dropping Fabrics (<{MIN_SAMPLES} samples): {dropped_fabrics}")

    # Create the final valid list
    valid_files = [f for f in all_files if get_fabric_code(f) in valid_fabrics]
    print(f"Final dataset size: {len(valid_files)} images (dropped {len(all_files) - len(valid_files)})")

    # Split Train/Val
    train_files, val_files = train_test_split(valid_files, test_size=0.2, random_state=42)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print(f"\nClean dataset: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()