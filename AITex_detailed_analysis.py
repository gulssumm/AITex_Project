import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from collections import Counter

base_path = "D:/AITexProject/aitex_dataset"

def analyze_dataset_statistics():
    # Count images
    defect_dir = os.path.join(base_path, "Defect_images")
    nodefect_dir = os.path.join(base_path, "NODefect_images")
    mask_dir = os.path.join(base_path, "Mask_images")

    defect_files = [f for f in os.listdir(defect_dir) if f.lower().endswith('.png')]
    nodefect_files = []
    for root, dirs, files in os.walk(nodefect_dir):
        nodefect_files.extend([f for f in files if f.lower().endswith('.png')])

    print(f"\nClass Distribution:")
    print(f"Defect images: {len(defect_files)}")
    print(f"No-Defect images: {len(nodefect_files)}")
    print(f"Total: {len(defect_files) + len(nodefect_files)}")
    print(f"Class imbalance ratio: {len(nodefect_files) / len(defect_files):.2f}:1")

    return defect_files, nodefect_files


def analyze_image_quality(defect_files, nodefect_files):
    # Analyze image properties for understanding data characteristics
    print("IMAGE QUALITY ANALYSIS")

    defect_dir = os.path.join(base_path, "Defect_images")
    nodefect_dir = os.path.join(base_path, "NODefect_images")

    def get_stats(files, directory):
        brightness = []
        contrast = []

        for f in files[:50]:  # Sample 50 images
            if directory == nodefect_dir:
                # Find the file in subdirectories
                for root, dirs, file_list in os.walk(directory):
                    if f in file_list:
                        img_path = os.path.join(root, f)
                        break
            else:
                img_path = os.path.join(directory, f)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness.append(np.mean(img))
                contrast.append(np.std(img))

        return brightness, contrast

    defect_bright, defect_contrast = get_stats(defect_files, defect_dir)
    nodefect_bright, nodefect_contrast = get_stats(nodefect_files, nodefect_dir)

    print(f"\nDefect Images:")
    print(f"Mean brightness: {np.mean(defect_bright):.2f} ± {np.std(defect_bright):.2f}")
    print(f"Mean contrast (std): {np.mean(defect_contrast):.2f} ± {np.std(defect_contrast):.2f}")

    print(f"\nNo-Defect Images:")
    print(f"Mean brightness: {np.mean(nodefect_bright):.2f} ± {np.std(nodefect_bright):.2f}")
    print(f"Mean contrast (std): {np.mean(nodefect_contrast):.2f} ± {np.std(nodefect_contrast):.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(defect_bright, alpha=0.5, label='Defect', bins=20)
    axes[0].hist(nodefect_bright, alpha=0.5, label='No-Defect', bins=20)
    axes[0].set_xlabel('Brightness')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Brightness Distribution')
    axes[0].legend()

    axes[1].hist(defect_contrast, alpha=0.5, label='Defect', bins=20)
    axes[1].hist(nodefect_contrast, alpha=0.5, label='No-Defect', bins=20)
    axes[1].set_xlabel('Contrast (Std Dev)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Contrast Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('aitex_image_quality_analysis.png', dpi=300)


def analyze_defect_types():
    # Analyze defect characteristics using masks
    print("DEFECT CHARACTERISTICS ANALYSIS")

    mask_dir = os.path.join(base_path, "Mask_images")
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]

    defect_sizes = []
    defect_ratios = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            defect_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            ratio = defect_pixels / total_pixels

            defect_sizes.append(defect_pixels)
            defect_ratios.append(ratio * 100)  # percentage

    print(f"\nDefect Size Statistics:")
    print(f"Mean defect area: {np.mean(defect_ratios):.2f}% of image")
    print(f"Std deviation: {np.std(defect_ratios):.2f}%")
    print(f"Min: {np.min(defect_ratios):.2f}%, Max: {np.max(defect_ratios):.2f}%")
    print(f"Median: {np.median(defect_ratios):.2f}%")

    # Categorize defect sizes
    small = sum(1 for r in defect_ratios if r < 5)
    medium = sum(1 for r in defect_ratios if 5 <= r < 15)
    large = sum(1 for r in defect_ratios if r >= 15)

    print(f"\nDefect Size Categories:")
    print(f"Small (<5%): {small} images ({small / len(defect_ratios) * 100:.1f}%)")
    print(f"Medium (5-15%): {medium} images ({medium / len(defect_ratios) * 100:.1f}%)")
    print(f"Large (>15%): {large} images ({large / len(defect_ratios) * 100:.1f}%)")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.hist(defect_ratios, bins=30, edgecolor='black')
    plt.xlabel('Defect Area (% of image)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Defect Sizes')
    plt.axvline(np.mean(defect_ratios), color='r', linestyle='--', label=f'Mean: {np.mean(defect_ratios):.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('aitex_defect_size_distribution.png', dpi=300)



# ==========================================
# VISUAL SAMPLES
# ==========================================
def create_sample_visualization():
    print("CREATING SAMPLE VISUALIZATIONS")

    defect_dir = os.path.join(base_path, "Defect_images")
    nodefect_dir = os.path.join(base_path, "NODefect_images")
    mask_dir = os.path.join(base_path, "Mask_images")

    # Select sample images
    defect_samples = [f for f in os.listdir(defect_dir) if f.endswith('.png')][:6]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for i, img_file in enumerate(defect_samples):
        # Load image
        img_path = os.path.join(defect_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Load mask
        mask_file = img_file.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Display
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Sample {i + 1}')
        axes[0, i].axis('off')

        # Overlay
        axes[1, i].imshow(img, cmap='gray')
        if mask is not None:
            axes[1, i].imshow(mask, cmap='Reds', alpha=0.5)
        axes[1, i].set_title('With Defect Mask')
        axes[1, i].axis('off')

    plt.suptitle('AITEX Dataset: Defect Examples with Ground Truth Masks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('aitex_sample_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: aitex_sample_visualization.png")

if __name__ == "__main__":
    defect_files, nodefect_files = analyze_dataset_statistics()
    analyze_image_quality(defect_files, nodefect_files)
    analyze_defect_types()
    create_sample_visualization()

    base = "D:/AITexProject/aitex_dataset"
    all_imgs = glob.glob(os.path.join(base, "**", "*.png"), recursive=True)

    fabrics = []
    for img in all_imgs:
        if 'mask' not in img:
            fname = os.path.basename(img)
            fabric = fname.split('_')[-1].replace('.png', '')
            fabrics.append(fabric)

    print(Counter(fabrics))