import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

# TRAINING AUGMENTATION (Aggressive)
train_transform = A.Compose([
    # GEOMETRIC TRANSFORMS (Defect-safe)
    A.HorizontalFlip(p=0.5),
    # No vertical flip - fabric has directional weave

    A.ShiftScaleRotate(
        shift_limit=0.1,  # 10% shift
        scale_limit=0.15,  # ±15% zoom
        rotate_limit=10,  # ±10° rotation
        border_mode=cv2.BORDER_REFLECT,
        p=0.7
    ),

    # Elastic deformation (simulates fabric stretching)
    A.ElasticTransform(
        alpha=50,
        sigma=5,
        alpha_affine=10,
        border_mode=cv2.BORDER_REFLECT,
        p=0.3
    ),

    # Grid distortion (preserves defects better than elastic)
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.2,
        border_mode=cv2.BORDER_REFLECT,
        p=0.3
    ),

    # PHOTOMETRIC TRANSFORMS (Simulates lighting variations)
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.6
    ),

    # Gamma correction (non-linear brightness)
    A.RandomGamma(
        gamma_limit=(80, 120),
        p=0.3
    ),

    # NOISE & BLUR (Simulates camera/sensor issues)
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
    ], p=0.4),

    A.OneOf([
        A.Blur(blur_limit=3, p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
    ], p=0.2),

    # DEFECT-AWARE AUGMENTATIONS
    # Coarse dropout (simulates occlusion but preserves defects)
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        min_holes=2,
        min_height=8,
        min_width=8,
        fill_value=128,  # Gray fill (fabric-like)
        p=0.3
    ),

    # NORMALIZATION
    A.Normalize(
        mean=[0.485],  # ImageNet grayscale
        std=[0.229],
        max_pixel_value=255.0,
        p=1.0
    ),

    ToTensorV2()
], p=1.0)

# VALIDATION AUGMENTATION (Only normalization)
val_transform = A.Compose([
    A.Normalize(
        mean=[0.485],
        std=[0.229],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()
], p=1.0)


# =============================================
# USAGE EXAMPLE
# =============================================
def load_and_augment(image_path, transform):
    # Load grayscale image and apply augmentation
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Convert to 3-channel (required for some augmentations)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply augmentation
    augmented = transform(image=image)
    return augmented['image']


# =============================================
# PYTORCH DATASET WITH AUGMENTATION
# =============================================
class AITEXDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Get all images from Defect and NoDefect folders
        self.defect_imgs = list((self.image_dir / "Defect").glob("*.png"))
        self.nodefect_imgs = list((self.image_dir / "NoDefect").glob("*.png"))

        self.images = self.defect_imgs + self.nodefect_imgs
        self.labels = [1] * len(self.defect_imgs) + [0] * len(self.nodefect_imgs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


# =============================================
# DATALOADER CREATION
# =============================================
def create_dataloaders(train_dir, val_dir, batch_size=32):
    # Create train and validation dataloaders
    train_dataset = AITEXDataset(train_dir, transform=train_transform)
    val_dataset = AITEXDataset(val_dir, transform=val_transform)

    # Calculate class weights for imbalanced dataset
    train_labels = train_dataset.labels
    class_counts = [train_labels.count(0), train_labels.count(1)]
    total = sum(class_counts)
    class_weights = [total / (2 * c) for c in class_counts]

    print(f"Class distribution: NoDefect={class_counts[0]}, Defect={class_counts[1]}")
    print(f"Class weights: NoDefect={class_weights[0]:.3f}, Defect={class_weights[1]:.3f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, class_weights


# =============================================
# USAGE
# =============================================
if __name__ == "__main__":
    train_dir = "D:/AITexProject/aitex_dataset_processed/train"
    val_dir = "D:/AITexProject/aitex_dataset_processed/val"

    train_loader, val_loader, class_weights = create_dataloaders(
        train_dir, val_dir, batch_size=16
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test augmentation
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"\nBatch shape: {images.shape}")  # Should be [16, 3, H, W]
    print(f"Labels: {labels}")