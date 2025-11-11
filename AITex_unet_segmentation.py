import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === PATHS ===
base_path = "D:/AITexProject/aitex_dataset"
img_dir = os.path.join(base_path, "Defect_images")
mask_dir = os.path.join(base_path, "Mask_images")

# === SETTINGS ===
IMG_WIDTH = 512
IMG_HEIGHT = 256
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1) # (H, W, C)
BATCH_SIZE = 4

# ==========================================
# ALBUMENTATIONS DEFINITIONS (Superior Augmentation)
# ==========================================
train_transform = A.Compose(
    [
        # GEOMETRIC
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.1, scale=(0.85, 1.15), rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT, p=0.7
        ),
        A.ElasticTransform(alpha=50, sigma=5, border_mode=cv2.BORDER_REFLECT, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, border_mode=cv2.BORDER_REFLECT, p=0.3),

        # PHOTOMETRIC
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # NOISE - BLUR
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.4),
        A.Blur(blur_limit=3, p=0.2),

        # DEFECT-AWARE
        A.CoarseDropout(
            num_holes_range=(2, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.3
        ),
    ],
    additional_targets={'mask': 'mask'}
)

# ==========================================
# ALBUMENTATIONS WRAPPER FOR TENSORFLOW
# ==========================================

# Wrapper to execute Albumentations within the TF graph
def augment_fn(image, mask):
    # Convert tensors back to numpy arrays for Albumentations
    data = {"image": image.numpy(), "mask": mask.numpy()}
    # TensorFlow usually outputs (H, W, 1)
    # Albumentations expects (H, W, C) where C=3 or C=1
    # Convert to C=3 if C=1 before augmentation
    if data["image"].shape[-1] == 1:
        data["image"] = cv2.cvtColor(data["image"], cv2.COLOR_GRAY2RGB)

    augmented_data = train_transform(**data)

    aug_image = tf.convert_to_tensor(augmented_data['image'], dtype=tf.float32)

    if aug_image.shape[-1] == 3:
        # Convert RGB tensor back to grayscale (H, W) or (H, W, 1) NumPy array first
        aug_image_np = cv2.cvtColor(aug_image.numpy().astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Convert back to tensor and expand dimensions to (H, W, 1)
        aug_image = tf.expand_dims(tf.convert_to_tensor(aug_image_np, dtype=tf.float32), axis=-1)

    aug_mask = tf.convert_to_tensor(augmented_data['mask'], dtype=tf.float32)

    # Albumentations outputs (H, W) or (H, W, C)
    if len(aug_mask.shape) == 2:
        aug_mask = tf.expand_dims(aug_mask, axis=-1)

    # The normalization in tf_augment_wrapper will apply: aug_image / 255.0
    return aug_image, aug_mask

def tf_augment_wrapper(image, mask):
    aug_image, aug_mask = tf.py_function(
        func=augment_fn,
        inp=[image, mask],
        Tout=[tf.float32, tf.float32]
    )
    aug_image.set_shape(IMG_SHAPE)
    aug_mask.set_shape((IMG_HEIGHT, IMG_WIDTH, 1))
    return aug_image, aug_mask


# ==========================================
# LOAD DATA FROM PROCESSED FOLDERS
# ==========================================
PROCESSED_BASE_PATH = "D:/AITexProject/aitex_dataset_processed"
TRAIN_DIR = os.path.join(PROCESSED_BASE_PATH, "train")
VAL_DIR = os.path.join(PROCESSED_BASE_PATH, "val")

def load_validation_arrays(data_dir):
    # Loads all validation images and masks into NumPy arrays for final analysis
    img_folder = os.path.join(data_dir, "images")
    mask_folder = os.path.join(data_dir, "masks")

    X_val_list = []
    y_val_list = []

    image_paths = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')])

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)

        # Load Image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        # Apply normalization
        X_val_list.append(image / 255.0)

        # Load Mask or Placeholder
        mask_path = os.path.join(mask_folder, img_filename.replace('.png', '_mask.png'))
        is_nodefect_placeholder = img_filename.replace('.png', '_EMPTY_MASK_PLACEHOLDER.txt')

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
        elif os.path.exists(os.path.join(mask_folder, is_nodefect_placeholder)):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        else:
            # Handle the two missing mask warnings from datasetprep
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        y_val_list.append(mask)

    X_val_loaded = np.array(X_val_list).astype(np.float32)
    y_val_loaded = np.array(y_val_list).astype(np.float32)

    # Add channel dimension (N, H, W) -> (N, H, W, 1)
    X_val_loaded = np.expand_dims(X_val_loaded, axis=-1)
    y_val_loaded = np.expand_dims(y_val_loaded, axis=-1)

    return X_val_loaded, y_val_loaded

def create_tf_datasets_from_processed_files(data_dir, batch_size, is_training):
    # Creates a TensorFlow Dataset pipeline for efficient data streaming
    img_folder = os.path.join(data_dir, "images")
    mask_folder = os.path.join(data_dir, "masks")

    image_paths = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')])

    # Load and process a single image/mask pair (TF-compatible)
    def load_and_process_pair(img_path_tensor):
        img_path = img_path_tensor.numpy().decode('utf-8')
        img_filename = os.path.basename(img_path)

        mask_path = os.path.join(mask_folder, img_filename.replace('.png', '_mask.png'))
        is_nodefect_placeholder = img_filename.replace('.png', '_EMPTY_MASK_PLACEHOLDER.txt')

        # Load Image (Grayscale)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.AREA)

        # Load or Generate Mask
        if os.path.exists(mask_path):
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.resize(mask, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            mask = tf.cast(mask > 0, tf.float32)
        elif os.path.exists(os.path.join(mask_folder, is_nodefect_placeholder)):
            mask = tf.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)
        else:
            mask = tf.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)

        if not is_training:
            image = tf.cast(image, tf.float32) / 255.0

        return tf.cast(image, tf.float32), mask

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(
        lambda x: tf.py_function(load_and_process_pair, [x], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    def set_shapes(image, mask):
        image.set_shape(IMG_SHAPE)
        mask.set_shape((IMG_HEIGHT, IMG_WIDTH, 1))
        return image, mask

    dataset = dataset.map(set_shapes)

    if is_training:
        dataset = dataset.map(tf_augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

X_val, y_val = load_validation_arrays(VAL_DIR)
print(f"Validation arrays loaded: X_val shape {X_val.shape}, y_val shape {y_val.shape}")

# ==========================================
# CREATE DATASETS (FOR TRAINING/EVALUATION)
# ==========================================
train_dataset = create_tf_datasets_from_processed_files(
    TRAIN_DIR, BATCH_SIZE, is_training=True
)
val_dataset = create_tf_datasets_from_processed_files(
    VAL_DIR, BATCH_SIZE, is_training=False
)
print(f"Train Dataset: {len(train_dataset)} batches")
print(f"Validation Dataset: {len(val_dataset)} batches")

# ==========================================
# LOSS FUNCTION WITH ADAPTIVE WEIGHTING
# ==========================================

def weighted_bce_dice_loss(y_true, y_pred):
    """
    Improved combination with adaptive weighting
    1. Moderate weighted BCE (not too extreme)
    2. Dice loss for overlap
    3. Focal loss component for hard examples
    """
    # Weighted BCE with moderate weight
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Fix shape if needed
    if len(bce.shape) == 3:
        bce = tf.expand_dims(bce, axis=-1)

    # 50x to prevent "predict everything" behavior
    weight_map = y_true * 50.0 + (1 - y_true) * 1.0
    weighted_bce = tf.reduce_mean(bce * weight_map)

    # Dice loss
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)
    dice_loss = 1.0 - dice

    # Focal loss component (helps with hard examples)
    focal_weight = tf.pow(1.0 - y_pred, 2.0)  # gamma=2
    focal_bce = focal_weight * bce
    focal_loss = tf.reduce_mean(focal_bce)

    # Balanced combination: equal weight to BCE and Dice, small focal component
    return 0.4 * weighted_bce + 0.4 * dice_loss + 0.2 * focal_loss


# ==========================================
# METRICS
# ==========================================

def dice_coefficient(y_true, y_pred, smooth=1.0):
    # Dice coefficient - measures overlap
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return dice

def precision_metric(y_true, y_pred):
    # Precision: of predicted defects, how many are real?
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)

    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def recall_metric(y_true, y_pred):
    # Recall: of actual defects, how many did we find?
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)

    recall = true_positives / (actual_positives + 1e-7)
    return recall

def iou_metric(y_true, y_pred):
    # IoU (Intersection over Union) - another useful metric
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = intersection / (union + 1e-7)
    return iou

# ==========================================
# U-NET FOR GRAYSCALE (1 CHANNEL)
# ==========================================
def tiny_unet(input_size=(IMG_SHAPE)):
    inputs = layers.Input(input_size)

    # Encoder - reduced filters
    # Level 1
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    # Level 2
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    # Level 3
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)

    # Bottleneck
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.3)(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.2)(c5)

    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.1)(c6)

    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# ==========================================
# BUILD MODEL
# ==========================================
print("\nBuilding Tiny U-Net for Grayscale Images")
model = tiny_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=weighted_bce_dice_loss,
    metrics=[dice_coefficient, recall_metric, precision_metric, iou_metric]
)

print(f"\nModel parameters: {model.count_params():,}")
model.summary()

# ==========================================
# CALLBACKS
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        patience=50,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'unet_tiny_best.keras',
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# TRAIN
# ==========================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# EVALUATE
# ==========================================
print("FINAL EVALUATION")

results = model.evaluate(val_dataset, verbose=0)
print(f"\nValidation Loss: {results[0]:.4f}")
print(f"Validation Dice: {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"Validation Recall: {results[2]:.4f} ({results[2]*100:.2f}%)")
print(f"Validation Precision: {results[3]:.4f} ({results[3]*100:.2f}%)")
print(f"Validation IoU: {results[4]:.4f} ({results[4]*100:.2f}%)")

print("INTERPRETATION")
print("\nDice Score (overlap):")
if results[1] > 0.5:
    print(">50% - Excellent segmentation!")
elif results[1] > 0.3:
    print("30-50% - Good detection, some improvement possible")
elif results[1] > 0.1:
    print("10-30% - Moderate detection, needs work")
else:
    print("<10% - Poor detection - task may be too challenging")

print("\nRecall (% of defects found):")
if results[2] > 0.7:
    print(">70% - Finding most defects")
elif results[2] > 0.4:
    print("40-70% - Finding many defects")
elif results[2] > 0.2:
    print("20-40% - Finding some defects")
else:
    print("<20% - Missing most defects")

print("\nPrecision (accuracy of predictions):")
if results[3] > 0.7:
    print(">70% - Few false alarms")
elif results[3] > 0.4:
    print("40-70% - Some false alarms")
else:
    print("<40% - Many false alarms")

# ==========================================
# VISUALIZE PREDICTIONS
# ==========================================
print("GENERATING VISUALIZATIONS")

predictions = model.predict(val_dataset)
predictions_binary = (predictions > 0.5).astype(np.uint8)

# Find images with actual defects
defect_indices = [i for i in range(len(y_val)) if np.sum(y_val[i]) > 0]
print(f"\nFound {len(defect_indices)} validation images with defects")

# Show up to 6 examples
num_samples = min(6, len(defect_indices))
sample_indices = defect_indices[:num_samples] if defect_indices else range(min(6, len(y_val)))

fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
if num_samples == 1:
    axes = axes.reshape(1, -1)

for i, idx in enumerate(sample_indices):
    # Original (grayscale)
    axes[i, 0].imshow(X_val[idx].squeeze(), cmap='gray')
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')

    # Ground truth
    gt_pct = (np.sum(y_val[idx]) / y_val[idx].size) * 100
    axes[i, 1].imshow(y_val[idx].squeeze(), cmap='gray')
    axes[i, 1].set_title(f'Ground Truth\n({gt_pct:.2f}% defect)')
    axes[i, 1].axis('off')

    # Prediction
    pred_pct = (np.sum(predictions_binary[idx]) / predictions_binary[idx].size) * 100
    axes[i, 2].imshow(predictions_binary[idx].squeeze(), cmap='gray')
    axes[i, 2].set_title(f'Prediction\n({pred_pct:.2f}% defect)')
    axes[i, 2].axis('off')

    # Overlay
    axes[i, 3].imshow(X_val[idx].squeeze(), cmap='gray')
    axes[i, 3].imshow(predictions_binary[idx].squeeze(), cmap='Reds', alpha=0.5)
    axes[i, 3].set_title('Overlay')
    axes[i, 3].axis('off')

plt.suptitle('Segmentation Results (Grayscale)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('unet_tiny_predictions.png', dpi=300, bbox_inches='tight')
print("Saved: unet_tiny_predictions.png")
plt.show()

# ==========================================
# PLOT TRAINING CURVES
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Dice
axes[0, 0].plot(history.history['dice_coefficient'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_dice_coefficient'], label='Val', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Dice Coefficient')
axes[0, 0].set_title('Dice Coefficient')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Recall
axes[0, 1].plot(history.history['recall_metric'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_recall_metric'], label='Val', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].set_title('Recall (% defects found)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[0, 2].plot(history.history['precision_metric'], label='Train', linewidth=2)
axes[0, 2].plot(history.history['val_precision_metric'], label='Val', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Precision')
axes[0, 2].set_title('Precision (prediction accuracy)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# IoU
axes[1, 0].plot(history.history['iou_metric'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_iou_metric'], label='Val', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('IoU')
axes[1, 0].set_title('IoU (Intersection over Union)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Loss
axes[1, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# F1 Score (computed from precision and recall)
train_precision = np.array(history.history['precision_metric'])
train_recall = np.array(history.history['recall_metric'])
val_precision = np.array(history.history['val_precision_metric'])
val_recall = np.array(history.history['val_recall_metric'])

train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-7)
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

axes[1, 2].plot(train_f1, label='Train', linewidth=2)
axes[1, 2].plot(val_f1, label='Val', linewidth=2)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('F1 Score')
axes[1, 2].set_title('F1 Score (harmonic mean)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('unet_tiny_training.png', dpi=300, bbox_inches='tight')
print("Saved: unet_tiny_training.png")
plt.show()

# ==========================================
# DETAILED ANALYSIS OF RESULTS
# ==========================================
print("DETAILED ANALYSIS")

# Per-image analysis
print("\nPer-image Dice scores:")
dice_scores = []
for i in range(len(y_val)):
    y_true_i = y_val[i].flatten()
    y_pred_i = predictions_binary[i].flatten()

    intersection = np.sum(y_true_i * y_pred_i)
    dice_i = (2.0 * intersection) / (np.sum(y_true_i) + np.sum(y_pred_i) + 1e-7)

    if np.sum(y_true_i) > 0:  # Only for images with defects
        dice_scores.append(dice_i)
        gt_pct = (np.sum(y_true_i) / len(y_true_i)) * 100
        pred_pct = (np.sum(y_pred_i) / len(y_pred_i)) * 100
        print(f"  Image {i}: Dice={dice_i:.3f}, GT={gt_pct:.2f}%, Pred={pred_pct:.2f}%")

if dice_scores:
    print(f"\nDice Score Statistics:")
    print(f"Mean: {np.mean(dice_scores):.3f}")
    print(f"Median: {np.median(dice_scores):.3f}")
    print(f"Std: {np.std(dice_scores):.3f}")
    print(f"Min: {np.min(dice_scores):.3f}, Max: {np.max(dice_scores):.3f}")

print("TRAINING COMPLETE")
print("\nSaved files:")
print("unet_tiny_best.keras (best model)")
print("unet_tiny_predictions.png (visual results)")
print("unet_tiny_training.png (training curves)")