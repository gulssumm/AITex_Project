"""
U-Net Strategy for Tiny Defects
Key corrections:
1. Fixed grayscale handling (1 channel instead of 3)
2. Updated model architecture for grayscale input
3. Corrected dataset statistics
4. Fixed weight calculation logic
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# === PATHS ===
base_path = "D:/AITexProject/aitex_dataset"
img_dir = os.path.join(base_path, "Defect_images")
mask_dir = os.path.join(base_path, "Mask_images")

# === SETTINGS ===
IMG_SIZE = 256
BATCH_SIZE = 4

# ==========================================
# LOAD DATA
# ==========================================
def load_data():
    """Load images as grayscale (1 channel) instead of RGB"""
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    images = []
    masks = []

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)

        mask_file = img_file.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        masks.append(mask)

    return np.array(images), np.array(masks)

X, y = load_data()
# Add channel dimension for grayscale: (N, H, W) -> (N, H, W, 1)
X = np.expand_dims(X, axis=-1)
y = np.expand_dims(y, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.reshape(len(y), -1).max(axis=1)
)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print(f"Image shape: {X_train.shape[1:]}")  # Should be (256, 256, 1)

# Normalize
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')

# Calculate class weights
total_pixels = y_train.size
defect_pixels = np.sum(y_train)
background_pixels = total_pixels - defect_pixels

print(f"\nClass Distribution:")
print(f"  Background: {background_pixels/total_pixels*100:.2f}%")
print(f"  Defect: {defect_pixels/total_pixels*100:.2f}%")
print(f"  Imbalance ratio: {background_pixels/defect_pixels:.1f}:1")

# ==========================================
# DATA AUGMENTATION (CRITICAL FOR SMALL DATASET)
# ==========================================
def augment_data(image, mask):
    """
    Apply random augmentations to increase effective dataset size
    This is CRUCIAL when you only have 85 training images
    """
    # Random flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Random rotation (90 degree increments)
    k = tf.cast(tf.random.uniform(()) * 4, tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)

    # Random brightness (only for image, not mask)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

# Create TF datasets with augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).map(augment_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# IMPROVED LOSS FUNCTION WITH ADAPTIVE WEIGHTING
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

    # REDUCED weight: 50x instead of 500x to prevent "predict everything" behavior
    # This is the KEY fix based on your analysis
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
    """Dice coefficient - measures overlap"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return dice

def precision_metric(y_true, y_pred):
    """Precision: of predicted defects, how many are real?"""
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)

    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def recall_metric(y_true, y_pred):
    """Recall: of actual defects, how many did we find?"""
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)

    recall = true_positives / (actual_positives + 1e-7)
    return recall

def iou_metric(y_true, y_pred):
    """IoU (Intersection over Union) - another useful metric"""
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = intersection / (union + 1e-7)
    return iou

# ==========================================
# CORRECTED U-NET FOR GRAYSCALE (1 CHANNEL)
# ==========================================
def tiny_unet(input_size=(IMG_SIZE, IMG_SIZE, 1)):  # Changed from 3 to 1
    """
    Smaller U-Net optimized for grayscale input
    Reason: With only 85 images, a large model will overfit
    """
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
print("\nBuilding Tiny U-Net for Grayscale Images...")
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
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print("Strategy:")
print("  1. Grayscale input (1 channel) - native format")
print("  2. Smaller model (fewer parameters to prevent overfitting)")
print("  3. Data augmentation (flips, rotations, brightness, contrast)")
print("  4. REDUCED weight for defect pixels (50x instead of 500x)")
print("  5. Balanced loss: BCE + Dice + Focal")
print("  6. Added BatchNormalization for stability")
print()

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
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

results = model.evaluate(val_dataset, verbose=0)
print(f"\nValidation Loss: {results[0]:.4f}")
print(f"Validation Dice: {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"Validation Recall: {results[2]:.4f} ({results[2]*100:.2f}%)")
print(f"Validation Precision: {results[3]:.4f} ({results[3]*100:.2f}%)")
print(f"Validation IoU: {results[4]:.4f} ({results[4]*100:.2f}%)")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
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
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

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
print("✓ Saved: unet_tiny_predictions.png")
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
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

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

print("\n" + "="*60)
print("KEY IMPROVEMENTS FROM PREVIOUS VERSION")
print("="*60)
print("1.Native grayscale processing (3x fewer parameters)")
print("2.Reduced weight from 500x to 50x (prevents 'predict everything')")
print("3.Added focal loss component (handles hard examples)")
print("4.Added BatchNormalization (improves stability)")
print("5.Added IoU metric (additional evaluation metric)")
print("6.Balanced loss combination (0.4 BCE + 0.4 Dice + 0.2 Focal)")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print("\nSaved files:")
print("  - unet_tiny_best.keras (best model)")
print("  - unet_tiny_predictions.png (visual results)")
print("  - unet_tiny_training.png (training curves)")
print("="*60)