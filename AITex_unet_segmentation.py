"""
U-Net Segmentation for AITEX Dataset
This approach is MUCH better for tiny defects (0.45% of image)
because it preserves spatial information instead of global pooling
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import glob

# === PATHS ===
base_path = "D:/AITexProject/aitex_dataset"
img_dir = os.path.join(base_path, "Defect_images")
mask_dir = os.path.join(base_path, "Mask_images")

# === SETTINGS ===
IMG_SIZE = 256  # Larger than 224 to preserve small defect details
BATCH_SIZE = 4  # Small due to memory constraints with U-Net


# ==========================================
# DATA LOADING WITH MASKS
# ==========================================
def load_data():
    """Load images and their corresponding masks"""
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    images = []
    masks = []

    for img_file in img_files:
        # Load image
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        images.append(img)

        # Load mask
        mask_file = img_file.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = (mask > 0).astype(np.uint8)  # Binary: 0 or 1
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        masks.append(mask)

    return np.array(images), np.array(masks)


print("Loading dataset...")
X, y = load_data()
print(f"Loaded {len(X)} images with masks")
print(f"Image shape: {X.shape}")
print(f"Mask shape: {y.shape}")

# Add channel dimension to masks
y = np.expand_dims(y, axis=-1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# Normalize images AND masks to float32
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
y_train = y_train.astype('float32')  # Convert masks to float32
y_val = y_val.astype('float32')  # Convert masks to float32


# ==========================================
# U-NET MODEL ARCHITECTURE
# ==========================================
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    """
    U-Net architecture optimized for small defect detection
    """
    inputs = layers.Input(input_size)

    # Encoder (Contracting Path)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder (Expanding Path)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# ==========================================
# CUSTOM METRICS FOR SMALL DEFECTS
# ==========================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient - better for imbalanced segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Loss function optimized for small defects"""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combination of dice loss and binary crossentropy"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


# ==========================================
# BUILD AND COMPILE MODEL
# ==========================================
print("\nBuilding U-Net model...")
model = unet_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=[dice_coefficient, 'accuracy']
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'unet_best.keras',
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# TRAIN MODEL
# ==========================================
print("\nTraining U-Net...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# EVALUATE
# ==========================================
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

results = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {results[0]:.4f}")
print(f"Validation Dice Coefficient: {results[1]:.4f}")
print(f"Validation Accuracy: {results[2]:.4f}")

# ==========================================
# VISUALIZE PREDICTIONS
# ==========================================
print("\nGenerating prediction visualizations...")

# Predict on validation set
predictions = model.predict(X_val)
predictions_binary = (predictions > 0.5).astype(np.uint8)

# Select 6 samples with largest defects for visualization
defect_sizes = [np.sum(mask) for mask in y_val]
top_indices = np.argsort(defect_sizes)[-6:]

fig, axes = plt.subplots(6, 4, figsize=(16, 24))

for i, idx in enumerate(top_indices):
    # Original image
    axes[i, 0].imshow(X_val[idx])
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')

    # Ground truth mask
    axes[i, 1].imshow(y_val[idx].squeeze(), cmap='gray')
    axes[i, 1].set_title('Ground Truth')
    axes[i, 1].axis('off')

    # Predicted mask
    axes[i, 2].imshow(predictions_binary[idx].squeeze(), cmap='gray')
    axes[i, 2].set_title('Prediction')
    axes[i, 2].axis('off')

    # Overlay
    axes[i, 3].imshow(X_val[idx])
    axes[i, 3].imshow(predictions_binary[idx].squeeze(), cmap='Reds', alpha=0.5)
    axes[i, 3].set_title('Overlay')
    axes[i, 3].axis('off')

plt.suptitle('U-Net Segmentation Results on Validation Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('unet_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# PLOT TRAINING HISTORY
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dice coefficient
axes[0].plot(history.history['dice_coefficient'], label='Train Dice', linewidth=2)
axes[0].plot(history.history['val_dice_coefficient'], label='Val Dice', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Dice Coefficient')
axes[0].set_title('Dice Coefficient over Training')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss over Training')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('unet_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Training complete!")
print("Saved: unet_best.keras")
print("Saved: unet_predictions.png")
print("Saved: unet_training_curves.png")
print("=" * 60)