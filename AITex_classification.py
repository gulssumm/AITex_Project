import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
data_dir = "D:/AITexProject/aitex_raw_clean"
crop_size = (256, 256)  # Square crops from original 4096x256 images
num_crops_per_image = 8  # How many crops to take from each image
batch_size = 32
epochs = 15


def load_dataset_with_crops(split='train'):
    """Load images and create multiple crops from each image"""
    img_dir = os.path.join(data_dir, split, "images")
    mask_dir = os.path.join(data_dir, split, "masks")

    images = []
    labels = []
    filenames = []

    for filename in sorted(os.listdir(img_dir)):
        if not filename.endswith('.png'):
            continue

        # Load original image (4096x256)
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Load mask
        mask_filename = filename.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w = img.shape
        crop_h, crop_w = crop_size

        # Calculate step size for evenly spaced crops
        step = (w - crop_w) // (num_crops_per_image - 1) if num_crops_per_image > 1 else 0

        # Extract crops from the image
        for i in range(num_crops_per_image):
            # Calculate crop position
            x_start = min(i * step, w - crop_w)
            x_end = x_start + crop_w
            y_start = 0  # Start from top since height is 256
            y_end = crop_h

            # Crop image and mask
            img_crop = img[y_start:y_end, x_start:x_end]
            mask_crop = mask[y_start:y_end, x_start:x_end]

            # Determine label for this crop
            is_defect = 1 if np.any(mask_crop > 0) else 0

            images.append(img_crop)
            labels.append(is_defect)
            filenames.append(f"{filename}_crop{i}")

    return np.array(images), np.array(labels), filenames


# Load data
X_train, y_train, train_files = load_dataset_with_crops('train')
X_val, y_val, val_files = load_dataset_with_crops('val')

# Dataset statistics
print(f"Training set: {len(X_train)} images")
print(f"  - Defects: {np.sum(y_train)} ({np.sum(y_train) / len(y_train) * 100:.1f}%)")
print(
    f"  - No defects: {len(y_train) - np.sum(y_train)} ({(len(y_train) - np.sum(y_train)) / len(y_train) * 100:.1f}%)")
print(f"\nValidation set: {len(X_val)} images")
print(f"  - Defects: {np.sum(y_val)} ({np.sum(y_val) / len(y_val) * 100:.1f}%)")
print(f"  - No defects: {len(y_val) - np.sum(y_val)} ({(len(y_val) - np.sum(y_val)) / len(y_val) * 100:.1f}%)")

# Normalize
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Add channel dimension
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")

# Build simple CNN model

model = models.Sequential([
    # Data augmentation
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(crop_size[0], crop_size[1], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight_dict,
    verbose=1
)

results = model.evaluate(X_val, y_val, verbose=0)
val_loss, val_acc, val_precision, val_recall = results

print(f"Validation Accuracy:  {val_acc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall:    {val_recall:.4f}")
f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
print(f"Validation F1-Score:  {f1_score:.4f}")

# Predictions
y_pred_probs = model.predict(X_val, verbose=0).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

# Show misclassified examples
misclassified_idx = np.where(y_pred != y_val)[0]
print(f"\nMisclassified: {len(misclassified_idx)}/{len(y_val)} images")

if len(misclassified_idx) > 0:
    print("\nFirst 10 misclassified examples:")
    for idx in misclassified_idx[:10]:
        print(f"  {val_files[idx]}: True={y_val[idx]}, Pred={y_pred[idx]}, Confidence={y_pred_probs[idx]:.3f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("\nSaved: training_history.png")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Defect', 'Defect'],
            yticklabels=['No Defect', 'Defect'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.show()

print("Classification Report")
print(classification_report(y_val, y_pred, target_names=['No Defect', 'Defect']))

model_name = 'aitex_simple_classifier.h5'
model.save(model_name)
print(f"\nModel saved as: {model_name}")