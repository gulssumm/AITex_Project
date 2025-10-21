import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# === PATH ===
data_dir = "D:/AITexProject/aitex_dataset_processed"

# === SETTINGS ===
img_size = (224, 224)
batch_size = 8
AUTOTUNE = tf.data.AUTOTUNE

# === LOAD DATASETS ===
train_ds = image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
)

val_ds = image_dataset_from_directory(
    os.path.join(data_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
)

# === DATA AUGMENTATION ===
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# === NORMALIZATION ===
normalization_layer = layers.Rescaling(1./255)

def preprocess(x, y):
    x = data_augmentation(x)
    x = normalization_layer(x)
    return x, y

train_ds = train_ds.map(preprocess).cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

# === PRETRAINED BASE MODEL ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze pretrained layers

# === MODEL HEAD ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === TRAIN MODEL ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# === MISCLASSIFICATION ===
pred_probs = model.predict(val_ds)  # outputs probabilities
pred_labels = (pred_probs >= 0.5).astype(int)  # convert to 0 or 1
true_labels = np.concatenate([y for x, y in val_ds], axis=0)
misclassified_indices = np.where(pred_labels.flatten() != true_labels)[0]
print("Number of misclassified images:", len(misclassified_indices))

val_dir = "D:/AITexProject/aitex_dataset_processed/val"
classes = ["Defect", "NoDefect"]

filenames = []
for cls in classes:
    cls_dir = os.path.join(val_dir, cls)
    cls_files = sorted(os.listdir(cls_dir))
    filenames.extend([os.path.join(cls_dir, f) for f in cls_files])

# Prepare misclassified info
misclassified_probs = pred_probs.flatten()[misclassified_indices]
misclassified_true = true_labels[misclassified_indices]
misclassified_files = [filenames[i] for i in misclassified_indices]

# --- Grid display ---
num_imgs = len(misclassified_files)
cols = 4
rows = math.ceil(num_imgs / cols)

plt.figure(figsize=(16, 4*rows))
for i, (file, true, prob) in enumerate(zip(misclassified_files, misclassified_true, misclassified_probs)):
    plt.subplot(rows, cols, i+1)
    img = mpimg.imread(file)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {true}\nPred: {prob:.2f}")

plt.tight_layout()
plt.show()


# === PLOT TRAINING CURVES ===
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Model Accuracy')

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# === SAVE MODEL ===
model.save("aitex_defect_classifier_pretrained.h5")
print("Model training complete and saved as aitex_defect_classifier_pretrained.h5")
