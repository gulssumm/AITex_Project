import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# === PATH ===
data_dir = "D:/AITexProject/aitex_raw_clean"

# === LOAD DATASETS ===
img_size = (224, 224)
batch_size = 16

train_ds = image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",  # convert grayscale to RGB automatically
)

val_ds = image_dataset_from_directory(
    os.path.join(data_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
)

# === NORMALIZATION ===
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# === PERFORMANCE SETTINGS ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === SIMPLE CNN MODEL ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === CHECK HOW MANY IMAGES THE DATASET HAVE PER CLASS ===
base = "D:/AITexProject/aitex_dataset_processed/train"
for cls in ["Defect", "NoDefect"]:
    count = len(os.listdir(os.path.join(base, cls)))
    print(cls, ":", count)

# === TRAIN MODEL ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# === PLOT TRAINING CURVES ===
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# === SAVE MODEL ===
model.save("aitex_defect_classifier.h5")
print("Model training complete and saved as aitex_defect_classifier.h5")
