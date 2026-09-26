import os, random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

base = "D:/AITexProject/aitex_dataset"
img_dir = os.path.join(base, "Defect_images")
mask_dir = os.path.join(base, "Mask_images")

# list all defect images
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

# pick a random image
img_file = random.choice(img_files)
img_path = os.path.join(img_dir, img_file)

# construct its corresponding mask filename
mask_file = img_file.replace(".png", "_mask.png")
mask_path = os.path.join(mask_dir, mask_file)

# check existence
if not os.path.exists(mask_path):
    print(f"No mask found for {img_file}")
else:
    img = Image.open(img_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    img_a = np.array(img)
    mask_a = np.array(mask) > 0  # anything > 0 = defect area

    print("Image:", img_file)
    print("Mask:", mask_file)
    print("Unique mask values:", np.unique(np.array(mask)))
    print("White pixel ratio (>0):", np.mean(mask_a))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_a, cmap='gray'); plt.title('Image')
    plt.subplot(1,3,2); plt.imshow(mask_a, cmap='gray'); plt.title('Mask (binary)')
    plt.subplot(1,3,3); plt.imshow(img_a, cmap='gray'); plt.imshow(mask_a, alpha=0.4, cmap='jet'); plt.title('Overlay')
    plt.show()
