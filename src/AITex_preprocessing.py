import os

# FOLDER STRUCTURE
base_path = "D:/AITexProject/aitex_dataset"
folders = ["Defect_images", "NODefect_images", "Mask_images"]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    count = 0
    for root, dirs, files in os.walk(folder_path):
        count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"{folder}: {count} images")

no_defect_path = os.path.join(base_path, "NODefect_images")
for root, dirs, files in os.walk(no_defect_path):
    print("Folder:", root)
    if dirs:
        print("Subfolders:", dirs)
    break  # only show the first level


# IMAGE PROPERTIES
from PIL import Image

def check_properties(folder_path):
    sizes = []
    modes = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                with Image.open(path) as img:
                    sizes.append(img.size)
                    modes.append(img.mode)
    print("Unique sizes:", set(sizes))
    print("Color modes:", set(modes))

print("\nDefect images:")
check_properties(os.path.join(base_path, "Defect_images"))
print("\nNoDefect images:")
check_properties(os.path.join(base_path, "NODefect_images"))


# CLASS DISTRIBUTION
import matplotlib.pyplot as plt

labels = ['Defect', 'No Defect']
counts = [106, 141]
plt.bar(labels, counts)
plt.title("Class Distribution")
plt.show()
