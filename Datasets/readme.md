# CIFAR10 Dataset
# https://www.cs.toronto.edu/%7Ekriz/cifar-10-python.tar.gz
# To extract -

import os
from PIL import Image
from tqdm import tqdm

# Base output directory 
output_dir = "cifar10_dataset"
os.makedirs(output_dir, exist_ok=True)

# Create subfolders for train/test and class names
for split in ["train", "test"]:
    for label in label_names:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

def save_images(images, labels, split_name):
    for i in tqdm(range(len(images)), desc=f"Saving {split_name} images"):
        img = Image.fromarray(images[i])
        label = label_names[labels[i]]
        filename = f"{split_name}_{i:05d}.png"
        img.save(os.path.join(output_dir, split_name, label, filename))

save_images(train_images, train_labels, "train")
save_images(test_images, test_labels, "test")
print("Done! Images saved to:", os.path.abspath(output_dir))


