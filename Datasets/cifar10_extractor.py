import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.array(labels)

data_dir = "C:\\Users\\Public\\Downloads\\Pycharm_projects\\Deep_Learning\\data\\cifar-10-batches-py"

train_images, train_labels = [], []
for i in range(1, 6):
    imgs, lbls = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
    train_images.append(imgs)
    train_labels.append(lbls)

train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

test_images, test_labels = load_cifar10_batch(os.path.join(data_dir, "test_batch"))

meta = pickle.load(open(os.path.join(data_dir, "batches.meta"), 'rb'), encoding='bytes')
label_names = [x.decode('utf-8') for x in meta[b'label_names']]

# print("Train images shape:", train_images.shape)
# print("Train labels shape:", train_labels.shape)
# print("Test images shape:", test_images.shape)
# print("Test labels shape:", test_labels.shape)
# print("Labels:", label_names)

output_dir = "C:\\Users\\Public\\Downloads\\Pycharm_projects\\Deep_Learning\\data"
os.makedirs(output_dir, exist_ok=True)

for split in ["train", "test"]:
    for label in label_names:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

def save_images(images, labels, split_name):
    for i in tqdm(range(len(images)), desc=f"Saving {split_name} images"):
        img = Image.fromarray(images[i])
        label = label_names[labels[i]]
        filename = f"{split_name}_{i:05d}.png"
        img.save(os.path.join(output_dir, split, label, filename))

save_images(train_images, train_labels, "cifar10_train")
save_images(test_images, test_labels, "cifar10_test")

print("Done! Images saved to:", os.path.abspath(output_dir))
