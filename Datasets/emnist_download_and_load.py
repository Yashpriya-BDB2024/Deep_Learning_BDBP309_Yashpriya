import torch
from torchvision import datasets, transforms
from PIL import Image
import os
from tqdm import tqdm

# --- Transformation ---
transform = transforms.ToTensor()

# --- Choose a split, e.g., letters ---
split = "letters"
train_set = datasets.EMNIST(root='./data', split=split, train=True, download=True, transform=transform)
test_set = datasets.EMNIST(root='./data', split=split, train=False, download=True, transform=transform)

# --- Create output folders ---
output_dir = f"emnist_{split}_dataset"
for subset_name, dataset in [("train", train_set), ("test", test_set)]:
    for i in range(1 if split == "letters" else len(dataset.classes)):
        pass  # We'll dynamically create folders below
    os.makedirs(os.path.join(output_dir, subset_name), exist_ok=True)

# --- Label names ---
if split == "letters":
    label_names = [chr(i + 64) for i in range(1, 27)]  # A-Z
else:
    label_names = [str(i) for i in range(len(dataset.classes))]

# --- Save images ---
def save_all_images(dataset, subset_name):
    for i in tqdm(range(len(dataset)), desc=f"Saving {subset_name} images"):
        img, label = dataset[i]
        img_pil = Image.fromarray((img.squeeze().numpy() * 255).astype('uint8'))
        label_name = label_names[label-1] if split == "letters" else label_names[label]
        folder = os.path.join(output_dir, subset_name, label_name)
        os.makedirs(folder, exist_ok=True)
        img_pil.save(os.path.join(folder, f"{subset_name}_{i:05d}.png"))

save_all_images(train_set, "train")
save_all_images(test_set, "test")

print("✅ Done! Check directory:", os.path.abspath(output_dir))
