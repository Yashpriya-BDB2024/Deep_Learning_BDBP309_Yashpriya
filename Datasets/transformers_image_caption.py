
# ##### Transformers (Flickr8k image caption) ####

import torch
from torch import nn
import torch.nn.functional as F    # Functional API for activations, loss ops, etc.
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os, random, json    # os: for path manipulation, file listings, json: module to read/write JSON files
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class Config:
    path_to_text = "flicker8k/captions.txt"
    path_to_images = "flicker8k/Images"
    maxlen = 38
    image_size = 256    # Resized
    batch_size = 32
    epochs = 8
    lr = 1e-4
    d_model = 512    # embedding dimension
    n_heads_enc = 8   # no. of attention heads in encoder
    n_heads_dec = 8   # no. of attention heads in decoder
    enc_layers = 1
    dec_layers = 2
    min_token_freq = 1   # min. word frequency to include in vocab, i.e., words appearing less than 1 will be skipped.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_captions(path_to_text):








def main():
    path_to_text = "flicker8k/captions.txt"
    path_to_images = "flicker8k/Images"
    df = pd.read_csv(path_to_text, sep = ",")
    print(f"Total images: {df['image'].nunique()}")   # Total images: 8091
    img_path = "flicker8k/Images/101654506_8eb26cfb60.jpg"
    img = Image.open(img_path)
    print(img.size)    # Images are of varying sizes, so need to resize it to 256
    print(f"Total captions: {df['caption'].nunique()}")   # Total captions: 40201
    with open("flicker8k/captions.txt", "r") as f:
        data = f.readlines()[1:]  # skip header if CSV
    lengths = []
    for line in data:
        _, caption = line.strip().split(",", 1)  # split only on first comma
        lengths.append(len(caption.split()))
    print("Maximum caption length:", max(lengths))    # Max. length: 38
    print("Average caption length:", sum(lengths) / len(lengths))    # Avg. length: 11.78

if __name__ == "__main__":
    main()

