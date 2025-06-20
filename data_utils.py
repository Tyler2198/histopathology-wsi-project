import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from tqdm import tqdm


def load_and_resize_images(df, img_dir="train", img_size=96, max_samples=None):
    """Load and resize images from directory"""
    X = []
    y = []

    if max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(img_dir, row["id"] + ".tif")
        img = Image.open(img_path).resize((img_size, img_size))
        arr = np.array(img)
        if arr.ndim == 2:  # if grayscale, make it RGB-like
            arr = np.stack([arr]*3, axis=-1)
        X.append(arr)
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)
    return X, y


def normalize_images(X):
    """Normalize pixel values to [0, 1]"""
    return X / 255.0


def build_tensor_dataset(X, y, resnet=False):
    """Convert numpy arrays to PyTorch dataset"""
    if resnet:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    X_torch = torch.stack([transform(Image.fromarray((img * 255).astype(np.uint8))) for img in X])
    y_torch = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_torch, y_torch)


def split_dataset(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
