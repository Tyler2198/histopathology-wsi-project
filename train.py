import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data_utils import (
    load_and_resize_images,
    normalize_images,
    build_tensor_dataset,
    split_dataset,
)

from model_baseline import BaselineCNN
from model_resnet import ResNetBinary

import pandas as pd
import numpy as np


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            true.extend(labels.numpy())
    auc = roc_auc_score(true, preds)
    acc = np.mean((np.array(preds) > 0.5) == np.array(true))
    return auc, acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load labels
    df = pd.read_csv("train_labels.csv")

    # Load and preprocess images
    print("Loading and resizing images...")
    X, y = load_and_resize_images(df, max_samples=args.samples, img_size=args.img_size)
    X = normalize_images(X)
    X_train, X_val, y_train, y_val = split_dataset(X, y)

    # Dataset prep
    print(f"Building datasets with model: {args.model}")
    resnet_flag = args.model == "resnet"
    train_dataset = build_tensor_dataset(X_train, y_train, resnet=resnet_flag)
    val_dataset = build_tensor_dataset(X_val, y_val, resnet=resnet_flag)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Load model
    if args.model == "baseline":
        model = BaselineCNN(input_size=args.img_size).to(device)
    elif args.model == "resnet":
        model = ResNetBinary(pretrained=True, freeze_base=True).to(device)
    else:
        raise ValueError("Unknown model name")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        loss = train(model, train_loader, optimizer, criterion, device)
        auc, acc = evaluate(model, val_loader, device)
        print(f"Loss: {loss:.4f} | Val AUC: {auc:.4f} | Val Acc: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), f"{args.model}_model.pth")
    print(f"\nModel saved as {args.model}_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
