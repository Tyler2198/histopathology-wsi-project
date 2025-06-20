import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    load_and_resize_images,
    normalize_images,
    build_tensor_dataset,
    split_dataset,
)

from model_baseline import BaselineCNN
from model_resnet import ResNetBinary


def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            true.extend(labels.numpy())
    preds = np.array(preds)
    true = np.array(true)
    auc = roc_auc_score(true, preds)
    acc = np.mean((preds > 0.5) == true)
    return preds, true, auc, acc


def plot_confusion_matrix(y_true, y_pred, save_path="plots/confusion_matrix.png"):
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    df = pd.read_csv("train_labels.csv")

    # Load and preprocess data
    X, y = load_and_resize_images(df, max_samples=args.samples, img_size=args.img_size)
    X = normalize_images(X)
    _, X_val, _, y_val = split_dataset(X, y)

    val_dataset = build_tensor_dataset(X_val, y_val, resnet=(args.model == "resnet"))
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Load model
    if args.model == "baseline":
        model = BaselineCNN(input_size=args.img_size)
    elif args.model == "resnet":
        model = ResNetBinary(pretrained=False, freeze_base=False)
    else:
        raise ValueError("Unknown model name")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Evaluate
    preds, true, auc, acc = evaluate(model, val_loader, device)
    print(f"\n📈 Validation AUC: {auc:.4f} | Accuracy: {acc:.4f}")

    # Plot
    plot_confusion_matrix(true, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "resnet"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--img_size", type=int, default=96)
    args = parser.parse_args()
    main(args)
