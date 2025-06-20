# Histopathologic Cancer Detection using Deep Learning

This project detects metastatic tissue in histopathologic images using CNNs and transfer learning, based on the Kaggle challenge.

## 🔬 Goal
Build scalable and interpretable models for binary cancer classification from WSIs.

## 📁 Dataset
- [Kaggle: Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
- 96x96 image tiles with cancer/no cancer labels.

## 🧠 Models

### ✅ Baseline CNN (PyTorch)
- Input: 96x96 RGB
- Accuracy: 79.5%
- AUC: 0.87

### ✅ ResNet18 (Transfer Learning)
- Input: 224x224, ImageNet normalized
- Accuracy: **86.1%**
- AUC: **0.928**

## 📊 Results
- ROC AUC used to account for class imbalance.
- ResNet significantly outperforms baseline.
- Clean modular code for reuse.

## 💡 Tools
- PyTorch, TorchVision, PIL, NumPy
- Kaggle CLI, Matplotlib, Scikit-Learn

## 🧬 Relevance
This pipeline reflects real-world digital pathology workflows and oncology applications — aligning with the aims of Roche pRED Basel.
