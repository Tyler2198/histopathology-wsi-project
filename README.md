# 🧬 Histopathologic Cancer Detection with Deep Learning

This project implements deep learning models for binary classification of metastatic tissue in histopathologic images. It is inspired by the [Kaggle Histopathologic Cancer Detection Challenge](https://www.kaggle.com/competitions/histopathologic-cancer-detection) and demonstrates practical, scalable methods for digital pathology in oncology — a field of key interest in precision medicine.

---

## 🎯 Project Goal

To develop a clean, interpretable pipeline for classifying histopathologic Whole Slide Image (WSI) patches as cancerous or non-cancerous using:

- Baseline CNNs trained from scratch
- Transfer learning with pretrained ResNet18
- Full data preprocessing, normalization, and evaluation

---

## 🗂 Dataset

- **Source**: Kaggle Competition
- **Images**: 96x96 `.tif` RGB tiles extracted from WSIs
- **Labels**: 0 = no cancer, 1 = metastatic cancer
- **Size Used**: 10,000 samples (balanced across classes)

---

## 🧠 Models

### ✅ Baseline CNN (PyTorch)
- 2 convolutional layers + max pooling
- Fully connected classifier with dropout
- Input: 96x96 images
- Performance:
  - Accuracy: **79.5%**
  - AUC: **0.87**

### ✅ ResNet18 (Transfer Learning)
- Pretrained on ImageNet
- Final FC layers replaced for binary output
- Inputs resized to 224x224 + normalization
- Performance:
  - Accuracy: **86.1%**
  - AUC: **0.928**

---

## 📈 Results

| Model        | Accuracy | ROC AUC |
|--------------|----------|---------|
| Baseline CNN | 79.5%    | 0.870   |
| ResNet18     | 86.1%    | 0.928   |

---

## 🧪 Tools & Libraries

- Python, PyTorch, Torchvision
- NumPy, Pandas, Scikit-learn
- PIL, Matplotlib, Seaborn
- Kaggle CLI

---

## 💡 Relevance to Roche

This project simulates a real-world digital pathology use case in oncology. It reflects the challenges Roche teams face in capturing cellular patterns from Whole Slide Images (WSIs), particularly for scalable and interpretable deep learning analysis.

---

## 🚀 Getting Started

```bash
# Clone repo
git clone https://github.com/your_username/histopathology-cancer-prediction
cd histopathology-cancer-prediction

# Install dependencies
pip install -r requirements.txt

# Train baseline CNN
python train.py --model baseline

# Train ResNet18
python train.py --model resnet

# Evaluate saved model
python eval.py --model resnet --model_path resnet_model.pth
```
