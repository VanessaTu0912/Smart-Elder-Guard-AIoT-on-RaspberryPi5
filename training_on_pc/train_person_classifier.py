import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np

# =========================
# 路徑設定
# =========================

DATASET_DIR = r"C:\Users\Vanessa\Desktop\person_dataset\person_classifier_dataset"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

SAVE_PATH = r"C:\Users\Vanessa\Desktop\person_dataset\target_person_classifier.pth"

# =========================
# 參數
# =========================

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0003
IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# Transform
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# Dataset
# =========================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# Model
# =========================

weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(weights=weights)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)

model = model.to(DEVICE)

# =========================
# Loss / Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# =========================
# Training
# =========================

train_acc_list = []
val_acc_list = []

train_loss_list = []
val_loss_list = []

best_val_acc = 0.0

for epoch in range(EPOCHS):

    # =========================
    # Train
    # =========================

    model.train()

    train_correct = 0
    train_total = 0
    train_loss_sum = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        train_loss_sum += loss.item() * labels.size(0)

    train_acc = train_correct / train_total
    train_loss = train_loss_sum / train_total

    # =========================
    # Validation
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    y_true = []
    y_pred = []
    y_prob = []

    wrong_samples = []

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(val_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_loss_sum += loss.item() * labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            # 找出錯誤樣本
            for i in range(len(labels)):

                global_index = batch_idx * BATCH_SIZE + i

                if global_index >= len(val_dataset.samples):
                    continue

                if preds[i].item() != labels[i].item():

                    img_path = val_dataset.samples[global_index][0]

                    true_name = train_dataset.classes[labels[i].item()]
                    pred_name = train_dataset.classes[preds[i].item()]
                    confidence = probs[i][preds[i]].item()

                    wrong_samples.append(
                        (img_path, true_name, pred_name, confidence)
                    )

    val_acc = val_correct / val_total
    val_loss = val_loss_sum / val_total

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print("-" * 40)

    # 只儲存目前 val acc 最好的模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Best model saved. Val Acc = {best_val_acc:.4f}")
        print("-" * 40)

# =========================
# Final Report
# =========================

print("\n========== Final Validation Report ==========")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=train_dataset.classes
))

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_dataset.classes
)

disp.plot(cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.show()

# =========================
# 每一類 Accuracy
# =========================

print("\nPer-class Validation Accuracy:")

class_acc = cm.diagonal() / cm.sum(axis=1)

for class_name, acc in zip(train_dataset.classes, class_acc):
    print(f"{class_name}: {acc:.4f}")

# =========================
# 被認錯的圖片
# =========================

print("\n========== Wrong Samples ==========")

if len(wrong_samples) == 0:
    print("No wrong samples. Validation all correct.")
else:
    for path, true_name, pred_name, conf in wrong_samples:
        print(f"File: {path}")
        print(f"True: {true_name}")
        print(f"Pred: {pred_name}")
        print(f"Confidence: {conf:.4f}")
        print("-" * 60)

# =========================
# Plot Accuracy
# =========================

plt.figure()
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training / Validation Accuracy")
plt.show()

# =========================
# Plot Loss
# =========================

plt.figure()
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training / Validation Loss")
plt.show()

print("\nTraining finished.")
print(f"Best Val Acc: {best_val_acc:.4f}")
print(f"Best model saved to: {SAVE_PATH}")