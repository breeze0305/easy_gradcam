# === data preprocess === 
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# === model (maybe choose one?) ===
import torch
import timm

# === this visualization tool ===
from easy_gradcam.classification import EasyGradCAM
from easy_gradcam.visualization import save_heatmap, save_mix_heatmap


# -----------------------------
# 1. 載入模型
# -----------------------------
from model import Model  # your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example : ResNet-18 (from timm, finetune by author)
"""
Dataset link:https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification
"""
model = Model()
state_dict = torch.load("./runs/concrete_crack_model.pth")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
# print(model)
# exit()

# -----------------------------
# 2. 準備圖片
# -----------------------------

img_path = Path("./test_img/test1.jpg")  # change to your image path
# 方式一: OpenCV
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

totensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

t = totensor(img)
t = resize(t)
t = normalize(t)
t = t.unsqueeze(0)  # add batch dimension
t = t.to(device)
# -----------------------------
# 3. 計算 Grad-CAM
# -----------------------------
gradcam = EasyGradCAM(model, targets=["conv1", "conv2"]) 

# Extract features and gradients
feats, grads = gradcam.cal_feat_and_grad(t)

# Generate heatmaps
heats = gradcam.cal_heats(img, feats, grads)


# -----------------------------
# 4. 儲存結果
# -----------------------------
output_path = Path("results")
output_path.mkdir(parents=True, exist_ok=True)

for i in range(len(heats)):
    for name in heats[i]:
        # Save plain heatmap
        save_heatmap(
            save_path=f"{output_path}/{img_path.stem}-{i}-{name}.jpg",
            heat=heats[i][name],
            cmap="jet",
            title="grad-cam"
        )

        # Save overlay with original image
        save_mix_heatmap(
            save_path=f"{output_path}/{img_path.stem}-{i}-{name}-mix.jpg",
            heat=heats[i][name],
            ori_img=img,
            cmap="jet"
        )

