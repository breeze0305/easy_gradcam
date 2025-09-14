# easy_gradcam

一個輕量級工具，用於為影像分類模型生成 Grad-CAM 視覺化。
它支援常見的骨幹網路，如 **ResNet**、**Vision Transformers (ViT)** 和 **Hugging Face Transformers**。

此套件基於 PyTorch 建立，並提供由 Matplotlib 與 Seaborn 驅動的可視化功能。
設計目的是幫助使用者輕鬆地訓練、評估與視覺化結果，並產生清晰且可自訂的圖表。

---

## 安裝
[PyPI 連結](https://pypi.org/project/easy-gradcam/)
```bash
pip install easy_gradcam
```

## 從原始碼編譯

若你想安裝最新的開發版本（或修改程式碼後安裝），可以直接從 GitHub 編譯安裝：
```bash
git clone https://github.com/breeze0305/easy_gradcam
cd easy_gradcam
```

安裝編譯工具：
```bash
pip install build twine setuptools wheel
```

然後執行以下指令進行編譯與檢查：
```bash
python setup.py sdist bdist_wheel
twine check dist/*
```

最後安裝編譯好的套件：
```bash
pip install dist/easy_gradcam-x.x.x-py3-none-any.whl
```

## 快速開始

### 1. 匯入依賴
```python
# === 資料前處理 === 
import cv2
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

# === 模型（可擇一使用） ===
import torchvision.models as models
import timm
from transformers import AutoModelForImageClassification

# === 本工具 ===
from easy_gradcam.classification import EasyGradCAM
from easy_gradcam.visualization import save_heatmap, save_mix_heatmap
```

### 2. 載入模型
你可以使用不同的骨幹模型：
```python
# 範例 1: ResNet-50 (torchvision 提供)
model = models.resnet50(pretrained=True)   # 目標層: "layer4"

# 範例 2: ViT (timm 提供)
model = timm.create_model("vit_base_patch16_224_miil", pretrained=True)   # 目標層: "blocks.10"

# 範例 3: DINOv2 (Hugging Face 提供)
model = AutoModelForImageClassification.from_pretrained(
    "facebook/dinov2-small-imagenet1k-1-layer"
) # 目標層: "dinov2.encoder.layer.11"

# 範例 4: 自訂模型
# model = CustomModel(...)

model.eval()
```

### 2.1 確認目標層名稱
要找到模型正確的層名稱，可以直接列印模型結構：
```python
print(model)
```

### 3. 準備輸入圖片
```python
img_path = Path("./test_img/test5.jpg")
# 方法 1. 使用 OpenCV 讀取圖片
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 方法 2. 使用 Pillow 讀取圖片
img = Image.open(img_path).convert("RGB")
img = np.array(img)

totensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

t = totensor(img)
t = resize(t)
t = normalize(t)
t = t.unsqueeze(0)  # 增加 batch 維度
```

### 4. 計算 Grad-CAM
```python
# 範例 1: 單一目標層
gradcam = EasyGradCAM(model, targets="dinov2.encoder.layer.11")

# 範例 2: 多個目標層
gradcam = EasyGradCAM(model, targets=["dinov2.encoder.layer.10", "dinov2.encoder.layer.11"])  

# 提取特徵與梯度
feats, grads = gradcam.cal_feat_and_grad(t)

# 生成熱力圖
heats = gradcam.cal_heats(img, feats, grads)
```

### 5. 儲存結果
```python
output_path = Path("results")
output_path.mkdir(parents=True, exist_ok=True)

for i in range(len(heats)):
    for name in heats[i]:
        # 儲存純熱力圖
        save_heatmap(
            save_path=f"{output_path}/{img_path.stem}-{i}-{name}.jpg",
            heat=heats[i][name],
            cmap="jet",
            title="grad-cam"
        )

        # 儲存與原圖疊合的熱力圖
        save_mix_heatmap(
            save_path=f"{output_path}/{img_path.stem}-{i}-{name}-mix.jpg",
            heat=heats[i][name],
            ori_img=img,
            cmap="jet"
        )
```

### 範例輸出
- test_img/test5.jpg: 原始輸入圖片
- results/test5-0-conv2.jpg: 純熱力圖
- results/test5-0-conv2-mix.jpg: 疊合熱力圖

### 注意事項
* 請確認設定的目標層名稱符合模型內部結構。
* 支援來自 torchvision、timm 與 Hugging Face 的預訓練模型。
* 熱力圖會以 .jpg 格式儲存在 `results/` 資料夾內。

### 問題/需求
請透過 [GitHub issue tracker](https://github.com/breeze0305/easy_gradcam/issues) 回報問題與需求。
