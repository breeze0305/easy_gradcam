import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ConcreteCrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        classes = ["healthy", "crack"]
        for idx, cls in enumerate(classes):
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                if img_name.endswith((".jpg", ".png")):
                    self.images.append(os.path.join(cls_folder, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
