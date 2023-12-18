import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image

class Base(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = os.listdir(image_folder)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(os.path.join(self.image_folder, image_path)).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)