import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import albumentations as A

class Dataset():
    def __init__(self, path: str = "compressed_dataset", image_size: Tuple[int, int] = (224, 224), test_split: float = 0.2, val_split: float = 0.1, augment: bool = True):
        self.path = path
        self.image_size = image_size
        self.test_split = test_split
        self.val_split = val_split
        self.augment = augment
        self.augmenter = self._build_augmenter()
        self._split_data()

    def _build_augmenter(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ])

    def _load_images(self) -> List[np.ndarray]:
        data = []
        for img_name in os.listdir(self.path):
            if not img_name.lower().endswith('.jpg'):
                continue
            img_path = os.path.join(self.path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img)
                data.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        return data

    def _split_data(self):
        full_data = self._load_images()
        X_trainval, self.X_test = train_test_split(
            full_data, test_size=self.test_split, random_state=42
        )
        self.X_train, self.X_val = train_test_split(
            X_trainval, test_size=self.val_split, random_state=42
        )

        if self.augment:
            augmented = [self.augmenter(image=img)['image'] for img in self.X_train]
            self.X_train.extend(augmented)

    def train(self):
        return np.array(self.X_train)

    def test(self):
        return np.array(self.X_test)

    def val(self):
        return np.array(self.X_val)
