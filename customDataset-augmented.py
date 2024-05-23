import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from collections import Counter
import random
import os

class AugmentImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_transform=None, augment_factor=5):
        self.root_dir = root_dir
        self.transform = transform
        self.augment_transform = augment_transform if augment_transform else transform
        self.augment_factor = augment_factor
        self.data = []
        self.augmented_count ={}
        self.class_to_idx = {}
        self.load_and_balance_dataset()

    def load_and_balance_dataset(self):
        class_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        class_counts = Counter()

        # Count the number of images in each class
        for class_idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = class_idx
            class_dir = os.path.join(self.root_dir, class_name)
            num_images = len(os.listdir(class_dir))
            class_counts[class_idx] = num_images

        max_count = max(class_counts.values())

        # Load images and augment to balance the dataset
        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(self.root_dir, class_name)
            images = os.listdir(class_dir)
            num_images = len(images)

            # Load all the original images
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.data.append((img_path, class_idx, False))

            # Augment images to balance the classes
            if num_images < max_count:
                num_augments = (max_count - num_images) * self.augment_factor
                self.augmented_count[class_name] = num_augments
                for _ in range(num_augments):
                    img_name = random.choice(images)
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append((img_path, class_idx, True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, is_augmented = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if is_augmented:
            if self.augment_transform:
                image = self.augment_transform(image)
        else:
            if self.transform:
                image = self.transform(image)
        return image, label
