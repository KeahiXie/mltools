import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from collections import Counter

def load_data(train_dir=None, test_dir=None, transform=None, augment_transform=None, augment_factor=5, config=None, mode="original", dataset_class=None):
    if train_dir is None or test_dir is None or config is None:
        raise ValueError("Parameters 'train_dir', 'test_dir', and 'config' must be provided.")
    
    if mode not in ["original", "weighted", "augmented"]:
        raise ValueError("Mode must be one of 'original', 'weighted', or 'augmented'.")
    
    if mode == "augmented" and dataset_class is None:
        raise ValueError("Parameter 'dataset_class' must be provided for 'augmented' mode.")

    # Default to ImageFolder for original and weighted modes if no custom dataset_class is provided
    if dataset_class is None:
        dataset_class = ImageFolder
    
    train_transform = transform
    test_transform = transform

    # Load datasets
    if mode == "augmented":
        train_dataset = dataset_class(root_dir=train_dir, transform=train_transform, augment_transform=augment_transform, augment_factor=augment_factor)
    else:
        train_dataset = dataset_class(train_dir, transform=train_transform)
    
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    
    if mode == "original":
        train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
    
    elif mode == "weighted":
        class_weights = calculate_class_weights(train_dataset)
        sample_weights = assign_sample_weights(train_dataset, class_weights)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
    
    elif mode == "augmented":
        print(f"Overview of the dataset after augmentation:")
        total_count = Counter({key: train_dataset.original_count[key] + train_dataset.augmented_count[key] for key in train_dataset.original_count})
        for class_name in train_dataset.original_count:
            print(f"{class_name}: {total_count[class_name]}")
        
        train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)

    class_to_idx = train_dataset.class_to_idx
    return train_dataloader, test_dataloader, class_to_idx

def calculate_class_weights(dataset):
    class_counts = Counter([label for _, label in dataset])
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    return class_weights

def assign_sample_weights(dataset, class_weights):
    sample_weights = [class_weights[label] for _, label in dataset]
    return sample_weights
