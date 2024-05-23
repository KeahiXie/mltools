import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler, ImageFolder

def load_data(train_dir=None,
              test_dir=None,
              transform=None,
              augment_transform=None,
              augment_factor=5,
              config=None,
              mode="original",
              dataset_class=None):
    """
    Load datasets and create data loaders based on the specified mode.
    
    Parameters:
    - train_dir (str): Directory for training data.
    - test_dir (str): Directory for testing data.
    - transform (callable): Default transformations to apply to both train and test data.
    - train_transform (callable): Transformations to apply to the training data.
    - test_transform (callable): Transformations to apply to the testing data.
    - config (dict): Configuration dictionary containing batch size.
    - mode (str): Mode of DataLoader to return. One of 'original', 'weighted', 'augmented'.
    - dataset_class (class): Custom dataset class to use for the 'augmented' mode.
    
    Returns:
    - Depending on the mode:
      - 'original': (original_train_dataloader, original_test_dataloader, class_to_idx)
      - 'weighted': (weighted_train_dataloader, original_test_dataloader, class_to_idx)
      - 'augmented': (augmented_train_dataloader, augmented_test_dataloader, class_to_idx)
    """
    
    if train_dir is None or test_dir is None or config is None:
        raise ValueError("Parameters 'train_dir', 'test_dir', and 'config' must be provided.")
    
    if mode not in ["original", "weighted", "augmented"]:
        raise ValueError("Mode must be one of 'original', 'weighted', or 'augmented'.")
    
    if mode == "augmented" and dataset_class is None:
        raise ValueError("Parameter 'dataset_class' must be provided for 'augmented' mode.")
    
    # Select the appropriate dataset class
    if mode in ["original", "weighted"] :
        dataset_class = torchvision.datasets.ImageFolder
    
    # Set transformations
    if transform is not None:
        train_transform = transform
        test_transform = transform
    
    # Load the datasets
    train_dataset = dataset_class(train_dir, transform=train_transform)
    test_dataset = dataset_class(test_dir, transform=test_transform)
    
    if mode == "original":
        # Create DataLoaders for the original datasets
        train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
        class_to_idx = train_dataset.class_to_idx
        return train_dataloader, test_dataloader, class_to_idx
    
    elif mode == "weighted":
        # Calculate class weights for the original train dataset
        class_weights = calculate_class_weights(train_dataset)
        sample_weights = assign_sample_weights(train_dataset, class_weights)
        
        # Create DataLoader with weighted sampling for the training dataset
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
        class_to_idx = train_dataset.class_to_idx
        return train_dataloader, test_dataloader, class_to_idx
    
    elif mode == "augmented":
        #
        augmented_train_dataset = dataset_class(root_dir=train_dir, 
                                                transform=train_transform,
                                                augment_transform=augment_transform,
                                                augment_factor=augment_factor)
        augmented_test_dataset = ImageFolder(test_dir, transform=test_transform)
        # Create DataLoaders for the augmented dataset
        train_dataloader = DataLoader(augmented_train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
        test_dataloader = DataLoader(augmented_test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
        class_to_idx = train_dataset.class_to_idx
        return train_dataloader, test_dataloader, class_to_idx

def calculate_class_weights(dataset):
    """Calculate class weights based on the entire dataset."""
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    return class_weights

def assign_sample_weights(dataset, class_weights):
    """Assign sample weights to the dataset based on class weights."""
    sample_weights = [0] * len(dataset)
    for idx, (_, label) in enumerate(dataset):
        sample_weights[idx] = class_weights[label]
    return sample_weights
