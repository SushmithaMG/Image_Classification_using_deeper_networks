import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset_name, dataset_dir, batch_size, input_size, train_split=0.9):
    if dataset_name.upper() == 'MNIST':
        transform_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_dataset = datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose MNIST")

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader