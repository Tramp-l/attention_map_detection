from torchvision import datasets, transforms

def preprocess_data(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return dataset


