from pathlib import Path

import typer

from datasets import load_dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from djtt_gpt.config import RAW_DATA_DIR, HF_PATH_DIR, DATASET_SUBSET, TABLE_GPT_DIR_TRAIN

import pandas as pd

def main(
    input_path: Path = HF_PATH_DIR,
    output_path: Path = RAW_DATA_DIR,
    data_subset: str = DATASET_SUBSET
):
    
    train_dataset = load_dataset(f"{input_path}", f"{data_subset}", split="train", cache_dir=f"{output_path}")
    test_dataset = load_dataset(f"{input_path}", f"{data_subset}", split="test", cache_dir=f"{output_path}")

# define transformations: resize to 224x224 and normalize
transform = transforms.Compose([
transforms.Grayscale(num_output_channels=3),  # convert grayscale to RGB
transforms.Resize((224, 224)),  # resize to 224x224
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image():
    # load the Fashion MNIST dataset
    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader

if __name__ == "__main__":
    print("din far")