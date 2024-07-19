import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from djtt_gpt.dataset import load_image

# load dataset 
trainloader, testloader = load_image()

# Load the pre-trained ResNet50 model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Remove the final classification layer
modules = list(resnet_model.children())[:-1]
resnet_model = torch.nn.Sequential(*modules)

# run a singular batch to see the output shape
def extract_embeddings_sing(trainloader):
    for images, _ in trainloader:
        with torch.no_grad():
            embeddings_batch = resnet_model(images)
            embeddings_batch = embeddings_batch.view(embeddings_batch.size(0), -1)
        break  # break after the first batch
    return embeddings_batch

# function to extract features (embeddings)
def extract_embeddings(dataloader):
    all_embeddings = []
    for images, _ in dataloader:
        with torch.no_grad():
            output = resnet_model(images)
            output = output.view(output.size(0), -1)
            all_embeddings.append(output.cpu().numpy())
    return np.vstack(all_embeddings)

# Extract embeddings for training and test data
train_embeddings = extract_embeddings_sing(trainloader)
test_embeddings = extract_embeddings_sing(testloader)

print(f"Training embeddings shape: {train_embeddings.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")



