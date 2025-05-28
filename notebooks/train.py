import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset import RetinaDataset  # you'll write this

def train_model(model, dataloaders, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")
    return model

