import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from torchvision.datasets import ImageFolder

def train_model(model, criterion, optimizer, train_loader, device):
    # Training loop
    for epoch in range(5):  # adjust epochs
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def preprocess_image(path):
    #Preprocessing the image for use to predict
    # Using cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Option A: using PIL
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # add batch dimension
    tensor = tensor.to(device)
    return tensor

def predict_image(path, model, class_names=None):
    #predict the class of an image
    input_tensor = preprocess_image(path)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    label = pred.item()
    if class_names:
        return class_names[label]
    return label

# Load pretrained ResNet18
resnet = models.resnet18()
print("Model loaded.")

# Replace final layer for CIFAR-100 (100 classes)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 100)
print("Final layer replaced for CIFAR-100.")

# Data transforms
transform = transforms.Compose([
    transforms.Resize(224),   # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Data loaded for training and testing.")

#Using cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)
print("Model moved to CUDA device.")

#Loss criterion and optimizer declaration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
print("Optimizer and loss function declared.")

#Training the model
print("Training started.")
train_model(resnet, criterion, optimizer, train_loader, device)
print("Training completed.")

#Evaluation after training
print("Evaluation started.")
resnet.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Evaluation completed.")
print("Test Accuracy:", 100 * correct / total)

#Transform the local database to resnet format
local_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

local_dataset = ImageFolder(root="data/pytorchTraining", transform=local_transform)
train_loader = DataLoader(local_dataset, batch_size=32, shuffle=True)
print("Local dataset loaded.")
print("Detected local dataset classes:", local_dataset.classes)

# Update final layer for local dataset classes (3)
num_classes = len(local_dataset.classes)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)
print("Final layer updated to local dataset classes.")

#Retrain model
print("Retraining model.")
train_model(resnet, criterion, optimizer, train_loader, device)
print("Retraining completed.")

print("Retraining evaluation started.")
resnet.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Retraining evaluation completed.")
print("Test Accuracy:", 100 * correct / total)

#Save model
torch.save(resnet.state_dict(), "resnet18_cifar100_potentiated.pth")
print("Model saved.")

#Test the model
print("Testing model.")
class_names = ["0","1","2"]
results = predict_image("data/pytorchTraining/1/010.jpeg", resnet, class_names)
print(results)