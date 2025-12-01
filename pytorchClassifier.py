import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(path):
    #Preprocessing the image for use to predict
    # Using cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Option A: using PIL
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # add batch dimension
    tensor = tensor.to(device)
    return tensor

def preprocess_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transforms.ToTensor()(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

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

def predict_frame(frame, model, class_names=None):
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    label = pred.item()
    if class_names:
        return class_names[label]
    return label

def load_model(model_path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path))
    return model

