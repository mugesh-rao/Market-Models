import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Function to scan and process an image using OpenCV
def scan_image(image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Unable to load the image. Please check the file path.")
            return None

        # Preprocess the image (resize, normalize, etc.)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocessed_image = preprocess(image).unsqueeze(0)  # Add batch dimension

        return preprocessed_image

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Define a simple model for demonstration
class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.fc = nn.Linear(224 * 224 * 3, 2)  # Example: Binary classification

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x

# Function to predict disease using the model
def predict_disease(model, image):
    try:
        # Load the trained model weights (replace with your model)
        model.load_state_dict(torch.load('plant_disease_model.pth'))
        model.eval()

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Input: Specify the path to the image you want to scan
    image_path = input("Enter the path to the image: ")

    # Scan and preprocess the image
    preprocessed_image = scan_image(image_path)

    if preprocessed_image is not None:
        # Create and load the model for disease prediction
        model = PlantDiseaseModel()

        # Predict the disease using the model
        prediction = predict_disease(model, preprocessed_image)

        if prediction is not None:
            if prediction == 0:
                print("No disease detected.")
            else:
                print("Disease detected.")
