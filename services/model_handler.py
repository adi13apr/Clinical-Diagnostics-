import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# 1. Redefining CLAHE for Inference
class CLAHETransform:
    def __init__(self, clip_limit=1.5, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2: 
            img_np = self.clahe.apply(img_np)
        else:
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            img_lab[:,:,0] = self.clahe.apply(img_lab[:,:,0])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)

# 2. Redefining the Architecture (Must match your Student model exactly)
class StudentEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None) # Weights loaded via .pth
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        features = self.model.features(x)
        avg_features = self.model.avgpool(features)
        avg_features = torch.flatten(avg_features, 1)
        logits = self.model.classifier(avg_features)
        return logits, features

class ModelService:
    def __init__(self, model_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_weights_path)
        self.classes = {0: 'Normal', 1: 'Pneumonia', 2: 'Tuberculosis'}
        
        # 3. Matching your val_test_transforms exactly
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            CLAHETransform(clip_limit=1.5), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @st.cache_resource
    def _load_model(_self, path):
        model = StudentEfficientNet(num_classes=3)
        # Loading state_dict as saved in your Kaggle notebook
        model.load_state_dict(torch.load(path, map_location=_self.device))
        model.to(_self.device)
        model.eval()
        return model

    def predict(self, pil_image):
        # Ensure image is RGB (as required by Swin/EfficientNet teachers)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, prediction = torch.max(probabilities, dim=0)
            prediction = prediction.item()
            confidence = confidence.item()
            
        return {
            'diagnosis': self.classes[prediction],
            'confidence': confidence,
            'probabilities': {
                self.classes[i]: probabilities[i].item() 
                for i in range(len(self.classes))
            }
        }
    
    def generate_heatmap(self, pil_image, prediction_result=None):
        """Generate Grad-CAM heatmap for the image
        
        Args:
            pil_image: PIL Image to generate heatmap for
            prediction_result: Optional pre-computed prediction dict {diagnosis, confidence, probabilities}
                              If not provided, will compute prediction
        """
        from services.heatmap_handler import HeatmapGenerator
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Store original image for heatmap overlay
        original_image_np = np.array(pil_image)
        
        # Get prediction (or use provided one)
        if prediction_result is None:
            result = self.predict(pil_image)
        else:
            result = prediction_result
        
        # Map diagnosis string back to class index
        diagnosis = result['diagnosis']
        prediction_class = None
        for idx, class_name in self.classes.items():
            if class_name == diagnosis:
                prediction_class = idx
                break
        
        if prediction_class is None:
            raise ValueError(f"Unknown diagnosis: {diagnosis}")
        
        # Prepare tensor
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Generate heatmap
        heatmap_gen = HeatmapGenerator(self.model, self.device)
        heatmap_image = heatmap_gen.generate_heatmap_figure(
            img_tensor, 
            prediction_class, 
            original_image_np,
            diagnosis
        )
        heatmap_gen.cleanup()
        
        return heatmap_image
        
        return heatmap_image