import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

class GradCAM:
    """Generate Grad-CAM heatmaps for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks on target layer
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.append(forward_handle)
        self.hook_handles.append(backward_handle)
    
    def generate_cam(self, input_tensor, target_class):
        """Generate CAM for target class"""
        # Forward pass
        self.model.eval()
        logits, _ = self.model(input_tensor)
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        
        # Backward pass
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().numpy()[0]  # [C, H, W]
        
        # Weight activations by gradients
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        cam = np.zeros(activations.shape[1:])  # [H, W]
        
        for i in range(len(weights)):
            cam += weights[i] * activations[i]
        
        # ReLU to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for handle in self.hook_handles:
            handle.remove()

class HeatmapGenerator:
    """Generate and visualize heatmaps from model predictions"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Use the avgpool layer as target for Grad-CAM
        self.grad_cam = GradCAM(model, model.model.avgpool)
    
    def generate_heatmap(self, image_tensor, prediction_class, original_image_np):
        """
        Generate heatmap for the image
        
        Args:
            image_tensor: Preprocessed image tensor [1, 3, 224, 224]
            prediction_class: Class index (0, 1, or 2)
            original_image_np: Original image as numpy array
        
        Returns:
            Heatmap overlay PIL image
        """
        # Generate CAM
        cam = self.grad_cam.generate_cam(image_tensor, prediction_class)
        
        # Upsample CAM to original image size
        cam_resized = cv2.resize(cam, (original_image_np.shape[1], original_image_np.shape[0]))
        
        # Normalize to 0-255
        cam_resized = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        # Convert to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        if len(original_image_np.shape) == 2:
            # Grayscale image - convert to RGB
            original_rgb = cv2.cvtColor(original_image_np, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_image_np
        
        # Overlay (60% original, 40% heatmap)
        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_rgb, 0.4, 0)
        
        return Image.fromarray(overlay.astype('uint8'))
    
    def generate_heatmap_figure(self, image_tensor, prediction_class, original_image_np, diagnosis):
        """
        Generate matplotlib figure with original and heatmap
        
        Returns:
            PIL Image of the figure
        """
        # Generate CAM
        cam = self.grad_cam.generate_cam(image_tensor, prediction_class)
        
        # Upsample CAM
        cam_resized = cv2.resize(cam, (original_image_np.shape[1], original_image_np.shape[0]))
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Diagnosis: {diagnosis} - Interpretation Map', fontsize=16, fontweight='bold')
        
        # Original image
        if len(original_image_np.shape) == 2:
            axes[0].imshow(original_image_np, cmap='gray')
        else:
            axes[0].imshow(original_image_np)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(original_image_np, cmap='gray')
        axes[1].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[1].set_title('Attention Regions')
        axes[1].axis('off')
        
        # Colorbar for CAM
        axes[2].imshow(cam_resized, cmap='jet')
        axes[2].set_title('Intensity Map\n(Red = High Influence)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        buf.close()
        plt.close(fig)
        
        return img
    
    def cleanup(self):
        """Clean up hooks"""
        self.grad_cam.remove_hooks()
