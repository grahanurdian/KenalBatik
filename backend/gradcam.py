import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

class BatikGradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks on the last layer of the features block
        # MobileNetV2 features end with a Conv2d layer
        target_layer = self.model.features[-1]
        
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple, we want the first element
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap and overlays it on the input image.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (1, 3, 224, 224).
            class_idx (int, optional): The class index to visualize. If None, uses the predicted class.
            
        Returns:
            PIL.Image: The final image with the heatmap overlay.
        """
        # a. Run a forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # b. Zero gradients, then run backward pass on the class_idx score
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # c. Calculate the weights (global average pooling of gradients)
        gradients = self.gradients
        # Global average pooling over spatial dimensions (H, W)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # d. Multiplies weights by feature maps to get the CAM
        # feature_maps shape: (1, C, H, W)
        feature_maps = self.feature_maps
        
        # We can do this efficiently with broadcasting
        # weights: (C,), feature_maps: (1, C, H, W)
        # We want to weight each channel of the feature map
        for i in range(feature_maps.shape[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        
        # e. Applies ReLU, normalizes to 0-1, and resizes it to 224x224
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        heatmap = heatmap.detach().cpu().numpy()
        
        # Resize to 224x224
        if cv2 is not None:
            heatmap = cv2.resize(heatmap, (224, 224))
            # f. Converts the heatmap to a Jet colormap (Blue-Red)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # Convert BGR to RGB
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        else:
            # Fallback using PIL and matplotlib
            heatmap_img = Image.fromarray(heatmap)
            heatmap_img = heatmap_img.resize((224, 224), resample=Image.BILINEAR)
            heatmap_resized = np.array(heatmap_img)
            
            # Apply Jet colormap
            colormap = plt.get_cmap('jet')
            heatmap_colored = colormap(heatmap_resized) # Returns RGBA 0-1
            heatmap = np.uint8(255 * heatmap_colored[:, :, :3]) # Convert to RGB 0-255

        # f. Overlays it on the original image
        # Reconstruct original image from input_tensor for visualization
        # Assuming input_tensor is (1, 3, 224, 224)
        img_tensor = input_tensor.squeeze().detach().cpu()
        
        # Denormalize for visualization (simple min-max scaling to 0-1)
        img_tensor = img_tensor - img_tensor.min()
        if img_tensor.max() != 0:
            img_tensor = img_tensor / img_tensor.max()
            
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.uint8(255 * img_np)
        
        # Overlay: 0.4 * heatmap + 0.6 * original
        superimposed_img = heatmap * 0.4 + img_np * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # g. Returns the final image as a PIL Image object
        return Image.fromarray(superimposed_img)
