import torch
import torch.nn.functional as F
from torch import nn

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    This class generates heatmaps highlighting important regions in the input image
    for a given target class.
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.

        Args:
            model (nn.Module): The trained model.
            target_layer (nn.Module): The target convolutional layer to hook.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_image (torch.Tensor): Input image tensor.
            target_class (int, optional): Target class index. If None, uses the predicted class.

        Returns:
            torch.Tensor: Heatmap tensor.
        """
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        # Compute gradients and activations
        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Pool gradients
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weight activations by pooled gradients
        weighted_activations = activations * pooled_gradients
        heatmap = torch.sum(weighted_activations, dim=1, keepdim=True)

        # Apply ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap.squeeze()