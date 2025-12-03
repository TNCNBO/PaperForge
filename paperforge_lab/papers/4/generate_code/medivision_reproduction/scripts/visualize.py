import torch
import matplotlib.pyplot as plt
import numpy as np
from medivision.grad_cam import GradCAM
from medivision.preprocessing import MedicalImagePreprocessor


def visualize_grad_cam(model, target_layer, image_path, target_class=None, save_path=None):
    """
    Visualize Grad-CAM heatmap overlayed on the input image.

    Args:
        model (torch.nn.Module): Trained model.
        target_layer (torch.nn.Module): Target convolutional layer for Grad-CAM.
        image_path (str): Path to the input image.
        target_class (int, optional): Target class for Grad-CAM. Defaults to None.
        save_path (str, optional): Path to save the visualization. Defaults to None.
    """
    # Load and preprocess the image
    preprocessor = MedicalImagePreprocessor()
    image = plt.imread(image_path)
    input_tensor = preprocessor(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor, target_class)

    # Convert heatmap to numpy
    heatmap = heatmap.squeeze().cpu().numpy()

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    # Resize heatmap to match image dimensions
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]

    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 1)

    # Display or save the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(superimposed_img)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    from medivision.model import MediVision

    # Load a trained model
    model = MediVision(input_channels=3, num_classes=10)
    model.load_state_dict(torch.load("results/models/best_model.pth"))
    model.eval()

    # Specify target layer (e.g., last convolutional layer)
    target_layer = model.cnn_feature_extractor.conv_layers[-1]

    # Visualize Grad-CAM
    visualize_grad_cam(
        model,
        target_layer,
        image_path="data/alzheimer/sample_image.png",
        target_class=0,
        save_path="results/plots/grad_cam_sample.png",
    )