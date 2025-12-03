import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

class MedicalImagePreprocessor:
    """
    Data preprocessing pipeline for medical images.
    Implements augmentation and normalization as specified in the paper.
    """

    def __init__(self, input_size=224):
        """
        Initialize the preprocessing pipeline.

        Args:
            input_size (int): Size to which images will be resized (default: 224).
        """
        self.transform = A.Compose([
            # Augmentations
            A.Rotate(limit=90, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Convert to tensor
            ToTensorV2(),
        ])

    def __call__(self, image):
        """
        Apply preprocessing transformations to an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        transformed = self.transform(image=image)
        return transformed["image"]