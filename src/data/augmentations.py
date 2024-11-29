import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    """
    Define transformations for data augmentation and preprocessing.

    Returns:
        A.Compose: Augmentation pipeline.
    """
    transforms = A.Compose([
        A.Resize(224, 224),  # Resize to 224x224
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize
        ToTensorV2()  # Convert to PyTorch Tensor
    ])
    return transforms

def preprocess_image(image_path, transforms):
    """
    Apply preprocessing transformations to an image.

    Args:
        image_path (str): Path to the image file.
        transforms (A.Compose): Augmentation pipeline.

    Returns:
        Tensor: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms(image=image)['image']
