import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple

def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from file path or return if already a numpy array."""
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        image = cv2.imread(path_str)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {path_str}")
        return image
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise ValueError("Invalid image input type. Expected file path or numpy array.")

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it is in BGR format."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def denoise(image: np.ndarray, method: str = "gaussian", kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """Apply noise reduction blur to the image."""
    if method == "gaussian":
        return cv2.GaussianBlur(image, kernel_size, 0)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size[0])
    elif method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return image

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def binarize(image: np.ndarray, method: str = "adaptive", block_size: int = 31, c: int = 11) -> np.ndarray:
    """Binarize the grayscale image using adaptive thresholding or Otsu's method."""
    gray = to_grayscale(image)
    if method == "adaptive":
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
    elif method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    return gray

def deskew(image: np.ndarray) -> np.ndarray:
    """Detect text angle using image contours and deskew the image."""
    gray = to_grayscale(image)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate image if angle is significant
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    return image

def preprocess(
    image_input: Union[str, Path, np.ndarray],
    denoise_flag: bool = True,
    clahe_flag: bool = True,
    binarize_flag: bool = True,
    deskew_flag: bool = False
) -> np.ndarray:
    """
    Master preprocessing function.
    Applies optional deskew, grayscale conversion, denoising, contrast enhancement, and thresholding.
    """
    img = load_image(image_input)
    
    if deskew_flag:
        img = deskew(img)
        
    gray = to_grayscale(img)
    
    if denoise_flag:
        gray = denoise(gray, method="gaussian", kernel_size=(3, 3))
        
    if clahe_flag:
        gray = enhance_contrast(gray)
        
    if binarize_flag:
        processed = binarize(gray, method="adaptive")
    else:
        processed = gray
        
    return processed
