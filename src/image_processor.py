import cv2
import os
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    denoise: bool = True
) -> np.ndarray:
    """
    Preprocess an image for feature matching.
    
    Args:
        image: Input image
        target_size: Target size for resizing
        denoise: Whether to apply denoising
        
    Returns:
        Preprocessed image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        if denoise:
            image = cv2.fastNlMeansDenoising(image)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # Normalize
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def find_coin_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the contour of the coin in the image.
    
    Args:
        image: Input image
        
    Returns:
        Contour of the coin or None if not found
    """
    try:
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Find the largest contour (assuming it's the coin)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        return mask
        
    except Exception as e:
        logger.error(f"Error finding coin contour: {e}")
        return None

def split_coin_image(
    image_path: str,
    output_dir: str,
    save_intermediate: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Split a coin image into front and back sides, removing the scale bar.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images
        save_intermediate: Whether to save intermediate processing steps
        
    Returns:
        Tuple of (front_image, back_image) or (None, None) if processing fails
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load {image_path}")
            return None, None
            
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Remove scale bar (last ~70 pixels)
        img = img[:, :-70]
        
        # Calculate split point (middle of the remaining image)
        new_width = width - 70
        split_point = new_width // 2
        
        # Split into front and back
        front = img[:, :split_point]
        back = img[:, split_point:]
        
        # Verify split images are valid
        if front.size == 0 or back.size == 0:
            logger.error(f"Invalid split for {image_path}: front size {front.size}, back size {back.size}")
            return None, None
            
        # Save results
        base_name = Path(image_path).stem
        
        if save_intermediate:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), img)
        
        # Save front and back images
        front_path = os.path.join(output_dir, f"{base_name}_front.jpg")
        back_path = os.path.join(output_dir, f"{base_name}_back.jpg")
        
        if not cv2.imwrite(front_path, front):
            logger.error(f"Failed to save front image: {front_path}")
            return None, None
            
        if not cv2.imwrite(back_path, back):
            logger.error(f"Failed to save back image: {back_path}")
            return None, None
        
        logger.info(f"Successfully split {image_path} into front ({front.shape}) and back ({back.shape})")
        return front, back
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None, None

def preprocess_directory(
    input_dir: str,
    output_dir: str,
    save_intermediate: bool = False,
    is_api_images: bool = False
) -> Dict[str, np.ndarray]:
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Directory to save processed images
        save_intermediate: Whether to save intermediate processing steps
        is_api_images: Whether the images are from Harvard Art Museums API
        
    Returns:
        Dictionary mapping filenames to preprocessed images
    """
    images = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_dir, filename)
            
            if is_api_images:
                # For API images, split into front and back
                front, back = split_coin_image(path, str(output_dir), save_intermediate)
                
                if front is not None:
                    front_processed = preprocess_image(front)
                    if front_processed is not None:
                        front_name = f"{Path(filename).stem}_front.jpg"
                        front_path = output_dir / front_name
                        cv2.imwrite(str(front_path), front_processed)
                        images[Path(filename).stem + "_front"] = front_processed
                    
                if back is not None:
                    back_processed = preprocess_image(back)
                    if back_processed is not None:
                        back_name = f"{Path(filename).stem}_back.jpg"
                        back_path = output_dir / back_name
                        cv2.imwrite(str(back_path), back_processed)
                        images[Path(filename).stem + "_back"] = back_processed
            else:
                # For missing images, just preprocess the whole image
                img = cv2.imread(path)
                if img is not None:
                    processed = preprocess_image(img)
                    if processed is not None:
                        processed_name = f"{Path(filename).stem}.jpg"
                        processed_path = output_dir / processed_name
                        cv2.imwrite(str(processed_path), processed)
                        images[Path(filename).stem] = processed
    
    return images 