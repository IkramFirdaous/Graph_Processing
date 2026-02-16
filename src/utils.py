"""
Utility Functions for Image Processing Pipeline
"""

import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt


def load_image_from_data(img_data) -> Optional[Image.Image]:
    """
    Load PIL Image from various data formats

    Args:
        img_data: Image data (bytes, dict, or PIL Image)

    Returns:
        PIL Image or None if loading fails
    """
    try:
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            return Image.open(io.BytesIO(img_data['bytes']))
        elif hasattr(img_data, 'convert'):  # Already a PIL Image
            return img_data
        else:
            print(f"Unknown image format: {type(img_data)}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def pil_to_numpy(img: Image.Image, grayscale: bool = False) -> np.ndarray:
    """
    Convert PIL Image to numpy array

    Args:
        img: PIL Image
        grayscale: Convert to grayscale if True

    Returns:
        Numpy array (H, W, C) or (H, W) if grayscale
    """
    if grayscale:
        img = img.convert('L')
        return np.array(img)
    else:
        img = img.convert('RGB')
        return np.array(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image

    Args:
        arr: Numpy array (H, W) or (H, W, C)

    Returns:
        PIL Image
    """
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode='L')
    elif arr.ndim == 3:
        return Image.fromarray(arr.astype(np.uint8), mode='RGB')
    else:
        raise ValueError(f"Invalid array shape: {arr.shape}")


def visualize_images(images: list, titles: list = None,
                     figsize: Tuple[int, int] = (15, 5),
                     save_path: Optional[str] = None):
    """
    Visualize multiple images in a row

    Args:
        images: List of images (PIL or numpy)
        titles: List of titles for each image
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for idx, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, Image.Image):
            ax.imshow(img)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

        if titles and idx < len(titles):
            ax.set_title(titles[idx])

        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    plt.show()


def get_image_stats(img: Union[Image.Image, np.ndarray]) -> dict:
    """
    Get basic statistics about an image

    Args:
        img: PIL Image or numpy array

    Returns:
        Dictionary with image statistics
    """
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = img

    stats = {
        'shape': arr.shape,
        'dtype': arr.dtype,
        'min': arr.min(),
        'max': arr.max(),
        'mean': arr.mean(),
        'std': arr.std(),
    }

    if isinstance(img, Image.Image):
        stats['pil_mode'] = img.mode
        stats['pil_size'] = img.size

    return stats


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_image(img: np.ndarray,
                   min_val: float = 0.0,
                   max_val: float = 255.0) -> np.ndarray:
    """
    Normalize image to [min_val, max_val] range

    Args:
        img: Input image array
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Normalized image
    """
    img_min = img.min()
    img_max = img.max()

    if img_max - img_min == 0:
        return np.zeros_like(img)

    normalized = (img - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val

    return normalized.astype(img.dtype)


def resize_image(img: Union[Image.Image, np.ndarray],
                 target_size: Tuple[int, int],
                 keep_aspect: bool = True) -> Union[Image.Image, np.ndarray]:
    """
    Resize image to target size

    Args:
        img: Input image
        target_size: (width, height)
        keep_aspect: Keep aspect ratio if True

    Returns:
        Resized image (same type as input)
    """
    is_numpy = isinstance(img, np.ndarray)

    if is_numpy:
        pil_img = numpy_to_pil(img)
    else:
        pil_img = img

    if keep_aspect:
        pil_img.thumbnail(target_size, Image.Resampling.LANCZOS)
    else:
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

    if is_numpy:
        return pil_to_numpy(pil_img, grayscale=(img.ndim == 2))
    else:
        return pil_img
