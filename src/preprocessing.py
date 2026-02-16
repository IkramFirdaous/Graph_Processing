"""
Module 5.1: Preprocessing - Image Formation / Traitement de l'image

Concepts from CV course:
- Image formation
- Spatial filtering
- Noise reduction
- Contrast enhancement

This module normalizes images for downstream processing
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, Dict

# Handle both package and standalone imports
try:
    from .utils import pil_to_numpy, numpy_to_pil
except ImportError:
    from utils import pil_to_numpy, numpy_to_pil


class ImagePreprocessor:
    """
    Handles image preprocessing for graph analysis

    Pipeline:
    1. Resize to standard dimensions
    2. Convert color space (RGB/Grayscale)
    3. Denoise
    4. Enhance contrast
    5. Optional: Binarization
    """

    def __init__(self,
                 target_size: Optional[Tuple[int, int]] = (800, 800),
                 keep_aspect: bool = True):
        """
        Initialize preprocessor

        Args:
            target_size: (width, height) for resizing
            keep_aspect: Preserve aspect ratio if True
        """
        self.target_size = target_size
        self.keep_aspect = keep_aspect

    def resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size

        Args:
            img: Input image (H, W, C) or (H, W)

        Returns:
            Resized image
        """
        if self.target_size is None:
            return img

        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        if self.keep_aspect:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = target_w, target_h

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return resized

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale if needed

        Args:
            img: Input image

        Returns:
            Grayscale image
        """
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def denoise(self, img: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Apply denoising filter

        Methods:
        - 'bilateral': Edge-preserving bilateral filter
        - 'gaussian': Gaussian blur
        - 'median': Median filter

        Args:
            img: Input image
            method: Denoising method

        Returns:
            Denoised image
        """
        if method == 'bilateral':
            # Bilateral filter preserves edges
            if img.ndim == 3:
                return cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
            else:
                return cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

        elif method == 'gaussian':
            return cv2.GaussianBlur(img, (5, 5), 0)

        elif method == 'median':
            return cv2.medianBlur(img, 5)

        else:
            return img

    def enhance_contrast(self, img: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast

        Methods:
        - 'clahe': Contrast Limited Adaptive Histogram Equalization
        - 'histogram': Standard histogram equalization
        - 'normalize': Min-max normalization

        Args:
            img: Input image (grayscale)
            method: Enhancement method

        Returns:
            Enhanced image
        """
        if img.ndim == 3:
            # Convert to grayscale first
            img = self.to_grayscale(img)

        if method == 'clahe':
            # CLAHE is good for local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)

        elif method == 'histogram':
            return cv2.equalizeHist(img)

        elif method == 'normalize':
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        else:
            return img

    def binarize(self, img: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Binarize image (black/white)

        Methods:
        - 'otsu': Otsu's automatic thresholding
        - 'adaptive': Adaptive thresholding
        - 'fixed': Fixed threshold at 127

        Args:
            img: Input image (grayscale)
            method: Binarization method

        Returns:
            Binary image (0 or 255)
        """
        if img.ndim == 3:
            img = self.to_grayscale(img)

        if method == 'otsu':
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary

        elif method == 'fixed':
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return binary

        else:
            return img

    def preprocess(self,
                   img: np.ndarray,
                   grayscale: bool = False,
                   denoise_method: Optional[str] = 'bilateral',
                   enhance_contrast_method: Optional[str] = None,
                   binarize_method: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline

        Args:
            img: Input image (RGB numpy array)
            grayscale: Convert to grayscale
            denoise_method: Denoising method or None
            enhance_contrast_method: Contrast enhancement or None
            binarize_method: Binarization method or None

        Returns:
            Dictionary with:
            - 'original': Original input
            - 'resized': Resized image
            - 'processed': Final processed image
            - intermediate steps
        """
        results = {'original': img.copy()}

        # Step 1: Resize
        resized = self.resize(img)
        results['resized'] = resized

        current = resized.copy()

        # Step 2: Grayscale conversion
        if grayscale:
            current = self.to_grayscale(current)
            results['grayscale'] = current

        # Step 3: Denoising
        if denoise_method:
            current = self.denoise(current, method=denoise_method)
            results['denoised'] = current

        # Step 4: Contrast enhancement
        if enhance_contrast_method:
            current = self.enhance_contrast(current, method=enhance_contrast_method)
            results['enhanced'] = current

        # Step 5: Binarization
        if binarize_method:
            current = self.binarize(current, method=binarize_method)
            results['binary'] = current

        results['processed'] = current

        return results

    def get_statistics(self, img: np.ndarray) -> Dict[str, float]:
        """
        Calculate image statistics

        Args:
            img: Input image

        Returns:
            Dictionary with statistics
        """
        stats = {
            'mean': np.mean(img),
            'std': np.std(img),
            'min': np.min(img),
            'max': np.max(img),
            'median': np.median(img),
        }

        # Estimate noise level (using Laplacian variance)
        if img.ndim == 2:
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            stats['noise_estimate'] = laplacian.var()

        # Estimate sharpness (using gradient magnitude)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        stats['sharpness'] = np.mean(gradient_magnitude)

        return stats


def preprocess_image(img: np.ndarray,
                    target_size: Tuple[int, int] = (800, 800),
                    grayscale: bool = True,
                    denoise: bool = True,
                    enhance_contrast: bool = True) -> np.ndarray:
    """
    Convenience function for basic preprocessing

    Args:
        img: Input image (RGB)
        target_size: Target size
        grayscale: Convert to grayscale
        denoise: Apply denoising
        enhance_contrast: Apply contrast enhancement

    Returns:
        Preprocessed image
    """
    preprocessor = ImagePreprocessor(target_size=target_size)

    results = preprocessor.preprocess(
        img,
        grayscale=grayscale,
        denoise_method='bilateral' if denoise else None,
        enhance_contrast_method='clahe' if enhance_contrast else None
    )

    return results['processed']
