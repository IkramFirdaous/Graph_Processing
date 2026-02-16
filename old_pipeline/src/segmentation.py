"""
Module 5.3: Graph Segmentation - Component Separation

Concepts from CV course:
- Image segmentation
- Connected components analysis
- Watershed algorithm
- Region-based segmentation
- Clustering

This module separates graph components and identifies sub-structures
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from scipy import ndimage


class GraphSegmentator:
    """
    Segments graphs into components and sub-graphs

    Methods:
    1. Connected components (basic segmentation)
    2. Watershed segmentation (advanced)
    3. Clustering-based segmentation
    """

    def __init__(self):
        pass

    def connected_components(self, binary_img: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Find connected components in binary image

        Args:
            binary_img: Binary image (0 or 255)

        Returns:
            (labeled_image, num_components)
        """
        # Ensure binary
        _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)

        # num_labels includes background (label 0), so actual components = num_labels - 1
        return labels, num_labels - 1

    def segment_by_threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Segment using adaptive thresholding

        Args:
            img: Grayscale image

        Returns:
            Binary segmentation
        """
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        return binary

    def watershed_segmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Watershed segmentation for separating touching objects

        Args:
            img: Grayscale image

        Returns:
            Labeled segmentation map
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        if img.ndim == 3:
            markers = cv2.watershed(img, markers)
        else:
            # Convert to color for watershed
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_color, markers)

        return markers

    def extract_component_features(self, labels: np.ndarray, num_components: int) -> List[Dict]:
        """
        Extract features for each connected component

        Args:
            labels: Labeled image from connected_components
            num_components: Number of components

        Returns:
            List of feature dictionaries
        """
        components = []

        for i in range(1, num_components + 1):
            # Create mask for this component
            mask = (labels == i).astype(np.uint8) * 255

            # Get bounding box
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Calculate features
            area = np.sum(mask > 0)
            bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            fill_ratio = area / bbox_area if bbox_area > 0 else 0

            # Centroid
            centroid_y, centroid_x = coords.mean(axis=0)

            component_info = {
                'id': i,
                'area': area,
                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                'centroid': (int(centroid_x), int(centroid_y)),
                'fill_ratio': fill_ratio,
                'aspect_ratio': (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 0
            }

            components.append(component_info)

        return components

    def segment_hierarchical(self, img: np.ndarray) -> Dict:
        """
        Hierarchical segmentation at multiple levels

        Args:
            img: Input grayscale image

        Returns:
            Dictionary with segmentation at different levels
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        results = {}

        # Level 1: Coarse segmentation
        binary_coarse = self.segment_by_threshold(gray)
        labels_coarse, num_coarse = self.connected_components(binary_coarse)
        results['coarse'] = {
            'labels': labels_coarse,
            'num_components': num_coarse,
            'components': self.extract_component_features(labels_coarse, num_coarse)
        }

        # Level 2: Fine segmentation with watershed
        labels_fine = self.watershed_segmentation(gray)
        num_fine = len(np.unique(labels_fine)) - 2  # Exclude background and boundary
        results['fine'] = {
            'labels': labels_fine,
            'num_components': num_fine
        }

        return results

    def visualize_segmentation(self, img: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Visualize segmentation with colored regions

        Args:
            img: Original image
            labels: Labeled segmentation

        Returns:
            Visualization image
        """
        # Create color map
        unique_labels = np.unique(labels)
        colors = np.random.randint(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)

        # Create colored segmentation
        h, w = labels.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for idx, label in enumerate(unique_labels):
            if label == 0 or label == -1:  # Background or boundary
                continue
            colored[labels == label] = colors[idx]

        # Overlay on original image
        if img.ndim == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()

        vis = cv2.addWeighted(img_color, 0.5, colored, 0.5, 0)

        return vis


def segment_graph(img: np.ndarray, method: str = 'connected_components') -> Dict:
    """
    Convenience function for graph segmentation

    Args:
        img: Input image
        method: 'connected_components', 'watershed', or 'hierarchical'

    Returns:
        Segmentation results
    """
    segmentator = GraphSegmentator()

    if method == 'connected_components':
        binary = segmentator.segment_by_threshold(img)
        labels, num_components = segmentator.connected_components(binary)
        components = segmentator.extract_component_features(labels, num_components)
        return {
            'labels': labels,
            'num_components': num_components,
            'components': components
        }

    elif method == 'watershed':
        labels = segmentator.watershed_segmentation(img)
        num_components = len(np.unique(labels)) - 2
        return {
            'labels': labels,
            'num_components': num_components
        }

    elif method == 'hierarchical':
        return segmentator.segment_hierarchical(img)

    else:
        raise ValueError(f"Unknown method: {method}")
