"""
Module 5.4: Feature Extraction - Descripteurs et caractéristiques

Concepts from CV course:
- Feature descriptors
- Geometric features
- Structural features
- Spatial features
- Statistical descriptors

This module transforms images into interpretable feature vectors
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from scipy.stats import skew, kurtosis


class GraphFeatureExtractor:
    """
    Extracts interpretable features from graph images

    Categories:
    1. Geometric features (counts, sizes, densities)
    2. Structural features (connectivity, degree distribution)
    3. Spatial features (layout, positioning)
    4. Visual features (texture, intensity)
    """

    def __init__(self):
        pass

    def extract_geometric_features(self,
                                   nodes: List,
                                   edges: List,
                                   img_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Extract geometric features

        Args:
            nodes: List of detected nodes
            edges: List of detected edges
            img_shape: (height, width) of image

        Returns:
            Dictionary of geometric features
        """
        h, w = img_shape[:2]
        total_area = h * w

        features = {}

        # Node features
        features['num_nodes'] = len(nodes)

        if len(nodes) > 0:
            node_areas = [node.area for node in nodes]
            features['total_node_area'] = sum(node_areas)
            features['mean_node_area'] = np.mean(node_areas)
            features['std_node_area'] = np.std(node_areas)
            features['min_node_area'] = np.min(node_areas)
            features['max_node_area'] = np.max(node_areas)

            # Node density (nodes per unit area)
            features['node_density'] = len(nodes) / total_area
        else:
            features['total_node_area'] = 0
            features['mean_node_area'] = 0
            features['std_node_area'] = 0
            features['min_node_area'] = 0
            features['max_node_area'] = 0
            features['node_density'] = 0

        # Edge features
        features['num_edges'] = len(edges)

        if len(edges) > 0:
            edge_lengths = [edge.length for edge in edges]
            features['total_edge_length'] = sum(edge_lengths)
            features['mean_edge_length'] = np.mean(edge_lengths)
            features['std_edge_length'] = np.std(edge_lengths)
            features['min_edge_length'] = np.min(edge_lengths)
            features['max_edge_length'] = np.max(edge_lengths)

            # Edge density (total length per unit area)
            features['edge_density'] = sum(edge_lengths) / total_area
        else:
            features['total_edge_length'] = 0
            features['mean_edge_length'] = 0
            features['std_edge_length'] = 0
            features['min_edge_length'] = 0
            features['max_edge_length'] = 0
            features['edge_density'] = 0

        # Graph density
        if len(nodes) > 1:
            max_possible_edges = len(nodes) * (len(nodes) - 1) / 2
            features['graph_density'] = len(edges) / max_possible_edges if max_possible_edges > 0 else 0
        else:
            features['graph_density'] = 0

        return features

    def extract_structural_features(self,
                                    nodes: List,
                                    edges: List,
                                    components: List[Dict]) -> Dict[str, float]:
        """
        Extract structural features

        Args:
            nodes: Detected nodes
            edges: Detected edges
            components: Connected components

        Returns:
            Dictionary of structural features
        """
        features = {}

        # Component features
        features['num_components'] = len(components)

        if len(components) > 0:
            component_areas = [comp['area'] for comp in components]
            features['mean_component_area'] = np.mean(component_areas)
            features['std_component_area'] = np.std(component_areas)
            features['max_component_area'] = np.max(component_areas)
        else:
            features['mean_component_area'] = 0
            features['std_component_area'] = 0
            features['max_component_area'] = 0

        # Estimate degree distribution (approximate)
        if len(nodes) > 0 and len(edges) > 0:
            # Average degree = 2 * edges / nodes (for undirected graphs)
            features['avg_degree_estimate'] = (2 * len(edges)) / len(nodes)
        else:
            features['avg_degree_estimate'] = 0

        # Connectivity estimate
        if len(nodes) > 0:
            features['connectivity_ratio'] = len(components) / len(nodes)
        else:
            features['connectivity_ratio'] = 0

        # Shape diversity (entropy of shape types)
        if len(nodes) > 0:
            shape_counts = {}
            for node in nodes:
                shape_counts[node.shape] = shape_counts.get(node.shape, 0) + 1

            shape_probs = np.array(list(shape_counts.values())) / len(nodes)
            features['shape_entropy'] = -np.sum(shape_probs * np.log2(shape_probs + 1e-10))
            features['num_shape_types'] = len(shape_counts)
        else:
            features['shape_entropy'] = 0
            features['num_shape_types'] = 0

        return features

    def extract_spatial_features(self, nodes: List, img_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Extract spatial layout features

        Args:
            nodes: Detected nodes
            img_shape: Image shape (height, width)

        Returns:
            Dictionary of spatial features
        """
        features = {}
        h, w = img_shape[:2]

        if len(nodes) == 0:
            return {
                'spatial_spread_x': 0,
                'spatial_spread_y': 0,
                'center_of_mass_x': 0,
                'center_of_mass_y': 0,
                'alignment_score': 0,
            }

        # Extract node centers
        centers = np.array([node.center for node in nodes])

        # Spatial spread
        features['spatial_spread_x'] = np.std(centers[:, 0])
        features['spatial_spread_y'] = np.std(centers[:, 1])

        # Center of mass
        features['center_of_mass_x'] = np.mean(centers[:, 0]) / w  # Normalized
        features['center_of_mass_y'] = np.mean(centers[:, 1]) / h

        # Alignment score (how aligned are nodes on x or y axis)
        # High alignment means nodes are on similar x or y coordinates
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]

        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)

        # Normalized alignment
        features['horizontal_alignment'] = 1 - (y_variance / (h * h / 12)) if h > 0 else 0
        features['vertical_alignment'] = 1 - (x_variance / (w * w / 12)) if w > 0 else 0

        # Overall layout compactness
        if len(centers) > 1:
            # Calculate average pairwise distance
            from scipy.spatial.distance import pdist
            distances = pdist(centers)
            features['avg_node_distance'] = np.mean(distances)
            features['std_node_distance'] = np.std(distances)
        else:
            features['avg_node_distance'] = 0
            features['std_node_distance'] = 0

        return features

    def extract_visual_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract visual/texture features

        Args:
            img: Grayscale image

        Returns:
            Dictionary of visual features
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        features = {}

        # Intensity statistics
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['intensity_range'] = np.ptp(gray)  # Peak-to-peak

        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize

        features['intensity_skewness'] = float(skew(hist))
        features['intensity_kurtosis'] = float(kurtosis(hist))

        # Entropy (measure of complexity)
        hist_nonzero = hist[hist > 0]
        features['intensity_entropy'] = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        # Edge strength (using Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        features['mean_edge_strength'] = np.mean(gradient_magnitude)
        features['std_edge_strength'] = np.std(gradient_magnitude)

        # Texture complexity (using Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_complexity'] = np.var(laplacian)

        return features

    def extract_all_features(self,
                            img: np.ndarray,
                            nodes: List = None,
                            edges: List = None,
                            components: List[Dict] = None) -> Dict[str, float]:
        """
        Extract all features from an image

        Args:
            img: Input image
            nodes: Detected nodes (optional)
            edges: Detected edges (optional)
            components: Segmentation components (optional)

        Returns:
            Complete feature dictionary
        """
        all_features = {}

        # Visual features (always available)
        visual = self.extract_visual_features(img)
        all_features.update(visual)

        # Geometric features (if nodes/edges available)
        if nodes is not None and edges is not None:
            geometric = self.extract_geometric_features(nodes, edges, img.shape)
            all_features.update(geometric)

            # Spatial features (if nodes available)
            spatial = self.extract_spatial_features(nodes, img.shape)
            all_features.update(spatial)

        # Structural features (if all available)
        if nodes is not None and edges is not None and components is not None:
            structural = self.extract_structural_features(nodes, edges, components)
            all_features.update(structural)

        return all_features


def extract_features(img: np.ndarray,
                    nodes: List = None,
                    edges: List = None,
                    components: List[Dict] = None) -> Dict[str, float]:
    """
    Convenience function for feature extraction

    Args:
        img: Input image
        nodes: Detected nodes
        edges: Detected edges
        components: Segmentation components

    Returns:
        Feature dictionary
    """
    extractor = GraphFeatureExtractor()
    return extractor.extract_all_features(img, nodes, edges, components)
