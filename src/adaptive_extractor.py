"""
Adaptive Feature Extractor - Adapte l'extraction selon le type d'image

Utilise des stratégies différentes selon le type de diagramme détecté
"""

import numpy as np
import cv2
from typing import Dict, Any
from dataclasses import dataclass

try:
    from .image_classifier import classify_image_type, DiagramType
    from .detection import GraphPrimitiveDetector
except ImportError:
    from image_classifier import classify_image_type, DiagramType
    from detection import GraphPrimitiveDetector


@dataclass
class AdaptiveFeatures:
    """Features extraites de manière adaptative"""
    # Type classification
    diagram_type: str
    type_confidence: float

    # Universal features (tous types)
    visual_complexity: float
    color_entropy: float
    text_density: float
    spatial_layout: str

    # Type-specific features
    specific_features: Dict[str, Any]

    # Enriched description
    description_enrichment: str


class AdaptiveFeatureExtractor:
    """
    Extracteur de features adaptatif

    Méthode:
    1. Classifie le type d'image
    2. Sélectionne la stratégie d'extraction appropriée
    3. Extrait features spécifiques au type
    4. Génère description enrichie
    """

    def __init__(self):
        self.detector = GraphPrimitiveDetector()


    def extract(self, img: np.ndarray, img_original: np.ndarray = None) -> AdaptiveFeatures:
        """
        Extraction adaptative de features

        Args:
            img: Image preprocessed (grayscale)
            img_original: Image originale (couleur) pour analyse couleur

        Returns:
            AdaptiveFeatures
        """
        # Step 1: Classify image type
        type_info = classify_image_type(img)

        # Step 2: Extract universal features
        visual_complexity = self._compute_visual_complexity(img)
        color_entropy = self._compute_color_entropy(img_original if img_original is not None else img)
        text_density = type_info.text_density
        layout = self._determine_layout(type_info)

        # Step 3: Extract type-specific features
        specific_features = self._extract_type_specific(img, type_info)

        # Step 4: Generate enriched description
        enrichment = self._generate_enrichment(type_info, specific_features)

        return AdaptiveFeatures(
            diagram_type=type_info.diagram_type.value,
            type_confidence=type_info.confidence,
            visual_complexity=visual_complexity,
            color_entropy=color_entropy,
            text_density=text_density,
            spatial_layout=layout,
            specific_features=specific_features,
            description_enrichment=enrichment
        )


    def _compute_visual_complexity(self, img: np.ndarray) -> float:
        """Complexité visuelle (0-1) basée sur entropie et contours"""
        # Edge density
        edges = cv2.Canny(img, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Intensity variance
        variance = np.var(img) / (255 ** 2)

        # Combine
        complexity = 0.6 * edge_ratio * 10 + 0.4 * variance
        return min(1.0, complexity)


    def _compute_color_entropy(self, img: np.ndarray) -> float:
        """Entropie de couleur (diversité)"""
        if len(img.shape) == 2:
            # Grayscale
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        else:
            # RGB - use all channels
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
            hist = hist_r + hist_g + hist_b

        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy / 8.0  # Normalize


    def _determine_layout(self, type_info) -> str:
        """Détermine le layout spatial"""
        if type_info.has_radial_symmetry:
            return "radial"
        elif type_info.has_grid_layout:
            return "grid"
        elif type_info.has_hierarchical_structure:
            return "hierarchical"
        elif type_info.num_lines > type_info.num_rectangles:
            return "networked"
        else:
            return "freeform"


    def _extract_type_specific(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Extrait features spécifiques au type d'image"""

        dtype = type_info.diagram_type

        if dtype == DiagramType.NETWORK_GRAPH:
            return self._extract_network_graph_features(img, type_info)

        elif dtype == DiagramType.PIE_CHART:
            return self._extract_pie_chart_features(img, type_info)

        elif dtype == DiagramType.FLOWCHART:
            return self._extract_flowchart_features(img, type_info)

        elif dtype == DiagramType.INFOGRAPHIC:
            return self._extract_infographic_features(img, type_info)

        elif dtype == DiagramType.BAR_CHART:
            return self._extract_bar_chart_features(img, type_info)

        elif dtype == DiagramType.TREE_DIAGRAM:
            return self._extract_tree_features(img, type_info)

        else:
            # Default: generic extraction
            return self._extract_generic_features(img, type_info)


    def _extract_network_graph_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features pour graphes de réseau"""
        return {
            "num_nodes": type_info.num_circles + type_info.num_rectangles,
            "num_edges": type_info.num_lines,
            "graph_density": self._compute_graph_density(
                type_info.num_circles + type_info.num_rectangles,
                type_info.num_lines
            ),
            "avg_node_degree": self._estimate_avg_degree(
                type_info.num_circles + type_info.num_rectangles,
                type_info.num_lines
            ),
            "is_connected": type_info.num_lines >= type_info.num_circles - 1
        }


    def _extract_pie_chart_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features pour diagrammes circulaires"""
        return {
            "num_segments": self._estimate_pie_segments(img),
            "has_labels": type_info.text_density > 0.2,
            "is_donut": type_info.has_circles and type_info.num_circles > 1,
            "color_coded": type_info.color_diversity > 0.5
        }


    def _extract_flowchart_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features pour organigrammes"""
        return {
            "num_steps": type_info.num_rectangles,
            "num_connections": type_info.num_lines,
            "num_levels": self._estimate_hierarchy_depth(type_info),
            "branching_factor": type_info.num_lines / max(type_info.num_rectangles, 1),
            "has_decision_nodes": type_info.num_circles > 0
        }


    def _extract_infographic_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features pour infographies"""
        return {
            "num_sections": self._estimate_sections(img),
            "has_icons": type_info.num_circles > 2 or type_info.num_rectangles > 5,
            "text_to_visual_ratio": type_info.text_density / max(type_info.edge_density, 0.1),
            "layout_complexity": "high" if type_info.edge_density > 0.3 else "medium",
            "color_scheme": "diverse" if type_info.color_diversity > 0.6 else "simple"
        }


    def _extract_bar_chart_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features pour histogrammes"""
        return {
            "num_bars": type_info.num_rectangles,
            "orientation": self._detect_bar_orientation(img),
            "has_grid": type_info.has_grid_layout,
            "num_categories": type_info.num_rectangles
        }


    def _extract_tree_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features pour arbres hiérarchiques"""
        return {
            "num_nodes": type_info.num_rectangles + type_info.num_circles,
            "tree_depth": self._estimate_hierarchy_depth(type_info),
            "branching_factor": type_info.num_lines / max(type_info.num_rectangles, 1),
            "is_binary": type_info.num_lines <= 2 * type_info.num_rectangles
        }


    def _extract_generic_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features génériques pour types inconnus"""
        return {
            "num_shapes": type_info.num_circles + type_info.num_rectangles,
            "num_lines": type_info.num_lines,
            "has_structure": type_info.has_grid_layout or type_info.has_hierarchical_structure,
            "dominant_feature": self._get_dominant_feature(type_info)
        }


    # Helper methods

    def _compute_graph_density(self, num_nodes: int, num_edges: int) -> float:
        """Densité d'un graphe"""
        if num_nodes < 2:
            return 0.0
        max_edges = num_nodes * (num_nodes - 1) / 2
        return num_edges / max_edges if max_edges > 0 else 0.0


    def _estimate_avg_degree(self, num_nodes: int, num_edges: int) -> float:
        """Degré moyen dans un graphe"""
        if num_nodes == 0:
            return 0.0
        return (2 * num_edges) / num_nodes


    def _estimate_pie_segments(self, img: np.ndarray) -> int:
        """Estime le nombre de segments dans un pie chart"""
        # Detect lines radiating from center
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

        if lines is not None:
            # Count distinct angles
            angles = [line[0][1] for line in lines]
            unique_angles = len(set([int(a * 180 / np.pi) // 10 for a in angles]))
            return max(2, unique_angles)

        return 2  # Default minimum


    def _estimate_hierarchy_depth(self, type_info) -> int:
        """Estime la profondeur d'une hiérarchie"""
        # Simple heuristic based on number of elements
        total_elements = type_info.num_rectangles + type_info.num_circles

        if total_elements < 3:
            return 1
        elif total_elements < 6:
            return 2
        elif total_elements < 12:
            return 3
        else:
            return 4


    def _estimate_sections(self, img: np.ndarray) -> int:
        """Estime le nombre de sections dans une infographie"""
        # Detect horizontal dividing lines
        h, w = img.shape[:2]
        horizontal_profile = np.sum(img < 128, axis=1)

        # Find valleys (section separators)
        threshold = 0.1 * w
        sections = 1

        for i in range(10, h - 10):
            if horizontal_profile[i] > threshold:
                # Check if local maximum
                if (horizontal_profile[i] > horizontal_profile[i-5] and
                    horizontal_profile[i] > horizontal_profile[i+5]):
                    sections += 1

        return min(sections, 10)  # Cap at 10


    def _detect_bar_orientation(self, img: np.ndarray) -> str:
        """Détecte l'orientation des barres (vertical/horizontal)"""
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        if lines is None:
            return "unknown"

        vertical_lines = 0
        horizontal_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 10:
                vertical_lines += 1
            if abs(y1 - y2) < 10:
                horizontal_lines += 1

        return "vertical" if vertical_lines > horizontal_lines else "horizontal"


    def _get_dominant_feature(self, type_info) -> str:
        """Identifie la feature dominante"""
        if type_info.num_circles > type_info.num_rectangles and type_info.num_circles > type_info.num_lines:
            return "circles"
        elif type_info.num_rectangles > type_info.num_lines:
            return "rectangles"
        elif type_info.num_lines > 0:
            return "lines"
        else:
            return "complex"


    def _generate_enrichment(self, type_info, specific_features: Dict) -> str:
        """Génère une description enrichie textuelle"""

        dtype = type_info.diagram_type

        if dtype == DiagramType.NETWORK_GRAPH:
            return (f"Network graph with {specific_features['num_nodes']} nodes and "
                   f"{specific_features['num_edges']} edges, "
                   f"density {specific_features['graph_density']:.2f}")

        elif dtype == DiagramType.PIE_CHART:
            return (f"Pie chart with {specific_features['num_segments']} segments, "
                   f"{'labeled' if specific_features['has_labels'] else 'unlabeled'}")

        elif dtype == DiagramType.FLOWCHART:
            return (f"Flowchart with {specific_features['num_steps']} steps across "
                   f"{specific_features['num_levels']} levels")

        elif dtype == DiagramType.INFOGRAPHIC:
            return (f"Infographic with {specific_features['num_sections']} sections, "
                   f"{specific_features['layout_complexity']} complexity, "
                   f"{specific_features['color_scheme']} color scheme")

        elif dtype == DiagramType.BAR_CHART:
            return (f"Bar chart with {specific_features['num_bars']} bars, "
                   f"{specific_features['orientation']} orientation")

        elif dtype == DiagramType.TREE_DIAGRAM:
            return (f"Tree diagram with {specific_features['num_nodes']} nodes, "
                   f"depth {specific_features['tree_depth']}")

        else:
            return (f"Diagram with {specific_features.get('num_shapes', 0)} shapes, "
                   f"dominant feature: {specific_features.get('dominant_feature', 'unknown')}")


def extract_adaptive_features(img: np.ndarray, img_original: np.ndarray = None) -> AdaptiveFeatures:
    """
    Fonction utilitaire pour extraction adaptative

    Args:
        img: Image preprocessed (grayscale)
        img_original: Image originale (couleur)

    Returns:
        AdaptiveFeatures
    """
    extractor = AdaptiveFeatureExtractor()
    return extractor.extract(img, img_original)
