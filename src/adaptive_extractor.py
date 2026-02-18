"""
Adaptive Feature Extractor - Adapte l'extraction selon le type d'image

Utilise des strategies differentes selon le type de diagramme detecte.
Integre classification heuristique + classificateur appris (Random Forest),
segmentation couleur (K-means LAB), et descripteurs visuels (HOG, LBP, ORB).
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from .image_classifier import classify_image_type, DiagramType
    from .detection import GraphPrimitiveDetector
    from .feature_descriptors import FeatureDescriptorExtractor, FeatureVector
    from .learned_classifier import LearnedClassifier
except ImportError:
    from image_classifier import classify_image_type, DiagramType
    from detection import GraphPrimitiveDetector
    from feature_descriptors import FeatureDescriptorExtractor, FeatureVector
    from learned_classifier import LearnedClassifier


@dataclass
class AdaptiveFeatures:
    """Features extraites de maniere adaptative"""
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

    # Color segmentation features (K-means LAB)
    color_features: Optional[Dict[str, Any]] = None

    # Feature descriptors (HOG, LBP, ORB)
    feature_vector: Optional[FeatureVector] = None

    # Learned classifier results
    learned_type: Optional[str] = None
    learned_confidence: Optional[float] = None


class AdaptiveFeatureExtractor:
    """
    Extracteur de features adaptatif

    Pipeline ameliore :
    1. Segmentation couleur K-means sur image RGB originale
    2. Classification heuristique (Hough, contours) sur grayscale
    3. Classification apprise (Random Forest) si modele disponible
    4. Extraction de descripteurs visuels (HOG, LBP, ORB)
    5. Extraction de features specifiques au type
    6. Generation de description enrichie
    """

    def __init__(self, use_learned_classifier: bool = True):
        self.detector = GraphPrimitiveDetector()
        self.feature_extractor = FeatureDescriptorExtractor()
        self.learned_classifier = None

        if use_learned_classifier:
            self.learned_classifier = LearnedClassifier()
            if not self.learned_classifier.load():
                self.learned_classifier = None


    def extract(self, img: np.ndarray, img_original: np.ndarray = None) -> AdaptiveFeatures:
        """
        Extraction adaptative de features

        Args:
            img: Image preprocessed (grayscale)
            img_original: Image originale (couleur) pour analyse couleur

        Returns:
            AdaptiveFeatures
        """
        # Step 1: Color segmentation on original RGB
        color_features = None
        if img_original is not None and img_original.ndim == 3:
            color_features = self._extract_color_features(img_original)

        # Step 2: Heuristic classification (always run for structural info)
        type_info = classify_image_type(img)

        # Step 3: Learned classification (if model available)
        learned_type = None
        learned_confidence = None
        if self.learned_classifier is not None:
            learned_type, learned_confidence = self.learned_classifier.predict(
                img, color_features=color_features
            )

        # Use learned type if confident, else heuristic
        if learned_type and learned_confidence and learned_confidence > 0.6:
            active_type_value = learned_type
            active_confidence = learned_confidence
        else:
            active_type_value = type_info.diagram_type.value
            active_confidence = type_info.confidence

        # Step 4: Extract universal features
        visual_complexity = self._compute_visual_complexity(img)
        color_entropy = self._compute_color_entropy(img_original if img_original is not None else img)
        text_density = type_info.text_density
        layout = self._determine_layout(type_info)

        # Step 5: Feature descriptors (HOG + LBP + ORB)
        geometric_features = {
            'num_circles': type_info.num_circles,
            'num_rectangles': type_info.num_rectangles,
            'num_lines': type_info.num_lines,
            'edge_density': type_info.edge_density,
            'text_density': text_density,
            'color_diversity': type_info.color_diversity,
        }
        feature_vector = self.feature_extractor.extract_all(
            img,
            color_features=color_features,
            geometric_features=geometric_features
        )

        # Step 6: Type-specific features
        specific_features = self._extract_type_specific(img, type_info)

        # Step 7: Enriched description
        enrichment = self._generate_enrichment(type_info, specific_features)

        return AdaptiveFeatures(
            diagram_type=active_type_value,
            type_confidence=active_confidence,
            visual_complexity=visual_complexity,
            color_entropy=color_entropy,
            text_density=text_density,
            spatial_layout=layout,
            specific_features=specific_features,
            description_enrichment=enrichment,
            color_features=color_features,
            feature_vector=feature_vector,
            learned_type=learned_type,
            learned_confidence=learned_confidence,
        )


    def _extract_color_features(self, img_rgb: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """
        Segmentation couleur par K-means en espace LAB

        Concepts CV (Theme 5 - Segmentation) :
        - Conversion RGB -> LAB pour uniformite perceptuelle
        - K-means clustering dans l'espace couleur
        - Analyse de la distribution des segments

        Args:
            img_rgb: Image RGB originale
            k: Nombre de clusters couleur

        Returns:
            Dict avec features couleur
        """
        # Conversion en espace LAB (perceptuellement uniforme)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # Reshape pour K-means : (N, 3) float32
        pixels = img_lab.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        compactness, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Tailles des segments
        labels_flat = labels.flatten()
        total_pixels = len(labels_flat)
        segment_sizes = []
        for i in range(k):
            count = int(np.sum(labels_flat == i))
            segment_sizes.append(count / total_pixels)

        segment_sizes.sort(reverse=True)

        # Nombre de couleurs "dominantes" (> 5% de l'image)
        num_dominant = sum(1 for s in segment_sizes if s > 0.05)

        # Balance couleur (std faible = equilibre)
        color_balance = float(1.0 - np.std(segment_sizes))

        return {
            'num_dominant_colors': num_dominant,
            'largest_segment_ratio': float(segment_sizes[0]),
            'color_balance': color_balance,
            'compactness': float(compactness),
            'segment_2_ratio': float(segment_sizes[1]) if len(segment_sizes) > 1 else 0.0,
        }


    def _compute_visual_complexity(self, img: np.ndarray) -> float:
        """Complexite visuelle (0-1) basee sur entropie et contours"""
        # Edge density
        edges = cv2.Canny(img, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Intensity variance
        variance = np.var(img) / (255 ** 2)

        # Combine
        complexity = 0.6 * edge_ratio * 10 + 0.4 * variance
        return min(1.0, complexity)


    def _compute_color_entropy(self, img: np.ndarray) -> float:
        """Entropie de couleur (diversite)"""
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
        """Determine le layout spatial"""
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
        """Extrait features specifiques au type d'image"""

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
        """Features pour graphes de reseau"""
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
        """Features pour arbres hierarchiques"""
        return {
            "num_nodes": type_info.num_rectangles + type_info.num_circles,
            "tree_depth": self._estimate_hierarchy_depth(type_info),
            "branching_factor": type_info.num_lines / max(type_info.num_rectangles, 1),
            "is_binary": type_info.num_lines <= 2 * type_info.num_rectangles
        }


    def _extract_generic_features(self, img: np.ndarray, type_info) -> Dict[str, Any]:
        """Features generiques pour types inconnus"""
        return {
            "num_shapes": type_info.num_circles + type_info.num_rectangles,
            "num_lines": type_info.num_lines,
            "has_structure": type_info.has_grid_layout or type_info.has_hierarchical_structure,
            "dominant_feature": self._get_dominant_feature(type_info)
        }


    # Helper methods

    def _compute_graph_density(self, num_nodes: int, num_edges: int) -> float:
        """Densite d'un graphe"""
        if num_nodes < 2:
            return 0.0
        max_edges = num_nodes * (num_nodes - 1) / 2
        return num_edges / max_edges if max_edges > 0 else 0.0


    def _estimate_avg_degree(self, num_nodes: int, num_edges: int) -> float:
        """Degre moyen dans un graphe"""
        if num_nodes == 0:
            return 0.0
        return (2 * num_edges) / num_nodes


    def _estimate_pie_segments(self, img: np.ndarray) -> int:
        """Estime le nombre de segments dans un pie chart"""
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

        if lines is not None:
            angles = [line[0][1] for line in lines]
            unique_angles = len(set([int(a * 180 / np.pi) // 10 for a in angles]))
            return max(2, unique_angles)

        return 2


    def _estimate_hierarchy_depth(self, type_info) -> int:
        """Estime la profondeur d'une hierarchie"""
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
        h, w = img.shape[:2]
        horizontal_profile = np.sum(img < 128, axis=1)

        threshold = 0.1 * w
        sections = 1

        for i in range(10, h - 10):
            if horizontal_profile[i] > threshold:
                if (horizontal_profile[i] > horizontal_profile[i-5] and
                    horizontal_profile[i] > horizontal_profile[i+5]):
                    sections += 1

        return min(sections, 10)


    def _detect_bar_orientation(self, img: np.ndarray) -> str:
        """Detecte l'orientation des barres (vertical/horizontal)"""
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
        """Genere une description enrichie textuelle"""

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


def extract_adaptive_features(img: np.ndarray, img_original: np.ndarray = None,
                               use_learned_classifier: bool = True) -> AdaptiveFeatures:
    """
    Fonction utilitaire pour extraction adaptative

    Args:
        img: Image preprocessed (grayscale)
        img_original: Image originale (couleur)
        use_learned_classifier: Utiliser le classificateur appris si disponible

    Returns:
        AdaptiveFeatures
    """
    extractor = AdaptiveFeatureExtractor(use_learned_classifier=use_learned_classifier)
    return extractor.extract(img, img_original)
