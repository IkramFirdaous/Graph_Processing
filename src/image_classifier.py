"""
Image Type Classifier - Classifie les diagrammes selon leur structure

Identifie automatiquement le type de diagramme pour adapter la stratégie d'extraction
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class DiagramType(Enum):
    """Types de diagrammes supportés"""
    NETWORK_GRAPH = "network_graph"          # Graphes avec nodes/edges
    FLOWCHART = "flowchart"                  # Organigrammes
    PIE_CHART = "pie_chart"                  # Diagrammes circulaires
    BAR_CHART = "bar_chart"                  # Histogrammes
    INFOGRAPHIC = "infographic"              # Infographies mixtes
    TREE_DIAGRAM = "tree_diagram"            # Arbres hiérarchiques
    TIMELINE = "timeline"                    # Lignes temporelles
    VENN_DIAGRAM = "venn_diagram"            # Diagrammes de Venn
    TABLE = "table"                          # Tableaux
    UNKNOWN = "unknown"                      # Type non déterminé


@dataclass
class ImageTypeFeatures:
    """Features pour classifier le type d'image"""
    # Geometric features
    has_circles: bool
    has_rectangles: bool
    has_lines: bool
    num_circles: int
    num_rectangles: int
    num_lines: int

    # Structural features
    has_radial_symmetry: bool
    has_grid_layout: bool
    has_hierarchical_structure: bool

    # Visual features
    color_diversity: float
    text_density: float
    edge_density: float

    # Predicted type
    diagram_type: DiagramType
    confidence: float


class ImageTypeClassifier:
    """
    Classifie automatiquement le type de diagramme

    Méthode: Extraction de features géométriques + règles heuristiques
    """

    def __init__(self):
        self.min_circle_confidence = 0.8


    def classify(self, img: np.ndarray) -> ImageTypeFeatures:
        """
        Classifie le type d'image

        Args:
            img: Image preprocessed (grayscale)

        Returns:
            ImageTypeFeatures avec type détecté et confidence
        """
        # Extract geometric features
        circles = self._detect_circles(img)
        rectangles = self._detect_rectangles(img)
        lines = self._detect_lines(img)

        # Extract structural features
        radial_symmetry = self._check_radial_symmetry(img)
        grid_layout = self._check_grid_layout(rectangles)
        hierarchical = self._check_hierarchical_structure(rectangles, lines)

        # Extract visual features
        color_div = self._compute_color_diversity(img)
        text_density = self._estimate_text_density(img)
        edge_density = self._compute_edge_density(img)

        # Classification rules
        diagram_type, confidence = self._classify_type(
            circles, rectangles, lines,
            radial_symmetry, grid_layout, hierarchical,
            color_div, text_density
        )

        return ImageTypeFeatures(
            has_circles=len(circles) > 0,
            has_rectangles=len(rectangles) > 0,
            has_lines=len(lines) > 0,
            num_circles=len(circles),
            num_rectangles=len(rectangles),
            num_lines=len(lines),
            has_radial_symmetry=radial_symmetry,
            has_grid_layout=grid_layout,
            has_hierarchical_structure=hierarchical,
            color_diversity=color_div,
            text_density=text_density,
            edge_density=edge_density,
            diagram_type=diagram_type,
            confidence=confidence
        )


    def _detect_circles(self, img: np.ndarray) -> List[Tuple[int, int, int]]:
        """Détecte les cercles (Hough Circle Transform)"""
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(x, y, r) for x, y, r in circles]
        return []


    def _detect_rectangles(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Détecte les rectangles (contours)"""
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for cnt in contours:
            # Approximate contour
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # Check if 4 corners (rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w > 20 and h > 20:  # Min size
                    rectangles.append((x, y, w, h))

        return rectangles


    def _detect_lines(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Détecte les lignes (Hough Line Transform)"""
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is not None:
            return [(x1, y1, x2, y2) for x1, y1, x2, y2 in lines[:, 0]]
        return []


    def _check_radial_symmetry(self, img: np.ndarray) -> bool:
        """Vérifie si l'image a une symétrie radiale (pie charts)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Sample pixels at different angles
        num_angles = 8
        radius = min(w, h) // 3

        samples = []
        for i in range(num_angles):
            angle = 2 * np.pi * i / num_angles
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))

            if 0 <= x < w and 0 <= y < h:
                samples.append(img[y, x])

        # Check variance (low variance = radial symmetry)
        if len(samples) > 0:
            variance = np.var(samples)
            return variance < 500  # Threshold

        return False


    def _check_grid_layout(self, rectangles: List[Tuple[int, int, int, int]]) -> bool:
        """Vérifie si les rectangles forment une grille"""
        if len(rectangles) < 4:
            return False

        # Extract y-coordinates
        y_coords = [y for x, y, w, h in rectangles]

        # Check if aligned horizontally (same y)
        y_coords_sorted = sorted(y_coords)
        aligned_count = 0

        for i in range(len(y_coords_sorted) - 1):
            if abs(y_coords_sorted[i] - y_coords_sorted[i+1]) < 20:
                aligned_count += 1

        return aligned_count >= 2


    def _check_hierarchical_structure(
        self,
        rectangles: List[Tuple[int, int, int, int]],
        lines: List[Tuple[int, int, int, int]]
    ) -> bool:
        """Vérifie si structure hiérarchique (flowchart/tree)"""
        if len(rectangles) < 3 or len(lines) < 2:
            return False

        # Check if rectangles at different vertical levels
        y_coords = [y for x, y, w, h in rectangles]
        levels = len(set([y // 50 for y in y_coords]))  # Bin by 50px

        # Check if lines connect rectangles vertically
        vertical_lines = sum(1 for x1, y1, x2, y2 in lines if abs(x1 - x2) < 20)

        return levels >= 2 and vertical_lines >= 1


    def _compute_color_diversity(self, img: np.ndarray) -> float:
        """Calcule la diversité de couleurs (0-1)"""
        # Histogram of pixel values
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Entropy as diversity measure
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # Normalize (max entropy = log2(256) = 8)
        return entropy / 8.0


    def _estimate_text_density(self, img: np.ndarray) -> float:
        """Estime la densité de texte (0-1)"""
        # Detect small connected components (likely text)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        # Count small components (text-like)
        text_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 500:  # Text-like size
                text_components += 1

        # Normalize by image size
        total_area = img.shape[0] * img.shape[1]
        return min(1.0, text_components / (total_area / 1000))


    def _compute_edge_density(self, img: np.ndarray) -> float:
        """Calcule la densité de contours (0-1)"""
        edges = cv2.Canny(img, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]

        return edge_pixels / total_pixels


    def _classify_type(
        self,
        circles: List,
        rectangles: List,
        lines: List,
        radial_symmetry: bool,
        grid_layout: bool,
        hierarchical: bool,
        color_diversity: float,
        text_density: float
    ) -> Tuple[DiagramType, float]:
        """
        Règles de classification heuristiques

        Returns:
            (DiagramType, confidence)
        """

        # Rule 1: Radial symmetry + circles → Pie Chart
        if radial_symmetry and len(circles) > 0:
            return DiagramType.PIE_CHART, 0.9

        # Rule 2: Grid layout + rectangles → Table or Bar Chart
        if grid_layout and len(rectangles) >= 4:
            if color_diversity > 0.5:
                return DiagramType.BAR_CHART, 0.8
            else:
                return DiagramType.TABLE, 0.7

        # Rule 3: Hierarchical structure → Flowchart or Tree
        if hierarchical:
            if len(rectangles) > len(circles):
                return DiagramType.FLOWCHART, 0.85
            else:
                return DiagramType.TREE_DIAGRAM, 0.8

        # Rule 4: Many circles + lines → Network Graph
        if len(circles) >= 3 and len(lines) >= 3:
            return DiagramType.NETWORK_GRAPH, 0.8

        # Rule 5: Overlapping circles → Venn Diagram
        if 2 <= len(circles) <= 5 and len(lines) < 5:
            return DiagramType.VENN_DIAGRAM, 0.75

        # Rule 6: High text density + color diversity → Infographic
        if text_density > 0.3 and color_diversity > 0.6:
            return DiagramType.INFOGRAPHIC, 0.7

        # Rule 7: Horizontal lines → Timeline
        horizontal_lines = sum(1 for x1, y1, x2, y2 in lines if abs(y1 - y2) < 10)
        if horizontal_lines >= 2 and len(rectangles) >= 2:
            return DiagramType.TIMELINE, 0.75

        # Default: Unknown
        return DiagramType.UNKNOWN, 0.5


def classify_image_type(img: np.ndarray) -> ImageTypeFeatures:
    """
    Fonction utilitaire pour classifier une image

    Args:
        img: Image preprocessed (grayscale)

    Returns:
        ImageTypeFeatures
    """
    classifier = ImageTypeClassifier()
    return classifier.classify(img)
