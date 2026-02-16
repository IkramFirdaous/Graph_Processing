"""
Module 5.2: Primitive Detection - Detecting Visual Elements

Concepts from CV course:
- Edge detection (Canny, Sobel)
- Hough Transform (lines, circles)
- Contour detection
- Morphological operations
- Feature detection

This module detects primitive visual elements in graph images:
- Nodes (circles, rectangles, polygons)
- Edges (lines, arrows, connections)
- Text regions (OCR ready)
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedNode:
    """Represents a detected node in the graph"""
    center: Tuple[int, int]  # (x, y)
    shape: str  # 'circle', 'rectangle', 'polygon'
    size: Tuple[int, int]  # (width, height) or (radius, radius) for circles
    contour: np.ndarray
    area: float
    confidence: float


@dataclass
class DetectedEdge:
    """Represents a detected edge/line in the graph"""
    start: Tuple[int, int]  # (x, y)
    end: Tuple[int, int]  # (x, y)
    angle: float  # degrees
    length: float
    is_arrow: bool


@dataclass
class TextRegion:
    """Represents a detected text region"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float


class GraphPrimitiveDetector:
    """
    Detects primitive elements in graph images

    Pipeline:
    1. Edge detection
    2. Contour extraction
    3. Shape classification (nodes)
    4. Line detection (edges)
    5. Text region detection
    """

    def __init__(self):
        pass

    def detect_edges_canny(self, img: np.ndarray,
                          low_threshold: int = 50,
                          high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detector

        Args:
            img: Input grayscale image
            low_threshold: Low threshold for hysteresis
            high_threshold: High threshold for hysteresis

        Returns:
            Binary edge map
        """
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(img, low_threshold, high_threshold)
        return edges

    def find_contours(self, edge_map: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in edge map

        Args:
            edge_map: Binary edge map

        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    def classify_shape(self, contour: np.ndarray) -> str:
        """
        Classify contour shape

        Args:
            contour: Contour points

        Returns:
            Shape type: 'circle', 'rectangle', 'triangle', 'polygon'
        """
        # Approximate polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        num_vertices = len(approx)

        # Classify based on number of vertices
        if num_vertices == 3:
            return 'triangle'
        elif num_vertices == 4:
            # Check if rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                return 'square'
            else:
                return 'rectangle'
        elif num_vertices > 8:
            # Check circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    return 'circle'

        return 'polygon'

    def detect_nodes(self, img: np.ndarray,
                    min_area: int = 100,
                    max_area: int = 50000) -> List[DetectedNode]:
        """
        Detect nodes (circles, rectangles, etc.)

        Args:
            img: Input grayscale image
            min_area: Minimum node area
            max_area: Maximum node area

        Returns:
            List of detected nodes
        """
        # Detect edges
        edges = self.detect_edges_canny(img)

        # Morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours = self.find_contours(closed)

        nodes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Classify shape
            shape = self.classify_shape(contour)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            # Calculate confidence (based on contour properties)
            peri = cv2.arcLength(contour, True)
            if peri > 0:
                compactness = (4 * np.pi * area) / (peri * peri)
                confidence = min(compactness, 1.0)
            else:
                confidence = 0.0

            node = DetectedNode(
                center=center,
                shape=shape,
                size=(w, h),
                contour=contour,
                area=area,
                confidence=confidence
            )

            nodes.append(node)

        return nodes

    def detect_lines_hough(self, img: np.ndarray,
                          threshold: int = 50,
                          min_line_length: int = 30,
                          max_line_gap: int = 10) -> List[DetectedEdge]:
        """
        Detect lines using Hough Line Transform

        Args:
            img: Input grayscale image
            threshold: Accumulator threshold
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments

        Returns:
            List of detected edges/lines
        """
        # Detect edges
        edges = self.detect_edges_canny(img)

        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        detected_edges = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                edge = DetectedEdge(
                    start=(x1, y1),
                    end=(x2, y2),
                    angle=angle,
                    length=length,
                    is_arrow=False  # Arrow detection would require additional logic
                )

                detected_edges.append(edge)

        return detected_edges

    def detect_text_regions(self, img: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using MSER or connected components

        Args:
            img: Input grayscale image

        Returns:
            List of text region bounding boxes
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        text_regions = []

        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

            # Filter by aspect ratio (text regions are typically wider than tall)
            aspect_ratio = float(w) / h if h > 0 else 0

            if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
                text_region = TextRegion(
                    bbox=(x, y, w, h),
                    confidence=1.0
                )
                text_regions.append(text_region)

        return text_regions

    def detect_circles(self, img: np.ndarray,
                      min_radius: int = 10,
                      max_radius: int = 100) -> List[Tuple[int, int, int]]:
        """
        Detect circles using Hough Circle Transform

        Args:
            img: Input grayscale image
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius

        Returns:
            List of (x, y, radius) tuples
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            return [(x, y, r) for x, y, r in circles[0, :]]

        return []

    def detect_all(self, img: np.ndarray) -> Dict:
        """
        Run full detection pipeline

        Args:
            img: Input image (grayscale or RGB)

        Returns:
            Dictionary with all detected primitives
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        results = {
            'nodes': self.detect_nodes(gray),
            'edges': self.detect_lines_hough(gray),
            'circles': self.detect_circles(gray),
            'text_regions': self.detect_text_regions(gray),
        }

        # Add statistics
        results['stats'] = {
            'num_nodes': len(results['nodes']),
            'num_edges': len(results['edges']),
            'num_circles': len(results['circles']),
            'num_text_regions': len(results['text_regions']),
        }

        return results

    def visualize_detections(self, img: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Visualize detected primitives

        Args:
            img: Original image
            detections: Detection results from detect_all()

        Returns:
            Visualization image
        """
        if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        # Draw nodes
        for node in detections['nodes']:
            cv2.drawContours(vis, [node.contour], -1, (0, 255, 0), 2)
            cv2.circle(vis, node.center, 3, (0, 0, 255), -1)

        # Draw edges
        for edge in detections['edges']:
            cv2.line(vis, edge.start, edge.end, (255, 0, 0), 2)

        # Draw circles
        for x, y, r in detections['circles']:
            cv2.circle(vis, (x, y), r, (255, 255, 0), 2)

        # Draw text regions
        for text_region in detections['text_regions']:
            x, y, w, h = text_region.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)

        return vis


def detect_primitives(img: np.ndarray) -> Dict:
    """
    Convenience function for primitive detection

    Args:
        img: Input image

    Returns:
        Detection results
    """
    detector = GraphPrimitiveDetector()
    return detector.detect_all(img)
