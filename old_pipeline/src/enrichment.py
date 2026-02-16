"""
Module 5.6: Text Enrichment - Augmenting Descriptions with Visual Features

Concepts from CV course:
- Multimodal understanding
- Image-text alignment
- Feature-based annotation

This module enriches text descriptions with objective visual measurements
"""

import numpy as np
from typing import Dict, List, Optional


class TextEnricher:
    """
    Enriches text descriptions with visual features extracted from images

    Goal: Add objective, measurable information from image analysis
    NOT generative text, just structured annotations
    """

    def __init__(self):
        pass

    def classify_graph_complexity(self, features: Dict[str, float]) -> str:
        """
        Classify graph visual complexity

        Args:
            features: Extracted features

        Returns:
            Complexity label: 'simple', 'moderate', 'complex'
        """
        num_nodes = features.get('num_nodes', 0)
        num_edges = features.get('num_edges', 0)

        if num_nodes <= 5 and num_edges <= 10:
            return 'simple'
        elif num_nodes <= 15 and num_edges <= 30:
            return 'moderate'
        else:
            return 'complex'

    def classify_graph_type(self, features: Dict[str, float]) -> str:
        """
        Infer visual graph type from features

        Args:
            features: Extracted features

        Returns:
            Graph type: 'linear', 'hierarchical', 'dense', 'sparse', 'tree-like'
        """
        num_nodes = features.get('num_nodes', 0)
        num_edges = features.get('num_edges', 0)
        graph_density = features.get('graph_density', 0)
        num_components = features.get('num_components', 0)

        if num_nodes == 0:
            return 'empty'

        # Linear structure: few nodes, connected sequentially
        if num_nodes <= 10 and num_edges == num_nodes - 1:
            return 'linear'

        # Hierarchical: tree-like structure
        if num_components == 1 and num_edges == num_nodes - 1:
            return 'tree-like'

        # Dense: many connections
        if graph_density > 0.5:
            return 'dense'

        # Sparse: few connections
        if graph_density < 0.2:
            return 'sparse'

        return 'network'

    def classify_layout(self, features: Dict[str, float]) -> str:
        """
        Classify visual layout pattern

        Args:
            features: Extracted features

        Returns:
            Layout type: 'horizontal', 'vertical', 'grid', 'circular', 'scattered'
        """
        h_align = features.get('horizontal_alignment', 0)
        v_align = features.get('vertical_alignment', 0)
        spatial_spread_x = features.get('spatial_spread_x', 0)
        spatial_spread_y = features.get('spatial_spread_y', 0)

        # Strong alignment indicates linear layout
        if h_align > 0.7:
            return 'horizontal'
        elif v_align > 0.7:
            return 'vertical'

        # Similar spread in both directions suggests grid or circular
        if abs(spatial_spread_x - spatial_spread_y) < 10:
            return 'grid-like'

        return 'scattered'

    def generate_enrichment(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Generate enrichment annotations from features

        Args:
            features: Extracted visual features

        Returns:
            Enrichment dictionary with structured annotations
        """
        enrichment = {}

        # Basic counts
        enrichment['node_count'] = int(features.get('num_nodes', 0))
        enrichment['edge_count'] = int(features.get('num_edges', 0))
        enrichment['component_count'] = int(features.get('num_components', 0))

        # Classifications
        enrichment['complexity'] = self.classify_graph_complexity(features)
        enrichment['graph_type'] = self.classify_graph_type(features)
        enrichment['layout'] = self.classify_layout(features)

        # Density measures
        enrichment['node_density'] = round(features.get('node_density', 0), 6)
        enrichment['edge_density'] = round(features.get('edge_density', 0), 6)
        enrichment['graph_density'] = round(features.get('graph_density', 0), 3)

        # Spatial measures
        enrichment['avg_node_distance'] = round(features.get('avg_node_distance', 0), 2)

        # Degree information
        enrichment['avg_degree'] = round(features.get('avg_degree_estimate', 0), 2)

        # Shape information
        enrichment['shape_types'] = int(features.get('num_shape_types', 0))

        return enrichment

    def format_enrichment_text(self, enrichment: Dict) -> str:
        """
        Format enrichment as human-readable text

        Args:
            enrichment: Enrichment dictionary

        Returns:
            Formatted text description
        """
        lines = []

        lines.append(f"Graph Structure: {enrichment['graph_type']}, {enrichment['complexity']} complexity")
        lines.append(f"Layout: {enrichment['layout']}")
        lines.append(f"Nodes: {enrichment['node_count']}, Edges: {enrichment['edge_count']}")

        if enrichment['component_count'] > 1:
            lines.append(f"Components: {enrichment['component_count']} (disconnected)")

        if enrichment['avg_degree'] > 0:
            lines.append(f"Average degree: {enrichment['avg_degree']:.1f}")

        return " | ".join(lines)

    def augment_caption(self,
                       original_caption: str,
                       enrichment: Dict) -> Dict[str, str]:
        """
        Augment original caption with enrichment

        Args:
            original_caption: Original text description
            enrichment: Enrichment dictionary

        Returns:
            Dictionary with original and augmented captions
        """
        enrichment_text = self.format_enrichment_text(enrichment)

        result = {
            'original_caption': original_caption,
            'enrichment_structured': enrichment,
            'enrichment_text': enrichment_text,
            'augmented_caption': f"{original_caption} [Visual Analysis: {enrichment_text}]"
        }

        return result


def enrich_text(original_text: str, visual_features: Dict[str, float]) -> Dict:
    """
    Convenience function for text enrichment

    Args:
        original_text: Original caption/description
        visual_features: Extracted visual features

    Returns:
        Enriched text dictionary
    """
    enricher = TextEnricher()

    # Generate enrichment
    enrichment = enricher.generate_enrichment(visual_features)

    # Augment caption
    result = enricher.augment_caption(original_text, enrichment)

    return result
