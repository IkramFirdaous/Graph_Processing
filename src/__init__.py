"""
Computer Vision Pipeline for Graph Image Analysis

Modules:
- preprocessing: Preprocessing (resize, CLAHE, bilateral, morphologie)
- detection: Detection de primitives (contours, Hough, Hu moments)
- image_classifier: Classification heuristique du type de diagramme
- feature_descriptors: Descripteurs visuels (HOG, LBP, ORB)
- learned_classifier: Classification apprise (Random Forest + PCA)
- adaptive_extractor: Extraction adaptative de features
"""

__version__ = "0.2.0"

from .utils import load_image_from_data, pil_to_numpy, numpy_to_pil
from .preprocessing import ImagePreprocessor, preprocess_image
from .detection import GraphPrimitiveDetector, detect_primitives
from .image_classifier import ImageTypeClassifier, DiagramType, classify_image_type
from .feature_descriptors import FeatureDescriptorExtractor, extract_feature_vector
from .learned_classifier import LearnedClassifier
from .adaptive_extractor import AdaptiveFeatureExtractor, extract_adaptive_features
