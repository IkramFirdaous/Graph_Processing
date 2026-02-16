"""
Module 5.5: Dataset Analysis - Statistical Analysis and Visualization

Concepts from CV course:
- Feature analysis
- Dimensionality reduction (PCA, t-SNE)
- Clustering
- Statistical visualization

This module analyzes the dataset to understand patterns and structure
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class DatasetAnalyzer:
    """
    Analyzes extracted features across the dataset

    Tasks:
    1. Statistical analysis
    2. Dimensionality reduction
    3. Clustering
    4. Visualization
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def create_feature_matrix(self, feature_list: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Convert list of feature dictionaries to DataFrame

        Args:
            feature_list: List of feature dictionaries

        Returns:
            DataFrame with features
        """
        df = pd.DataFrame(feature_list)
        return df

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Compute statistical summaries

        Args:
            df: Feature DataFrame

        Returns:
            Dictionary with statistics
        """
        stats = {
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'median': df.median(),
            'q25': df.quantile(0.25),
            'q75': df.quantile(0.75),
        }

        return stats

    def analyze_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature correlations

        Args:
            df: Feature DataFrame

        Returns:
            Correlation matrix
        """
        return df.corr()

    def dimensionality_reduction_pca(self,
                                     df: pd.DataFrame,
                                     n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions using PCA

        Args:
            df: Feature DataFrame
            n_components: Number of components

        Returns:
            Reduced features
        """
        # Handle missing values
        df_clean = df.fillna(0)

        # Standardize
        X_scaled = self.scaler.fit_transform(df_clean)

        # PCA
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)

        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        return X_reduced

    def dimensionality_reduction_tsne(self,
                                      df: pd.DataFrame,
                                      n_components: int = 2,
                                      perplexity: int = 30) -> np.ndarray:
        """
        Reduce dimensions using t-SNE

        Args:
            df: Feature DataFrame
            n_components: Number of components
            perplexity: t-SNE perplexity parameter

        Returns:
            Reduced features
        """
        # Handle missing values
        df_clean = df.fillna(0)

        # Standardize
        X_scaled = self.scaler.fit_transform(df_clean)

        # t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_reduced = tsne.fit_transform(X_scaled)

        return X_reduced

    def cluster_features(self,
                        df: pd.DataFrame,
                        n_clusters: int = 5) -> np.ndarray:
        """
        Cluster features using K-means

        Args:
            df: Feature DataFrame
            n_clusters: Number of clusters

        Returns:
            Cluster labels
        """
        # Handle missing values
        df_clean = df.fillna(0)

        # Standardize
        X_scaled = self.scaler.fit_transform(df_clean)

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        return labels

    def visualize_distribution(self,
                              df: pd.DataFrame,
                              feature_name: str,
                              save_path: Optional[str] = None):
        """
        Visualize feature distribution

        Args:
            df: Feature DataFrame
            feature_name: Feature to visualize
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df[feature_name].dropna(), bins=50, edgecolor='black')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature_name}')
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(df[feature_name].dropna(), vert=True)
        plt.ylabel(feature_name)
        plt.title(f'Box Plot of {feature_name}')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def visualize_scatter_2d(self,
                            X_reduced: np.ndarray,
                            labels: Optional[np.ndarray] = None,
                            title: str = "2D Visualization",
                            save_path: Optional[str] = None):
        """
        Visualize 2D reduced features

        Args:
            X_reduced: 2D features
            labels: Cluster labels (optional)
            title: Plot title
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 8))

        if labels is not None:
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                                c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, edgecolors='k')

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def visualize_correlation_matrix(self,
                                    corr_matrix: pd.DataFrame,
                                    save_path: Optional[str] = None):
        """
        Visualize correlation matrix as heatmap

        Args:
            corr_matrix: Correlation matrix
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(12, 10))

        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')

        # Add labels
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

        plt.title('Feature Correlation Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def generate_report(self,
                       df: pd.DataFrame,
                       save_path: str = "analysis_report.txt"):
        """
        Generate text report of analysis

        Args:
            df: Feature DataFrame
            save_path: Path to save report
        """
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Number of samples: {len(df)}\n")
            f.write(f"Number of features: {len(df.columns)}\n\n")

            f.write("Feature Statistics:\n")
            f.write("-"*60 + "\n")

            stats = self.compute_statistics(df)

            for col in df.columns:
                f.write(f"\n{col}:\n")
                f.write(f"  Mean:   {stats['mean'][col]:.4f}\n")
                f.write(f"  Std:    {stats['std'][col]:.4f}\n")
                f.write(f"  Min:    {stats['min'][col]:.4f}\n")
                f.write(f"  Max:    {stats['max'][col]:.4f}\n")
                f.write(f"  Median: {stats['median'][col]:.4f}\n")

        print(f"Report saved to {save_path}")


def analyze_features(feature_list: List[Dict[str, float]],
                    output_dir: str = "outputs/results") -> Dict:
    """
    Convenience function for feature analysis

    Args:
        feature_list: List of feature dictionaries
        output_dir: Directory for outputs

    Returns:
        Analysis results
    """
    analyzer = DatasetAnalyzer()

    # Create feature matrix
    df = analyzer.create_feature_matrix(feature_list)

    # Compute statistics
    stats = analyzer.compute_statistics(df)

    # Dimensionality reduction
    X_pca = analyzer.dimensionality_reduction_pca(df, n_components=2)
    X_tsne = analyzer.dimensionality_reduction_tsne(df, n_components=2)

    # Clustering
    clusters = analyzer.cluster_features(df, n_clusters=5)

    # Generate report
    analyzer.generate_report(df, save_path=f"{output_dir}/analysis_report.txt")

    results = {
        'dataframe': df,
        'statistics': stats,
        'pca': X_pca,
        'tsne': X_tsne,
        'clusters': clusters,
    }

    return results
