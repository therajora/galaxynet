"""
Specific metrics for domain shift analysis.

Implements various metrics to quantify differences between
feature and prediction distributions in different domains.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def calculate_domain_shift_metrics(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List],
    sdss_features: np.ndarray,
    splus_features: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive domain shift metrics.
    
    Args:
        sdss_predictions: Predictions in SDSS domain
        splus_predictions: Predictions in S-PLUS domain
        sdss_features: Features extracted from SDSS
        splus_features: Features extracted from S-PLUS
        
    Returns:
        Dictionary with domain shift metrics
    """
    metrics = {}
    
    # 1. Performance metrics
    metrics['performance_drop'] = calculate_performance_drop(
        sdss_predictions, splus_predictions
    )
    
    # 2. Feature distribution metrics
    metrics['feature_distribution'] = calculate_distribution_distance(
        sdss_features, splus_features
    )
    
    # 3. Prediction metrics
    metrics['prediction_distribution'] = calculate_prediction_distribution_shift(
        sdss_predictions, splus_predictions
    )
    
    # 4. Confidence metrics
    metrics['confidence_shift'] = calculate_confidence_shift(
        sdss_predictions, splus_predictions
    )
    
    # 5. Calibration metrics
    metrics['calibration_shift'] = calculate_calibration_shift(
        sdss_predictions, splus_predictions
    )
    
    return metrics


def calculate_performance_drop(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List]
) -> Dict[str, float]:
    """
    Calculate performance drop between domains.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        
    Returns:
        Dictionary with performance drops
    """
    # Calculate SDSS metrics
    sdss_metrics = {
        'accuracy': accuracy_score(sdss_predictions['y_true'], sdss_predictions['y_pred']),
        'precision': precision_score(sdss_predictions['y_true'], sdss_predictions['y_pred'], average='binary', zero_division=0),
        'recall': recall_score(sdss_predictions['y_true'], sdss_predictions['y_pred'], average='binary', zero_division=0),
        'f1': f1_score(sdss_predictions['y_true'], sdss_predictions['y_pred'], average='binary', zero_division=0),
        'auc': roc_auc_score(sdss_predictions['y_true'], sdss_predictions['y_proba'])
    }
    
    # Calculate S-PLUS metrics
    splus_metrics = {
        'accuracy': accuracy_score(splus_predictions['y_true'], splus_predictions['y_pred']),
        'precision': precision_score(splus_predictions['y_true'], splus_predictions['y_pred'], average='binary', zero_division=0),
        'recall': recall_score(splus_predictions['y_true'], splus_predictions['y_pred'], average='binary', zero_division=0),
        'f1': f1_score(splus_predictions['y_true'], splus_predictions['y_pred'], average='binary', zero_division=0),
        'auc': roc_auc_score(splus_predictions['y_true'], splus_predictions['y_proba'])
    }
    
    # Calculate drops
    performance_drop = {}
    for metric in sdss_metrics:
        drop = sdss_metrics[metric] - splus_metrics[metric]
        relative_drop = (drop / sdss_metrics[metric]) * 100 if sdss_metrics[metric] > 0 else 0
        performance_drop[f'{metric}_drop'] = drop
        performance_drop[f'{metric}_relative_drop'] = relative_drop
    
    performance_drop['sdss_metrics'] = sdss_metrics
    performance_drop['splus_metrics'] = splus_metrics
    
    return performance_drop


def calculate_distribution_distance(
    sdss_features: np.ndarray,
    splus_features: np.ndarray
) -> Dict[str, float]:
    """
    Calculate distances between feature distributions.
    
    Args:
        sdss_features: SDSS features
        splus_features: S-PLUS features
        
    Returns:
        Dictionary with calculated distances
    """
    distances = {}
    
    # 1. Jensen-Shannon Divergence
    try:
        # Reduce dimensionality if necessary
        if sdss_features.shape[1] > 100:
            pca = PCA(n_components=50)
            sdss_reduced = pca.fit_transform(sdss_features)
            splus_reduced = pca.transform(splus_features)
        else:
            sdss_reduced = sdss_features
            splus_reduced = splus_features
        
        # Calculate JS divergence for each feature
        js_divergences = []
        for i in range(min(sdss_reduced.shape[1], 10)):  # Limit to 10 features
            sdss_hist, _ = np.histogram(sdss_reduced[:, i], bins=50, density=True)
            splus_hist, _ = np.histogram(splus_reduced[:, i], bins=50, density=True)
            
            # Normalize
            sdss_hist = sdss_hist / (np.sum(sdss_hist) + 1e-8)
            splus_hist = splus_hist / (np.sum(splus_hist) + 1e-8)
            
            js_div = jensenshannon(sdss_hist, splus_hist)
            js_divergences.append(js_div)
        
        distances['jensen_shannon_divergence'] = np.mean(js_divergences)
        distances['jensen_shannon_std'] = np.std(js_divergences)
    except Exception as e:
        distances['jensen_shannon_divergence'] = np.nan
        distances['jensen_shannon_std'] = np.nan
    
    # 2. Wasserstein Distance (1D)
    try:
        wasserstein_distances = []
        for i in range(min(sdss_features.shape[1], 10)):
            wd = stats.wasserstein_distance(sdss_features[:, i], splus_features[:, i])
            wasserstein_distances.append(wd)
        
        distances['wasserstein_distance'] = np.mean(wasserstein_distances)
        distances['wasserstein_std'] = np.std(wasserstein_distances)
    except Exception as e:
        distances['wasserstein_distance'] = np.nan
        distances['wasserstein_std'] = np.nan
    
    # 3. Kolmogorov-Smirnov Test
    try:
        ks_statistics = []
        ks_pvalues = []
        for i in range(min(sdss_features.shape[1], 10)):
            ks_stat, ks_pval = stats.ks_2samp(sdss_features[:, i], splus_features[:, i])
            ks_statistics.append(ks_stat)
            ks_pvalues.append(ks_pval)
        
        distances['ks_statistic'] = np.mean(ks_statistics)
        distances['ks_pvalue'] = np.mean(ks_pvalues)
    except Exception as e:
        distances['ks_statistic'] = np.nan
        distances['ks_pvalue'] = np.nan
    
    # 4. MMD (Maximum Mean Discrepancy) - aproximação simples
    try:
        mmd = calculate_mmd_distance(sdss_features, splus_features)
        distances['mmd_distance'] = mmd
    except Exception as e:
        distances['mmd_distance'] = np.nan
    
    return distances


def calculate_prediction_distribution_shift(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List]
) -> Dict[str, float]:
    """
    Calculate shift in prediction distribution.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        
    Returns:
        Metrics of shift in predictions
    """
    shift_metrics = {}
    
    # 1. Shift in probability distribution
    sdss_proba = np.array(sdss_predictions['y_proba'])
    splus_proba = np.array(splus_predictions['y_proba'])
    
    shift_metrics['probability_mean_shift'] = np.mean(sdss_proba) - np.mean(splus_proba)
    shift_metrics['probability_std_shift'] = np.std(sdss_proba) - np.std(splus_proba)
    
    # 2. Shift in predicted class distribution
    sdss_pred_dist = np.bincount(sdss_predictions['y_pred'], minlength=2) / len(sdss_predictions['y_pred'])
    splus_pred_dist = np.bincount(splus_predictions['y_pred'], minlength=2) / len(splus_predictions['y_pred'])
    
    shift_metrics['class_distribution_shift'] = np.sum(np.abs(sdss_pred_dist - splus_pred_dist))
    
    # 3. Prediction entropy
    def entropy(probs):
        probs = np.array(probs)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))
    
    sdss_entropy = entropy(sdss_pred_dist)
    splus_entropy = entropy(splus_pred_dist)
    shift_metrics['entropy_shift'] = sdss_entropy - splus_entropy
    
    return shift_metrics


def calculate_confidence_shift(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List]
) -> Dict[str, float]:
    """
    Calculate shift in prediction confidence.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        
    Returns:
        Metrics of shift in confidence
    """
    confidence_metrics = {}
    
    sdss_proba = np.array(sdss_predictions['y_proba'])
    splus_proba = np.array(splus_predictions['y_proba'])
    
    # Confidence as distance from 0.5 threshold
    sdss_confidence = np.abs(sdss_proba - 0.5)
    splus_confidence = np.abs(splus_proba - 0.5)
    
    confidence_metrics['mean_confidence_shift'] = np.mean(sdss_confidence) - np.mean(splus_confidence)
    confidence_metrics['std_confidence_shift'] = np.std(sdss_confidence) - np.std(splus_confidence)
    
    # Proportion of high confidence predictions (>0.8)
    sdss_high_conf = np.mean(sdss_confidence > 0.3)
    splus_high_conf = np.mean(splus_confidence > 0.3)
    confidence_metrics['high_confidence_shift'] = sdss_high_conf - splus_high_conf
    
    return confidence_metrics


def calculate_calibration_shift(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List]
) -> Dict[str, float]:
    """
    Calculate shift in prediction calibration.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        
    Returns:
        Metrics of shift in calibration
    """
    calibration_metrics = {}
    
    # Expected Calibration Error (ECE) approximation
    def calculate_ece(y_true, y_proba, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    sdss_ece = calculate_ece(
        np.array(sdss_predictions['y_true']),
        np.array(sdss_predictions['y_proba'])
    )
    splus_ece = calculate_ece(
        np.array(splus_predictions['y_true']),
        np.array(splus_predictions['y_proba'])
    )
    
    calibration_metrics['ece_shift'] = sdss_ece - splus_ece
    calibration_metrics['sdss_ece'] = sdss_ece
    calibration_metrics['splus_ece'] = splus_ece
    
    return calibration_metrics


def calculate_mmd_distance(
    sdss_features: np.ndarray,
    splus_features: np.ndarray,
    gamma: float = 1.0
) -> float:
    """
    Calculate Maximum Mean Discrepancy (MMD) between distributions.
    
    Args:
        sdss_features: SDSS features
        splus_features: S-PLUS features
        gamma: RBF kernel parameter
        
    Returns:
        MMD distance
    """
    # Sample to reduce computational complexity
    n_samples = min(1000, len(sdss_features), len(splus_features))
    sdss_sample = sdss_features[:n_samples]
    splus_sample = splus_features[:n_samples]
    
    # RBF kernel
    def rbf_kernel(X, Y, gamma):
        X_norm = np.sum(X**2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y**2, axis=1)[np.newaxis, :]
        dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * dist)
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    k_xx = rbf_kernel(sdss_sample, sdss_sample, gamma)
    k_yy = rbf_kernel(splus_sample, splus_sample, gamma)
    k_xy = rbf_kernel(sdss_sample, splus_sample, gamma)
    
    mmd_squared = (
        np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    )
    
    return np.sqrt(max(0, mmd_squared))


def calculate_feature_statistics(
    sdss_features: np.ndarray,
    splus_features: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate descriptive statistics of features.
    
    Args:
        sdss_features: SDSS features
        splus_features: S-PLUS features
        
    Returns:
        Feature statistics
    """
    stats_dict = {}
    
    # Basic statistics
    stats_dict['sdss_features'] = {
        'mean': np.mean(sdss_features, axis=0).tolist(),
        'std': np.std(sdss_features, axis=0).tolist(),
        'min': np.min(sdss_features, axis=0).tolist(),
        'max': np.max(sdss_features, axis=0).tolist(),
        'shape': sdss_features.shape
    }
    
    stats_dict['splus_features'] = {
        'mean': np.mean(splus_features, axis=0).tolist(),
        'std': np.std(splus_features, axis=0).tolist(),
        'min': np.min(splus_features, axis=0).tolist(),
        'max': np.max(splus_features, axis=0).tolist(),
        'shape': splus_features.shape
    }
    
    # Differences in statistics
    stats_dict['feature_differences'] = {
        'mean_diff': np.mean(np.abs(
            np.mean(sdss_features, axis=0) - np.mean(splus_features, axis=0)
        )),
        'std_diff': np.mean(np.abs(
            np.std(sdss_features, axis=0) - np.std(splus_features, axis=0)
        ))
    }
    
    return stats_dict
