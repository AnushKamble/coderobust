import math
import re
import collections
from typing import List, Union
import numpy as np

def compute_text_token_entropy(text: str) -> float:
    """
    Computes genuine token-level empirical Shannon entropy:
    H(S) = - sum(p * log2(p)) for p > 0
    from the empirical vocabulary distribution of the code response.
    Eliminates synthetic random noise; guarantees mathematically sound reproducibility.
    """
    tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+|[^\w\s]', text)
    if not tokens:
        return 0.0
    total = len(tokens)
    counts = collections.Counter(tokens)
    probs = [c / total for c in counts.values()]
    return calculate_shannon_entropy(probs)

def calculate_shannon_entropy(probabilities: Union[List[float], np.ndarray]) -> float:
    """
    Computes token-level Shannon Entropy:
    H(X) = - sum(p * log2(p)) for p > 0.
    """
    probs = np.array(probabilities, dtype=np.float64)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    # Normalize to valid probability distribution if needed
    total = np.sum(probs)
    if total > 0 and not np.isclose(total, 1.0):
        probs = probs / total
    return float(-np.sum(probs * np.log2(probs)))

def calculate_entropy_drift(clean_entropy: float, perturbed_entropy: float) -> float:
    """
    Computes entropy drift:
    Delta H = H_perturbed - H_clean.
    Positive values signify heightened prediction uncertainty under perturbation.
    """
    return float(perturbed_entropy - clean_entropy)

def calculate_relative_robustness_score(clean_pass_rate: float, perturbed_pass_rate: float) -> float:
    """
    Computes the Relative Robustness Score (RRS):
    RRS = (Pass@1_perturbed / Pass@1_clean) * 100%
    """
    if clean_pass_rate <= 0:
        return 0.0
    return float((perturbed_pass_rate / clean_pass_rate) * 100.0)
