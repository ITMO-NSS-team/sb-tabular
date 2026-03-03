import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy import stats
from typing import List, Tuple, Optional

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 256) -> float:
    """
    Cut-off Wasserstein distance (SWD). 
    Evaluates the similarity of the joint distribution of all features.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    # Centering the data
    Xc, Yc = X_t - X_t.mean(0), Y_t - Y_t.mean(0)
    
    # Generating random projection
    thetas = torch.randn(n_proj, X_t.shape[1])
    thetas = thetas / thetas.norm(dim=1, keepdim=True)

    sw2 = 0.0
    for theta in thetas:
        # Projecting multidimensional data onto a line
        x1 = Xc @ theta
        y1 = Yc @ theta
        # We consider 1 day to be a weekend because of the schedules
        x1, _ = torch.sort(x1)
        y1, _ = torch.sort(y1)
        sw2 += F.mse_loss(x1, y1, reduction="mean")
        
    return float(sw2 / n_proj)

def avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    return float(np.mean([wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols]))

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Maximum average discrepancy in RBF units."""
    X, Y = np.asarray(X), np.asarray(Y)
    # Subsampling to calculate sigma (acceleration)
    X_sub = X[:1000]
    if sigma is None:
        dists = cdist(X_sub, X_sub, metric='euclidean')
        sigma = np.median(dists[dists > 0]) or 1.0
            
    def kernel(a, b, s):
        sq_dist = cdist(a, b, metric='sqeuclidean')
        return np.exp(-sq_dist / (2 * s**2))

    k_xx = kernel(X_sub, X_sub, sigma).mean()
    k_yy = kernel(Y[:1000], Y[:1000], sigma).mean()
    k_xy = kernel(X_sub, Y[:1000], sigma).mean()
    return float(k_xx + k_yy - 2 * k_xy)

def calculate_marginal_kl(X_real: np.ndarray, X_syn: np.ndarray, bins: int = 50) -> float:
    """The average KL is a divergence by all criteria (marginal fidelity)."""
    vals = []
    epsilon = 1e-8 
    for j in range(X_real.shape[1]):
        real_col, syn_col = X_real[:, j], X_syn[:, j]
        low, high = min(real_col.min(), syn_col.min()), max(real_col.max(), syn_col.max())
        if high == low:
            vals.append(0.0)
            continue
        p, _ = np.histogram(real_col, bins=np.linspace(low, high, bins+1), density=True)
        q, _ = np.histogram(syn_col, bins=np.linspace(low, high, bins+1), density=True)
        vals.append(float(stats.entropy(p + epsilon, q + epsilon)))
    return float(np.mean(vals))

def calculate_frobenius_corr_diff(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """Correlation distance (Frobenius norm of the difference of Pearson matrices)."""
    C_real = np.nan_to_num(np.corrcoef(X_real, rowvar=False))
    C_syn = np.nan_to_num(np.corrcoef(X_syn, rowvar=False))
    return float(np.linalg.norm(C_real - C_syn, ord="fro"))