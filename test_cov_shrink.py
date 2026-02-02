import numpy as np
import pandas as pd

def shrink_covariance_bayes(cov_short: np.ndarray, cov_long: np.ndarray, n_short: int, n_long: int):
    """
    Shrink a short-window (noisy) covariance matrix towards a long-window (stable) prior.
    
    Logic similar to beta shrinkage:
    Weight is determined by the relative uncertainty (variance) of the estimators.
    Variance of a sample covariance estimate is roughly proportional to 1/N.
    
    Var(Short) ~ k / n_short
    Var(Long)  ~ k / n_long
    
    Weight on Prior (Long) = Var(Short) / (Var(Short) + Var(Long))
                           = (1/n_short) / (1/n_short + 1/n_long)
                           = n_long / (n_long + n_short)
                           
    Parameters:
    -----------
    cov_short : np.ndarray
        Covariance matrix from short window (e.g. 30 days).
    cov_long : np.ndarray
        Covariance matrix from long window (e.g. 90 days).
    n_short : int
        Number of observations in short window.
    n_long : int
        Number of observations in long window.
        
    Returns:
    --------
    np.ndarray
        Shrunk covariance matrix.
    """
    # Uncertainty is inverse to sample size
    var_short_proxy = 1.0 / n_short
    var_long_proxy = 1.0 / n_long
    
    # Bayes weight for the PRIOR (Long window)
    # W_prior = Uncertainty_Evidence / (Uncertainty_Evidence + Uncertainty_Prior)
    w_prior = var_short_proxy / (var_short_proxy + var_long_proxy)
    
    print(f"Shrinkage Weight on Prior (90d): {w_prior:.4f}")
    print(f"Shrinkage Weight on Evidence (30d): {1-w_prior:.4f}")
    
    # Linear combination
    cov_shrunk = (1.0 - w_prior) * cov_short + w_prior * cov_long
    
    return cov_shrunk

# --- TEST ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Generate Dummy Returns (90 days, 3 assets)
    # Asset 0: High Vol
    # Asset 1: Low Vol
    # Asset 2: Correlated with 0
    returns = np.random.randn(100, 3)
    returns[:, 0] *= 2.0  # High vol
    returns[:, 2] += returns[:, 0] * 0.8 # Correlation
    
    df = pd.DataFrame(returns, columns=['A', 'B', 'C'])
    
    # 2. Compute Windows
    window_s = 30
    window_l = 90
    
    # Short window (Last 30)
    df_s = df.iloc[-window_s:]
    cov_s = df_s.cov().to_numpy()
    
    # Long window (Last 90)
    df_l = df.iloc[-window_l:]
    cov_l = df_l.cov().to_numpy()
    
    print("\n--- Short Window (30d) ---")
    print(np.round(cov_s, 4))
    
    print("\n--- Long Window (90d) - PRIOR ---")
    print(np.round(cov_l, 4))
    
    # 3. Apply Shrinkage
    cov_final = shrink_covariance_bayes(cov_s, cov_l, window_s, window_l)
    
    print("\n--- Shrunk Covariance ---")
    print(np.round(cov_final, 4))
    
    # 4. Verify specific element blending
    # Let's check Variance of A (index 0,0)
    var_s = cov_s[0,0]
    var_l = cov_l[0,0]
    var_f = cov_final[0,0]
    
    weight_prior = window_l / (window_s + window_l) # 90 / 120 = 0.75
    expected = (1 - weight_prior) * var_s + weight_prior * var_l
    
    print(f"\nVerification (Element 0,0):")
    print(f"Short: {var_s:.4f}")
    print(f"Long:  {var_l:.4f}")
    print(f"Calc:  {var_f:.4f}")
    print(f"Exp:   {expected:.4f}")
    
    if np.isclose(var_f, expected):
        print("SUCCESS: Weighting logic holds.")
    else:
        print("FAIL: Mismatch.")
