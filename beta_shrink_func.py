
import numpy as np
import pandas as pd
import os
from typing import Union, List, Optional
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def load_sector_groupings(csv_path: str, valid_tickers: list = None) -> dict:
    """
    Load sector groupings from the Binance categories CSV.
    
    Parameters:
    -----------
    csv_path : str
        Path to the binance_perp_categories_cg_with_sector.csv file
    valid_tickers : list, optional
        List of valid tickers in the dataset (e.g. ['BTCUSDT', 'ETHUSDT']).
        If provided, only tickers present in this list will be included in the sectors.
        If None, assumes all tickers are {Base_Asset}USDT.
        
    Returns:
    --------
    dict
        Dictionary mapping sector names to lists of tickers.
        e.g. {"Priority 1: Core / Base Layer": ["BTCUSDT", "ETHUSDT", ...]}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sector CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if 'Primary_Sector' not in df.columns or 'Binance_Base_Asset' not in df.columns:
        raise ValueError("CSV missing required columns: Primary_Sector, Binance_Base_Asset")
    
    # Clean sector names (drop NaNs)
    df = df.dropna(subset=['Primary_Sector', 'Binance_Base_Asset'])
    
    sector_map = {}
    
    valid_tickers_set = set(valid_tickers) if valid_tickers else None
    
    for sector, group in df.groupby('Primary_Sector'):
        base_assets = group['Binance_Base_Asset'].astype(str).tolist()
        
        # Convert base assets to USDT tickers
        # Assuming standard Binance format: BASE + USDT
        tickers = [f"{asset}USDT" for asset in base_assets]
        
        # Filter if universe is provided
        if valid_tickers_set:
            tickers = [t for t in tickers if t in valid_tickers_set]
            
        if tickers:
            sector_map[sector] = tickers
            
    return sector_map

def _cluster_robust_se(X, resid, clusters):
    """
    Return cluster-robust variance for OLS parameters.
    X: (n, k) design matrix
    resid: (n,) residuals
    clusters: array-like length n with cluster labels (hashable)
    """
    X = np.asarray(X)
    resid = np.asarray(resid)
    clusters = np.asarray(clusters)

    n, k = X.shape
    
    # Check for singular matrix or empty
    if n == 0:
        return np.full(k, np.nan), np.full((k, k), np.nan)

    # (X'X)^-1
    try:
        XtX = X.T @ X
        # Check if XtX contains NaNs or Infs
        if not np.isfinite(XtX).all():
             print("XtX contains non-finite values.")
             return np.full(k, np.nan), np.full((k, k), np.nan)
             
        xtx_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered in cluster-robust SE calculation.")
        return np.full(k, np.nan), np.full((k, k), np.nan)

    # Sum over clusters of (X_c' * resid_c) (vector of length k)
    # Vectorized approach:
    # 1. Map clusters to 0..M-1
    #converts cluster labels (like ['BTC', 'ETH', 'SOL']) into list of integers [0, 1, 2...] computers handle much faster.
    # Output 1 (cluster_ids): ['BTC', 'ETH', 'SOL'] Output 2 (inv): [0, 1, 2]
    cluster_ids, inv = np.unique(clusters, return_inverse=True)
    M = len(cluster_ids)
    
    # 2. Compute observation-level scores: X * resid
    # Shape (N, k)
    scores_obs = X * resid[:, None]
    
    # 3. Sum scores by cluster
    # Shape (M, k)
    scores_cluster = np.zeros((M, k))
    np.add.at(scores_cluster, inv, scores_obs)
    
    # 4. Compute S = sum( score_c @ score_c.T )
    # S = scores_cluster.T @ scores_cluster
    S = scores_cluster.T @ scores_cluster
    
    # cluster-robust covariance
    cov = xtx_inv @ S @ xtx_inv
    
    # small-sample correction
    N = n
    
    if M > 1 and N > k:
        scale = (M / (M - 1)) * ((N - 1) / (N - k))
        cov *= scale
    
    se = np.sqrt(np.diag(cov))
    return se, cov

def _estimate_pooled_beta(returns_slice: pd.DataFrame, 
                          valid_coins: list, 
                          market_col: str, 
                          min_obs: int, 
                          cluster_by_coin: bool) -> dict:
    """
    Internal helper to calculate pooled beta for a single time slice/window.
    """
    # 1. Stack
    # Filter only valid coins and market
    # We need market column separate
    if market_col not in returns_slice.columns:
        return {"beta": np.nan, "beta_se": np.nan, "beta_var": np.nan, "n_obs": 0, "se_ratio": np.nan}

    # Slice coins
    coin_data = returns_slice[valid_coins]
    
    # Demean Y (coin returns)
    coin_data_demeaned = coin_data - coin_data.mean()

    # Demean X (market returns) per coin
    # Expand market column to same shape as coin_data to apply coin-specific masks
    market_vals = returns_slice[market_col].values[:, None]
    
    market_expanded = pd.DataFrame(
        np.tile(market_vals, (1, coin_data.shape[1])),
        index=coin_data.index,
        columns=coin_data.columns
    )
    
    # Mask market data where coin data is missing (so mean calculation is correct)
    market_expanded = market_expanded.where(coin_data.notna())
    
    # Demean X
    x_expanded_demeaned = market_expanded - market_expanded.mean()
    
    # Stack per (date, coin)
    y_stacked = coin_data_demeaned.stack(future_stack=True) 
    x_stacked = x_expanded_demeaned.stack(future_stack=True)

    if y_stacked.empty:
        return {"beta": np.nan, "beta_se": np.nan, "beta_var": np.nan, "n_obs": 0, "se_ratio": np.nan}

    # Align X and Y (intersection of valid indices)
    common_idx = y_stacked.index.intersection(x_stacked.index)
    y_stacked = y_stacked.loc[common_idx]
    x_stacked = x_stacked.loc[common_idx]

    obs_coins = y_stacked.index.get_level_values(1)

    x_values = x_stacked.values
    y_values = y_stacked.values

    # Mask rows where market or y are NaN or Inf
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_clean = x_values[mask]
    y_clean = y_values[mask]
    coins_clean = np.array(obs_coins)[mask]

    n_obs = len(y_clean)
    if n_obs < min_obs:
        return {"beta": np.nan, "beta_se": np.nan, "beta_var": np.nan, "n_obs": n_obs, "se_ratio": np.nan}

    # Design matrix
    X = np.column_stack([np.ones(n_obs), x_clean])
    
    # Solve
    # rcond=None to let numpy decide
    beta_hat, residuals_sum_sq, rank, svals = np.linalg.lstsq(X, y_clean, rcond=None)
    
    if len(beta_hat) < 2:
        return {"beta": np.nan, "beta_se": np.nan, "beta_var": np.nan, "n_obs": n_obs, "se_ratio": np.nan}
        
    beta = float(beta_hat[1])

    # SSR
    if residuals_sum_sq.size > 0:
        ssr = residuals_sum_sq[0]
    else:
        preds = X @ beta_hat
        ssr = np.sum((y_clean - preds) ** 2)

    k = X.shape[1]
    dof = n_obs - k
    if dof <= 0:
        return {"beta": beta, "beta_se": np.nan, "beta_var": np.nan, "n_obs": n_obs, "se_ratio": np.nan}

    mse = ssr / dof

    # classical var(slope)
    x_demeaned = x_clean - np.mean(x_clean)
    Sxx = np.sum(x_demeaned ** 2)
    if Sxx == 0:
        return {"beta": beta, "beta_se": np.nan, "beta_var": np.nan, "n_obs": n_obs, "se_ratio": np.nan}
        
    beta_var_classic = mse / Sxx
    beta_se_classic = np.sqrt(beta_var_classic)

    # Clustered SE (preferred)
    if cluster_by_coin:
        with np.errstate(all='ignore'):
            se, cov = _cluster_robust_se(X, y_clean - X @ beta_hat, coins_clean)
        if np.isnan(se[1]):
             # Fallback
            beta_se = np.nan
            beta_var = np.nan
        else:
            beta_se = float(se[1])
            beta_var = float(cov[1, 1])
    else:
        beta_se = beta_se_classic
        beta_var = beta_var_classic

    se_ratio = beta_se / beta_se_classic if (beta_se_classic > 0 and cluster_by_coin) else np.nan
    #print( f"start date: {returns_slice.index[0]}, beta: {beta}, beta_se: {beta_se}, beta_var: {beta_var}, n_obs: {n_obs}, se_ratio: {se_ratio}")
    return {
        "beta": beta,
        "beta_se": beta_se,
        "beta_var": beta_var,
        "n_obs": int(n_obs),
        "se_ratio": se_ratio
    }

def _process_window(i, dates, returns, window, min_periods, valid_coins, market_col, min_obs, cluster_by_coin):
    """
    Helper function to process a single window for parallel execution.
    """
    end_idx = i + 1
    start_idx = max(0, end_idx - window)
    current_len = end_idx - start_idx
    
    if current_len < min_periods:
        return {"beta": np.nan, "beta_se": np.nan, "beta_var": np.nan, "n_obs": 0, "se_ratio": np.nan}, dates[i]
        
    window_df = returns.iloc[start_idx:end_idx]
    stats = _estimate_pooled_beta(window_df, valid_coins, market_col, min_obs, cluster_by_coin)
    return stats, dates[i]

def pooled_sector_beta(returns: pd.DataFrame, 
                       sector_coins: list, 
                       market_col: str = "BTCUSDT",
                       window: int = 30*24,
                       min_periods: int = 30*24,
                       min_obs: int = 3, 
                       step: int = 24, 
                       cluster_by_coin: bool = True,
                       n_jobs: int = 1) -> pd.DataFrame:
    """
    Calculate rolling pooled beta for a sector.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (index=date, columns=tickers)
    sector_coins : list
        List of tickers in the sector.
    market_col : str
        Market ticker column name.
    window : int
        Rolling window size (number of periods/days/hours).
    min_periods : int
        Minimum number of periods required in the window to attempt estimation.
    min_obs : int
        Minimum number of TOTAL stacked observations (across all coins and days) 
        required to return a valid beta.
    step : int, default 24 (24 hours = 1 day for hourly data)
        The number of periods to step forward for each rolling window calculation. 
        A step of 1 means a continuously moving window. A step > 1 means non-overlapping 
        or partially overlapping windows.
    cluster_by_coin : bool
        Whether to use cluster-robust SE (clustered by coin).
    n_jobs : int, default 1
        Number of parallel jobs to run. -1 means using all processors.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['beta', 'beta_se', 'beta_var', 'n_obs'], indexed by date.
    """
    
    # 1. Resolve Market Column
    if market_col not in returns.columns:
        if market_col == "BTCUSDT" and "BTC" in returns.columns:
            market_col = "BTC"
        elif market_col == "BTC" and "BTCUSDT" in returns.columns:
            market_col = "BTCUSDT"
        else:
            # Return empty structure with NaNs
            return pd.DataFrame(np.nan, index=returns.index, columns=["beta", "beta_se", "beta_var", "n_obs", "se_ratio"])

    # 2. Filter Valid Coins
    valid_coins = [c for c in sector_coins if c in returns.columns and c != market_col]
    if not valid_coins:
        return pd.DataFrame(np.nan, index=returns.index, columns=["beta", "beta_se", "beta_var", "n_obs", "se_ratio"])

    # 3. Iterate Rolling Window
    dates = returns.index
    indices = range(0, len(dates), step)

    if n_jobs != 1:
        # Parallel execution
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_window)(i, dates, returns, window, min_periods, valid_coins, market_col, min_obs, cluster_by_coin)
            for i in indices
        )
        # Unzip results
        results, output_dates = zip(*results_list)
        results = list(results)
        output_dates = list(output_dates)
    else:
        # Sequential execution
        results = []
        output_dates = [] 
        
        for i in indices:
            stats, date = _process_window(i, dates, returns, window, min_periods, valid_coins, market_col, min_obs, cluster_by_coin)
            results.append(stats)
            output_dates.append(date)
        
    # 4. Handle the "remainder" / last data point
    # If the loop didn't land exactly on the last index, we might miss the most recent data.
    # For a rolling window, we typically want the latest available estimate.
    last_idx = len(dates) - 1
    if last_idx >= 0 and (not results or output_dates[-1] != dates[last_idx]):
        end_idx = last_idx + 1
        start_idx = max(0, end_idx - window)
        
        if (end_idx - start_idx) >= min_periods:
            window_df = returns.iloc[start_idx:end_idx]
            stats = _estimate_pooled_beta(window_df, valid_coins, market_col, min_obs, cluster_by_coin)
            results.append(stats)
            output_dates.append(dates[last_idx])
            
    # 5. Construct Result DataFrame
    res_df = pd.DataFrame(results, index=output_dates)
    
    return res_df

def _calculate_single_robust_beta(coin_series, market_series, window, min_periods):
    """
    Helper to calculate rolling beta and robust variance for a single coin using iterative OLS.
    This is necessary because RollingOLS does not support robust covariance types (HC0) directly.
    """
    # Align
    common_idx = coin_series.index.intersection(market_series.index)
    if len(common_idx) < min_periods:
         return pd.Series(np.nan, index=coin_series.index), pd.Series(np.nan, index=coin_series.index)
         
    y = coin_series.loc[common_idx]
    x = market_series.loc[common_idx]
    dates = common_idx
    
    # Pre-allocate
    betas = np.full(len(dates), np.nan)
    beta_vars = np.full(len(dates), np.nan)
    
    # We loop through the data
    # This is O(N) but with OLS inside, so O(N * window_size) operations roughly.
    # For speed, we only iterate valid end-points.
    
    y_vals = y.values
    x_vals = x.values
    
    # Iterate
    # start at min_periods-1 to have enough data
    for i in range(min_periods - 1, len(dates)):
        start_idx = max(0, i + 1 - window)
        end_idx = i + 1
        
        # Slice
        y_sub = y_vals[start_idx:end_idx]
        x_sub = x_vals[start_idx:end_idx]
        
        # Check valid obs (NaN handling)
        mask = np.isfinite(y_sub) & np.isfinite(x_sub)
        n_valid = np.sum(mask)
        
        if n_valid < min_periods:
            continue
            
        y_clean = y_sub[mask]
        x_clean = x_sub[mask]
        
        # OLS
        # Add constant
        X_clean = sm.add_constant(x_clean)
        
        try:
            model = sm.OLS(y_clean, X_clean)
            results = model.fit(cov_type='HC0')
            
            # Beta is index 1 (const is 0)
            if len(results.params) > 1:
                betas[i] = results.params[1]
                # Variance of Beta is [1, 1]
                # results.cov_params() returns DataFrame or Array depending on input
                # standard OLS fit returns array usually if X is array? 
                # statsmodels returns pandas if input was pandas. Here we passed numpy.
                # So it returns numpy array (2x2).
                beta_vars[i] = results.cov_params()[1, 1]
        except Exception:
            continue
            
    return pd.Series(betas, index=dates), pd.Series(beta_vars, index=dates)

def rolling_coin_beta(returns: pd.DataFrame, 
                      market_col: str = "BTCUSDT", 
                      window: int = 30*24, 
                      min_periods: int = 20,
                      use_robust_se: bool = True,
                      n_jobs: int = -1) -> dict:
    """
    Calculate rolling beta for EACH coin in the returns DataFrame against the market column.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of asset returns (index=date, columns=tickers)
    market_col : str, default "BTCUSDT"
        Column name for the market return.
    window : int, default 720 (30 days * 24h)
        Size of the rolling window.
    min_periods : int, default 20
        Minimum number of valid observations required.
    use_robust_se : bool, default False
        If True, use statsmodels RollingOLS with HC0 robust standard errors. 
        Note: This is significantly slower than the default vectorized method.
    n_jobs : int, default 1
        Number of parallel jobs to use if use_robust_se is True.
        
    Returns:
    --------
    dict of pd.DataFrame
        {
            "beta": DataFrame (dates x coins),
            "beta_var": DataFrame (dates x coins),
            "n_obs": DataFrame (dates x coins)
        }
    """
    # 1. Validate and prep inputs
    if market_col not in returns.columns:
        if market_col == "BTCUSDT" and "BTC" in returns.columns:
            market_col = "BTC"
        elif market_col == "BTC" and "BTCUSDT" in returns.columns:
            market_col = "BTCUSDT"
        else:
            return {
                "beta": pd.DataFrame(),
                "beta_var": pd.DataFrame(),
                "n_obs": pd.DataFrame()
            }
            
    target_coins = [c for c in returns.columns if c != market_col]
    if not target_coins:
        return {
            "beta": pd.DataFrame(),
            "beta_var": pd.DataFrame(),
            "n_obs": pd.DataFrame()
        }
        
    # Isolate data
    mkt_series = returns[market_col]
    coin_returns = returns[target_coins]
    
    # 2. Robust SE Path (Slow, Loop/Parallel)
    if use_robust_se:
        # Parallel execution for speed
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_single_robust_beta)(coin_returns[coin], mkt_series, window, min_periods)
            for coin in target_coins
        )
        
        # Assemble
        beta_dict = {}
        var_dict = {}
        
        for coin, (b, v) in zip(target_coins, results_list):
            beta_dict[coin] = b
            var_dict[coin] = v
            
        betas = pd.DataFrame(beta_dict, index=returns.index)
        beta_vars = pd.DataFrame(var_dict, index=returns.index)
        n_obs = coin_returns.rolling(window=window, min_periods=min_periods).count() # Approximate
        
        return {
            "beta": betas,
            "beta_var": beta_vars,
            "n_obs": n_obs
        }

    # 3. Vectorized Path (Fast, Classical OLS SE)
    # Pandas rolling operations are highly optimized
    
    # Rolling Covariance: Cov(R_i, R_m)
    rolling_cov = coin_returns.rolling(window=window, min_periods=min_periods).cov(mkt_series)
    
    # Rolling Variance: Var(R_m)
    rolling_mkt_var = mkt_series.rolling(window=window, min_periods=min_periods).var()
    
    # Calculate Beta
    betas = rolling_cov.div(rolling_mkt_var, axis=0)
    
    # Calculate Variance of Beta (for shrinkage)
    rolling_corr = coin_returns.rolling(window=window, min_periods=min_periods).corr(mkt_series)
    rolling_coin_var = coin_returns.rolling(window=window, min_periods=min_periods).var()
    
    n_obs = coin_returns.rolling(window=window, min_periods=min_periods).count()
    
    # Var(beta) = [Var(coin) / Var(mkt)] * [(1 - corr^2) / (N - 2)]
    var_ratio = rolling_coin_var.div(rolling_mkt_var, axis=0)
    unexplained_corr = 1 - rolling_corr**2
    dof = n_obs - 2
    
    beta_vars = var_ratio * unexplained_corr / dof
    beta_vars[dof <= 0] = np.nan
    
    return {
        "beta": betas,
        "beta_var": beta_vars,
        "n_obs": n_obs
    }

def shrink_beta_estimate(empirical_beta: Union[float, np.ndarray, pd.DataFrame, pd.Series],
                     beta_var: Union[float, np.ndarray, pd.DataFrame, pd.Series],
                     prior: Union[float, np.ndarray, pd.DataFrame, pd.Series],
                     prior_var: Union[float, np.ndarray, pd.DataFrame, pd.Series] = 0.25) -> tuple:
    """
    Shrink empirical beta estimates towards a prior using Empirical Bayes.
    This function explicitly handles Pandas objects for alignment and supports scalar or numpy array inputs.

    Formula:
    weight = beta_var / (beta_var + prior_var)
    shrunk_beta = (1 - weight) * empirical_beta + weight * prior
    shrunk_var = (1 - weight) * beta_var

    Parameters:
    -----------
    empirical_beta : Union[float, np.ndarray, pd.DataFrame, pd.Series]
        Observed beta estimate(s).
    beta_var : Union[float, np.ndarray, pd.DataFrame, pd.Series]
        Variance of the beta estimate(s).
    prior : Union[float, np.ndarray, pd.DataFrame, pd.Series]
        The prior mean (e.g., a sector beta, or a market beta of 1.0).
    prior_var : Union[float, np.ndarray, pd.DataFrame, pd.Series], default 0.25
        The uncertainty/variance of the prior distribution.

    Returns:
    --------
    tuple
        (shrunk_beta, shrunk_var)
        shrunk_beta: The shrunk beta value(s).
        shrunk_var: The variance of the shrunk beta value(s).
    """
    denominator = beta_var + prior_var + 1e-9
    
    # Pandas path
    if isinstance(denominator, (pd.DataFrame, pd.Series)):
        weight = beta_var / denominator
        # We assume standard pandas behavior (NaN propagation) is desired.
        
        shrunk_beta = (1.0 - weight) * empirical_beta + weight * prior
        shrunk_var = (1.0 - weight) * beta_var
        return shrunk_beta, shrunk_var
        
    # Numpy/Scalar path
    if np.isscalar(denominator):
        if denominator <= 0:
            weight = 0.0
        else:
            weight = beta_var / denominator
    else:
        # Array
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(denominator > 0, beta_var / denominator, 0.0)
            
    shrunk_beta = (1.0 - weight) * empirical_beta + weight * prior
    shrunk_var = (1.0 - weight) * beta_var
    return shrunk_beta, shrunk_var

def calculate_sector_prior_betas(returns: pd.DataFrame, 
                                 volumes: pd.DataFrame, 
                                 sector_map: dict, 
                                 beta_window: int = 30*24, 
                                 beta_min_periods: int = 30*24,
                                 prior_avg_window: int = 365*24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate sector prior betas by:
    1. Computing volume-weighted sector returns.
    2. Computing volume-weighted market returns.
    3. Calculating rolling beta of Sector vs Market.
    4. Taking a rolling mean and variance (default 1 year) of the rolling beta to get the prior and its variance.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (index=date, columns=tickers).
    volumes : pd.DataFrame
        Asset volumes (index=date, columns=tickers), typically rolling 30d volume.
    sector_map : dict
        Dictionary mapping Ticker -> Sector Name. 
    beta_window : int, default 90
        Window for the rolling beta calculation.
    beta_min_periods : int, default 30
        Min periods for the rolling beta calculation.
    prior_avg_window : int, default 365
        Window for the rolling mean of the beta (to smooth it into a prior).
        
    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        (priors, prior_vars)
        priors: DataFrame of sector prior betas (index=date, columns=sectors).
        prior_vars: DataFrame of sector prior beta variances (index=date, columns=sectors).
    """
    # 0. Handle Sector Map Format
    # Check if sector_map is {Sector: [Coins]} (from load_sector_groupings) or {Coin: Sector}
    ticker_to_sector = sector_map
    if sector_map:
        first_val = next(iter(sector_map.values()))
        if isinstance(first_val, list):
            # Invert simple {Sector: [Coin, ...]} map to {Coin: Sector}
            ticker_to_sector = {}
            for sec, coins in sector_map.items():
                for c in coins:
                    ticker_to_sector[c] = sec

    # Align returns and volumes
    common_index = returns.index.intersection(volumes.index)
    common_columns = returns.columns.intersection(volumes.columns)
    
    returns = returns.loc[common_index, common_columns]
    volumes = volumes.loc[common_index, common_columns]
    
    # Stack to long format
    ret_long = returns.stack(future_stack=True).rename("ret")
    vol_long = volumes.stack(future_stack=True).rename("vol")
    
    # Combine
    df = pd.concat([ret_long, vol_long], axis=1)
    df.index.names = ["date", "ticker"]
    df = df.reset_index()
    
    # Map sectors
    df["sector"] = df["ticker"].map(ticker_to_sector)
    
    # Drop tickers with no sector
    df = df.dropna(subset=["sector"])
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 1. Sector Returns (Volume Weighted)
    df["sector_vol"] = df.groupby(["date", "sector"])["vol"].transform("sum")
    
    # Avoid division by zero
    df["weight"] = np.where(df["sector_vol"] > 0, df["vol"] / df["sector_vol"], np.nan)
    
    df["w_ret"] = df["weight"] * df["ret"]
    
    sector_ret = df.groupby(["date", "sector"])["w_ret"].sum().unstack("sector")
    
    # 2. Market Returns (Volume Weighted)
    df["mkt_vol"] = df.groupby("date")["vol"].transform("sum")
    
    df["mkt_weight"] = np.where(df["mkt_vol"] > 0, df["vol"] / df["mkt_vol"], np.nan)
    
    df["mkt_w_ret"] = df["mkt_weight"] * df["ret"]
    
    market_ret = df.groupby("date")["mkt_w_ret"].sum()
    
    # 3. Rolling Beta
    # Align sector_ret and market_ret
    rolling_cov = sector_ret.rolling(window=beta_window, min_periods=beta_min_periods).cov(market_ret)
    rolling_mkt_var = market_ret.rolling(window=beta_window, min_periods=beta_min_periods).var()
    
    betas = rolling_cov.div(rolling_mkt_var, axis=0)
    
    # 4. Prior (Rolling Mean and Var of Beta)
        
    if prior_avg_window and prior_avg_window > 1:
        priors = betas.rolling(window=prior_avg_window, min_periods=1).mean()
        prior_vars = betas.rolling(window=prior_avg_window, min_periods=1).var()
    else:
        priors = betas
        # If no window, variance is 0 or undefined, but for compatibility let's return 0s or NaNs
        # or maybe the variance of the beta estimation itself? 
        # But here we are asked for rolling var of the prior.
        prior_vars = pd.DataFrame(0, index=betas.index, columns=betas.columns) # Effectively 0 variance if no window
        
    return priors, prior_vars
    
def calculate_blume_beta(raw_beta: Union[float, pd.DataFrame, pd.Series], 
                            weight_raw: float = 0.67, 
                            weight_market: float = 0.33, 
                            market_beta: float = 1.0) -> Union[float, pd.DataFrame, pd.Series]:
    """
    Calculate Blume-adjusted beta.
    Formula: Adjusted Beta = (0.67 * Raw Beta) + (0.33 * 1.0)
    
    Parameters:
    -----------
    raw_beta : float, pd.DataFrame, or pd.Series
        The raw empirical beta estimates.
    weight_raw : float, default 0.67
        Weight assigned to the raw beta.
    weight_market : float, default 0.33
        Weight assigned to the market beta (mean reversion target).
    market_beta : float, default 1.0
        The target beta to shrink towards (usually 1.0 for the market).
        
    Returns:
    --------
    Adjusted Beta in the same format as input.
    """
    return (raw_beta * weight_raw) + (market_beta * weight_market)