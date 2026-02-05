"""
Betting Against Beta (BAB) Backtest Framework.

Implements the BAB factor strategy:
- Rank coins by forecasted beta
- Long low-beta coins, short high-beta coins
- Long side targets portfolio beta = 1
- Short side targets portfolio beta = -1 (i.e., absolute beta exposure = 1)
- Overall portfolio is beta neutral (1 - 1 = 0)

Reuses core infrastructure from funding_arb_framework.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import cvxpy as cvx

EPS = 1e-8
PERIODS_PER_YEAR = 365  # Daily data

# --- Scoring helpers ---

def compute_sharpe(ret_series: pd.Series, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    std = ret_series.std(ddof=1)
    if std < 1e-10 or not np.isfinite(std):
        return float("nan")
    return float((ret_series.mean() / std) * np.sqrt(periods_per_year))


def compute_sortino_ratio(ret_series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = ret_series - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 100.0 if ret_series.mean() > daily_rf else float("nan")
    downside_std = downside_returns.std(ddof=1)
    if downside_std < 1e-10 or not np.isfinite(downside_std):
        return float("nan")
    return float((ret_series.mean() / downside_std) * np.sqrt(periods_per_year))


def compute_calmar_ratio(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, min_periods: int = 30, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < min_periods:
        return float("nan")
    if equity_series is None:
        equity_series = (1 + ret_series).cumprod()
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    annualized_return = ret_series.mean() * periods_per_year
    if max_drawdown == 0:
        return 100.0 if annualized_return > 0 else float("nan")
    if not np.isfinite(max_drawdown) or not np.isfinite(annualized_return):
        return float("nan")
    return float(annualized_return / max_drawdown)


def compute_composite_score(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, w_sortino: float = 0.4, w_sharpe: float = 0.3, w_calmar: float = 0.3, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    sortino = compute_sortino_ratio(ret_series, periods_per_year=periods_per_year)
    sharpe = compute_sharpe(ret_series, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(ret_series, equity_series, periods_per_year=periods_per_year)
    if np.isnan(sortino) or np.isnan(sharpe) or np.isnan(calmar):
        return float("nan")
    total_weight = w_sortino + w_sharpe + w_calmar
    w_sortino /= total_weight
    w_sharpe /= total_weight
    w_calmar /= total_weight
    return float(w_sortino * sortino + w_sharpe * sharpe + w_calmar * calmar)


def select_score(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, mode: str = "composite", periods_per_year: float = PERIODS_PER_YEAR) -> float:
    mode = (mode or "composite").lower()
    sharpe = compute_sharpe(ret_series, periods_per_year=periods_per_year)
    sortino = compute_sortino_ratio(ret_series, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(ret_series, equity_series, periods_per_year=periods_per_year)
    if mode == "sharpe":
        return sharpe
    if mode == "sortino":
        return sortino
    if mode == "calmar":
        return calmar
    if mode == "composite":
        return compute_composite_score(
            ret_series,
            equity_series,
            periods_per_year=periods_per_year,
        )
    raise ValueError(f"Unsupported score_mode: {mode}")


def compute_ic(forecasted_values: pd.Series, actual_values: pd.Series) -> float:
    """
    Computes the Information Coefficient (IC) between forecasted and actual values.
    IC is the Spearman's rank correlation coefficient between the forecast and outcome.
    """
    if forecasted_values.empty or actual_values.empty or len(forecasted_values) != len(actual_values):
        return float("nan")
    
    # Drop NaNs from both series for accurate correlation
    combined = pd.DataFrame({'forecast': forecasted_values, 'actual': actual_values}).dropna()
    
    if len(combined) < 2: # Need at least 2 points for correlation
        return float("nan")
        
    return float(combined['forecast'].corr(combined['actual'], method='spearman'))


def compute_ir(
    portfolio_returns: pd.Series, 
    periods_per_year: float = PERIODS_PER_YEAR
) -> float:
    """
    Computes the Information Ratio (IR) of portfolio returns.
    Without a benchmark, this calculates Return / Volatility (similar to Sharpe Ratio with Rf=0).
    """
    if portfolio_returns.empty:
        return float("nan")

    # Drop NaNs
    returns = portfolio_returns.dropna()

    if len(returns) < 2:
        return float("nan")

    if returns.std() < EPS: # Volatility is near zero
        return float("inf") if returns.mean() > 0 else float("nan")

    annualized_return = returns.mean() * periods_per_year
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    
    if annualized_volatility < 1e-10 or not np.isfinite(annualized_volatility):
        return float("nan")

    return float(annualized_return / annualized_volatility)


def compute_rolling_ir(
    portfolio_returns: pd.Series, 
    window: int = 30,
    periods_per_year: float = PERIODS_PER_YEAR
) -> pd.Series:
    """
    Computes rolling Information Ratio (IR) over a specified window.
    Without a benchmark, this calculates Rolling Return / Rolling Volatility.
    """
    if portfolio_returns.empty:
        return pd.Series(dtype=float)

    # Drop NaNs
    returns = portfolio_returns.dropna()
    
    if len(returns) < window:
        return pd.Series(dtype=float)

    # Rolling Mean * Annualize (approximate for rolling window)
    rolling_mean = returns.rolling(window).mean() * periods_per_year
    # Rolling Std * Sqrt(Annualize)
    rolling_std = returns.rolling(window).std() * np.sqrt(periods_per_year)
    
    return rolling_mean / rolling_std


def compute_daily_ic(
    detailed_df: pd.DataFrame,
    date_col: str = 'date',
    forecast_col: str = 'beta',
    target_col: str = 'actual_return_total',
    method: str = 'spearman'
) -> pd.Series:
    """
    Computes the Cross-Sectional Information Coefficient (IC) for each date.
    Returns a Series of IC values indexed by date.
    """
    if detailed_df.empty or date_col not in detailed_df.columns:
        return pd.Series(dtype=float)

    # Helper to calculate corr for a group
    def _calc_corr(group):
        if len(group) < 2:
            return np.nan
        # Ensure numeric
        f = pd.to_numeric(group[forecast_col], errors='coerce')
        t = pd.to_numeric(group[target_col], errors='coerce')
        
        valid = pd.DataFrame({'f': f, 't': t}).dropna()
        if len(valid) < 2:
            return np.nan
        return valid['f'].corr(valid['t'], method=method)

    # Group by date
    return detailed_df.groupby(date_col).apply(_calc_corr, include_groups=False).sort_index()


def compute_icir(ic_series: pd.Series) -> float:
    """
    Computes the Information Coefficient Information Ratio (ICIR).
    ICIR = Mean(IC) / Std(IC)
    """
    if ic_series.empty or len(ic_series) < 2:
        return float("nan")
        
    mean_ic = ic_series.mean()
    std_ic = ic_series.std(ddof=1)
    
    if std_ic < 1e-10 or not np.isfinite(std_ic):
        return float("nan")
        
    return float(mean_ic / std_ic)

def compute_rolling_icir(
    ic_series: pd.Series, 
    window: int = 30
) -> pd.Series:
    """
    Computes rolling Information Coefficient Information Ratio (ICIR) over a specified window.
    ICIR = Rolling Mean(IC) / Rolling Std(IC)
    """
    if ic_series.empty or len(ic_series) < window:
        return pd.Series(dtype=float)
        
    rolling_mean = ic_series.rolling(window).mean()
    rolling_std = ic_series.rolling(window).std()
    
    return rolling_mean / rolling_std

# --- Data Structures ---

class FundingDataBundle:
    """
    Container for price, funding, and return data.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None,
        volume_df: Optional[pd.DataFrame] = None,
        btc_ret: Optional[pd.Series] = None,
        eth_ret: Optional[pd.Series] = None,
        min_hist_days: int = 30,
        beta_window: int = 30,
        # HF Data Support
        returns_df_hf: Optional[pd.DataFrame] = None,
        btc_ret_hf: Optional[pd.Series] = None,
        eth_ret_hf: Optional[pd.Series] = None,
        hf_window_multiplier: int = 1,  # e.g. 24 if hf is hourly and base is daily
        hf_resample_rule: Optional[str] = None # e.g. 'D' to resample HF betas to Daily
    ):
        # Align columns (tickers) across all dataframes to ensure consistent shapes
        # We need the intersection of tickers present in Price AND Funding data
        common_tickers = price_df.columns.intersection(funding_df.columns)
        if returns_df is not None:
            common_tickers = common_tickers.intersection(returns_df.columns)
        if volume_df is not None:
            common_tickers = common_tickers.intersection(volume_df.columns)
        
        # Sort tickers to ensure consistent order
        self.tickers = sorted(list(common_tickers))
        
        self.price_df = price_df[self.tickers]
        self.funding_df = funding_df[self.tickers]
        self.volume_df = volume_df[self.tickers] if volume_df is not None else None
        
        if returns_df is not None:
            self.returns_df = returns_df[self.tickers]
        else:
            self.returns_df = self.price_df.pct_change(fill_method=None)
        
        self.btc_ret = btc_ret
        self.eth_ret = eth_ret
        self.min_hist_days = min_hist_days
        self.beta_window = beta_window

        # HF Data
        self.returns_df_hf = returns_df_hf
        self.btc_ret_hf = btc_ret_hf
        self.eth_ret_hf = eth_ret_hf
        self.hf_window_multiplier = hf_window_multiplier
        self.hf_resample_rule = hf_resample_rule

        self.dates = self.price_df.index.to_numpy()
        
        # Store betas by window: {window: {'btc': df, 'eth': df, ...}}
        self.betas: Dict[int, Dict[str, pd.DataFrame]] = {}
        # Store beta variances by window: {window: {'btc': df, 'eth': df, ...}}
        self.beta_vars: Dict[int, Dict[str, pd.DataFrame]] = {}

        # Store predicted funding by AR window: {ar_window: pd.DataFrame}
        self.predicted_funding: Dict[int, pd.DataFrame] = {}

        # Pre-extract market returns if not provided
        if self.btc_ret is None and returns_df is not None and 'BTCUSDT' in returns_df.columns:
             self.btc_ret = returns_df['BTCUSDT']
        elif self.btc_ret is None and 'BTCUSDT' in self.returns_df.columns:
             self.btc_ret = self.returns_df['BTCUSDT']

        if self.eth_ret is None and returns_df is not None and 'ETHUSDT' in returns_df.columns:
             self.eth_ret = returns_df['ETHUSDT']
        elif self.eth_ret is None and 'ETHUSDT' in self.returns_df.columns:
             self.eth_ret = self.returns_df['ETHUSDT']
             
        # Pre-extract HF market returns if not provided
        if self.returns_df_hf is not None:
            if self.btc_ret_hf is None and 'BTCUSDT' in self.returns_df_hf.columns:
                self.btc_ret_hf = self.returns_df_hf['BTCUSDT']
            if self.eth_ret_hf is None and 'ETHUSDT' in self.returns_df_hf.columns:
                self.eth_ret_hf = self.returns_df_hf['ETHUSDT']
            
            # Align HF data to the selected universe (columns) to avoid shape mismatch
            self.returns_df_hf = self.returns_df_hf[self.tickers]

    def ensure_beta_matrix(self, windows: List[int]):
        """
        Precompute rolling betas against BTC, ETH, and Combined (BTC+ETH)/2 for specific windows.
        Also computes 'adaptive' beta (best fit between BTC and ETH).
        """
        if self.btc_ret is None or self.eth_ret is None:
            # Fallback check for HF
            if self.returns_df_hf is None or self.btc_ret_hf is None or self.eth_ret_hf is None:
                warnings.warn("BTC or ETH returns not found (neither daily nor HF). Betas cannot be computed.")
                return

        for w in windows:
            if w in self.betas:
                continue

            print(f"Precomputing rolling betas (window={w})...")
            
            # Decide whether to use HF or Daily
            use_hf = (self.returns_df_hf is not None) and \
                     (self.btc_ret_hf is not None) and \
                     (self.eth_ret_hf is not None)

            if use_hf:
                print(f"  -> Using High-Frequency Data (multiplier={self.hf_window_multiplier}, resample={self.hf_resample_rule})")
                
                eff_window = w * self.hf_window_multiplier
                combined_rets_hf = (self.btc_ret_hf + self.eth_ret_hf) / 2
                
                def compute_beta(market_ret, data_ret):
                    # Beta
                    rolling_cov = data_ret.rolling(eff_window).cov(market_ret)
                    rolling_mkt_var = market_ret.rolling(eff_window).var()
                    beta = rolling_cov.div(rolling_mkt_var, axis=0)
                    
                    # Beta Variance
                    # Var(beta) = [Var(coin) / Var(mkt)] * [(1 - corr^2) / (N - 2)]
                    rolling_corr = data_ret.rolling(eff_window).corr(market_ret)
                    rolling_coin_var = data_ret.rolling(eff_window).var()
                    n_obs = data_ret.rolling(eff_window).count()
                    
                    var_ratio = rolling_coin_var.div(rolling_mkt_var, axis=0)
                    unexplained_corr = 1 - rolling_corr**2
                    dof = n_obs - 2
                    
                    beta_var = var_ratio * unexplained_corr / dof
                    beta_var[dof <= 0] = np.nan
                    
                    # Resample/Align
                    if self.hf_resample_rule:
                        beta = beta.resample(self.hf_resample_rule).last()
                        beta_var = beta_var.resample(self.hf_resample_rule).last()
                        rolling_corr = rolling_corr.resample(self.hf_resample_rule).last()
                    
                    # Ensure alignment with base price index
                    beta = beta.reindex(self.price_df.index, method='ffill')
                    beta_var = beta_var.reindex(self.price_df.index, method='ffill')
                    rolling_corr = rolling_corr.reindex(self.price_df.index, method='ffill')
                    
                    return beta, beta_var, rolling_corr

                beta_btc, var_btc, corr_btc = compute_beta(self.btc_ret_hf, self.returns_df_hf)
                beta_eth, var_eth, corr_eth = compute_beta(self.eth_ret_hf, self.returns_df_hf)
                beta_combined, var_combined, _ = compute_beta(combined_rets_hf, self.returns_df_hf)
                
            else:
                combined_rets = (self.btc_ret + self.eth_ret) / 2
                
                def compute_beta(market_ret):
                    # Beta
                    rolling_cov = self.returns_df.rolling(w).cov(market_ret)
                    rolling_mkt_var = market_ret.rolling(w).var()
                    beta = rolling_cov.div(rolling_mkt_var, axis=0)
                    
                    # Beta Variance
                    rolling_corr = self.returns_df.rolling(w).corr(market_ret)
                    rolling_coin_var = self.returns_df.rolling(w).var()
                    n_obs = self.returns_df.rolling(w).count()
                    
                    var_ratio = rolling_coin_var.div(rolling_mkt_var, axis=0)
                    unexplained_corr = 1 - rolling_corr**2
                    dof = n_obs - 2
                    
                    beta_var = var_ratio * unexplained_corr / dof
                    beta_var[dof <= 0] = np.nan
                    
                    return beta, beta_var, rolling_corr

                beta_btc, var_btc, corr_btc = compute_beta(self.btc_ret)
                beta_eth, var_eth, corr_eth = compute_beta(self.eth_ret)
                beta_combined, var_combined, _ = compute_beta(combined_rets)

            # Compute Adaptive Beta (Max Correlation)
            # Use abs correlation to determine best fit (Highest R^2)
            corr_btc_abs = corr_btc.abs().fillna(-1.0)
            corr_eth_abs = corr_eth.abs().fillna(-1.0)
            
            mask_btc_better = corr_btc_abs >= corr_eth_abs
            
            beta_adaptive = beta_btc.where(mask_btc_better, beta_eth)
            var_adaptive = var_btc.where(mask_btc_better, var_eth)

            self.betas[w] = {
                'btc': beta_btc,
                'eth': beta_eth,
                'combined': beta_combined,
                'adaptive': beta_adaptive
            }
            
            self.beta_vars[w] = {
                'btc': var_btc,
                'eth': var_eth,
                'combined': var_combined,
                'adaptive': var_adaptive
            }

            # Match original script logic: Explicitly exclude BTC and ETH from beta-neutral trading
            # by setting their betas to NaN in ALL beta types.
            for key in self.betas[w]:
                for ticker in ['BTCUSDT', 'ETHUSDT']:
                    if ticker in self.betas[w][key].columns:
                        self.betas[w][key][ticker] = np.nan
                        self.beta_vars[w][key][ticker] = np.nan


# --- Strategy Interface ---

class Strategy:
    def prepare(self, bundle: FundingDataBundle) -> None:
        pass

    def signals(self, idx: int, bundle: FundingDataBundle) -> Dict[str, Any]:
        raise NotImplementedError

# --- Weighting Model ---

class WeightingModel:
    def weights(
        self,
        idx: int,
        signals: Dict[str, Any],
        bundle: FundingDataBundle,
        universe_mask: np.ndarray,
        params: Any,
    ) -> np.ndarray:
        raise NotImplementedError

# --- BAB Parameters ---

@dataclass
class BettingAgainstBetaParams:
    """Parameters for Betting Against Beta strategy."""
    portfolio_size_each_side: int = 5   # Number of assets on each side (long/short)
    target_side_beta: float = 1.0       # Target absolute beta for each side (pre-scaling)
    beta_tolerance: float = 0.1         # Tolerance for beta constraint 
    gross_exposure_limit: float = 1.0   # Standard leverage of 1.0 (will rescale weights down to this)
    tc_bps: float = 5.0                 # Transaction cost in basis points
    beta_type: str = "combined"         # "btc", "eth", "combined", or "adaptive"
    beta_window: int = 30               # Lookback for beta calculation
    use_shrinkage: bool = False         # Whether to use beta shrinkage
    prior_beta_window: int = 60         # Lookback for prior beta (shrinkage target)
    min_weight: float = 0.0             # Minimum weight per position (0 = no constraint)
    max_weight: float = 1.0             # Max weight allowed in optimization (loose, to allow shape discovery)
    weighting_method: str = "rank_optimized"  # "rank_optimized" or "frazzini_pedersen"
    max_funding_short: float = 0.0005   # Max funding allowed for shorts (5bps) - avoid costly shorts
    min_funding_long: float = -0.0005   # Min funding allowed for longs (-5bps) - avoid costly longs
    volatility_scaling: bool = False    # Whether to scale weights by inverse volatility
    leverage_cap: float = 5.0           # Max Leverage for Frazzini method (safety brake)
    volume_filter_threshold: float = 0.0 # Top N percentile volume filter (e.g. 0.8 to keep top 80%)
    optimization_objective: str = "diversification" # "diversification" (min sum sq) or "min_variance" (min risk)
    covariance_window: int = 90         # Lookback window for covariance estimation (if min_variance used)

class BettingAgainstBetaStrategy(Strategy):
    """
    Betting Against Beta strategy.
    
    Signals are based on forecasted betas - we use the current rolling beta
    as the forecast for next period's beta (persistence assumption).
    """
    
    def __init__(self, params: BettingAgainstBetaParams):
        self.params = params
    
    def prepare(self, bundle: FundingDataBundle) -> None:
        """
        Ensure betas are precomputed for the required windows.
        """
        windows = [self.params.beta_window]
        if self.params.use_shrinkage:
            windows.append(self.params.prior_beta_window)
        
        bundle.ensure_beta_matrix(windows)
        
        # Precompute Volatility if scaling is enabled
        # if getattr(self.params, "volatility_scaling", False):
        #     self._ensure_volatility(bundle, self.params.beta_window)

    def _ensure_volatility(self, bundle: FundingDataBundle, window: int) -> None:
        """Ensure rolling volatility is computed."""
        if not hasattr(bundle, "volatilities"):
            bundle.volatilities = {}
            
        if window not in bundle.volatilities:
            print(f"Precomputing rolling volatility (window={window})...")
            
            # Use HF data if available (better estimate)
            if bundle.returns_df_hf is not None:
                # Calculate Hourly Volatility then scale to Daily
                # Vol_Daily = Vol_Hourly * sqrt(24)
                # Note: We need the rolling std over 'window' days.
                # Window in hours = window * multiplier
                hf_window = window * bundle.hf_window_multiplier
                
                # Rolling STD on HF Returns
                vol_hf = bundle.returns_df_hf.rolling(window=hf_window, min_periods=hf_window//2).std()
                
                # Scale to Daily Volatility
                # If returns are hourly, annualizing involves * sqrt(24*365).
                # Here we just want "Daily Vol", so * sqrt(24).
                daily_vol_est = vol_hf * np.sqrt(bundle.hf_window_multiplier)
                
                # Resample to Daily (take last value of the day to align with decision time)
                # shift(1) done in signals logic? No, rolling includes current.
                # We align by resample.
                daily_vol_df = daily_vol_est.resample(bundle.hf_resample_rule or 'D').last()
                
            else:
                # Use Daily Data
                daily_vol_df = bundle.returns_df.rolling(window=window, min_periods=window//2).std()
                
            bundle.volatilities[window] = daily_vol_df
    
    def signals(self, idx: int, bundle: FundingDataBundle) -> Dict[str, Any]:
        """
        Returns the forecasted beta for each asset.
        """
        date = bundle.dates[idx]
        beta_type = self.params.beta_type.lower()
        window = self.params.beta_window
        
        # Get Betas
        if window not in bundle.betas:
            beta = np.full(len(bundle.tickers), np.nan)
            beta_var = np.full(len(bundle.tickers), np.nan)
        else:
            betas_for_window = bundle.betas[window]
            
            if beta_type not in betas_for_window:
                warnings.warn(f"Beta type '{beta_type}' not found for window {window}. Using 'combined'.")
                beta_type = "combined" if "combined" in betas_for_window else list(betas_for_window.keys())[0]
            
            if date in betas_for_window[beta_type].index:
                raw_beta = betas_for_window[beta_type].loc[date].to_numpy()
                
                # Retrieve Beta Variance for Shrinkage AND Volatility Scaling
                if window in bundle.beta_vars and beta_type in bundle.beta_vars[window]:
                     if date in bundle.beta_vars[window][beta_type].index:
                         beta_var = bundle.beta_vars[window][beta_type].loc[date].to_numpy()
                     else:
                         beta_var = np.full(len(bundle.tickers), np.nan)
                else:
                    beta_var = np.full(len(bundle.tickers), np.nan)

                # Apply Shrinkage if enabled
                if self.params.use_shrinkage:
                    from beta_shrink_func import shrink_beta_estimate
                    prior_window = self.params.prior_beta_window
                    
                    if prior_window in bundle.betas and prior_window in bundle.beta_vars:
                         # beta_var already retrieved above
                        
                        priors_for_window = bundle.betas[prior_window]
                        prior_vars_for_window = bundle.beta_vars[prior_window]
                        
                        if beta_type in priors_for_window and beta_type in prior_vars_for_window:
                            if date in priors_for_window[beta_type].index:
                                prior_beta = priors_for_window[beta_type].loc[date].to_numpy()
                                prior_var = prior_vars_for_window[beta_type].loc[date].to_numpy()
                                
                                shrunk_beta, _ = shrink_beta_estimate(raw_beta, beta_var, prior_beta, prior_var)
                                beta = shrunk_beta
                            else:
                                beta = raw_beta
                        else:
                            beta = raw_beta
                    else:
                        beta = raw_beta
                else:
                    beta = raw_beta
            else:
                beta = np.full(len(bundle.tickers), np.nan)
                beta_var = np.full(len(bundle.tickers), np.nan)
        
        # Get Volatility (if needed for scaling)
        vol = np.full(len(bundle.tickers), np.nan)
        if getattr(self.params, "volatility_scaling", False):
            # Use Beta Variance (Standard Deviation of Beta Estimate)
            vol = np.sqrt(beta_var)

        return {
            "beta": beta,
            "volatility": vol,
            "date": date
        }



# --- BAB Weighting Model ---


class BettingAgainstBetaWeighting(WeightingModel):
    """
    Betting Against Beta Weighting.
    
    Constructs a portfolio where:
    - Long side: low-beta assets with portfolio beta = target_side_beta (default 1)
    - Short side: high-beta assets with portfolio beta = -target_side_beta (default -1)
    - Overall: beta neutral (target_side_beta - target_side_beta = 0)
    
    Optimization:
    - Select top K lowest-beta assets for long, top K highest-beta for short
    - Weight each side so that sum(w_i * beta_i) = target for that side
    """
    
    def weights(
        self,
        idx: int,
        signals: Dict[str, Any],
        bundle: FundingDataBundle,
        universe_mask: np.ndarray,
        params: BettingAgainstBetaParams,
    ) -> np.ndarray:
        
        beta = signals["beta"]
        date = signals["date"]
        volatility = signals.get("volatility", None) # Retrieve volatility if available
        
        # 1. Basic Universe Filter
        # Require valid beta and positive beta (standard BAB assumes positive betas)
        mask = universe_mask & np.isfinite(beta) & (beta > 0)
        
        # 2. Funding / Cost Filter (Important for Crypto)
        # Avoid Longing if Funding is very High (Costly)
        # Avoid Shorting if Funding is very Negative (Costly)
        
        funding_vals = None
        if date in bundle.funding_df.index:
            funding_vals = bundle.funding_df.loc[date].to_numpy()
            funding_vals = np.nan_to_num(funding_vals, 0.0)
            
        # Cost thresholds (default 5bps per period if not specified properly)
        cost_tol = 0.0005
        if hasattr(params, "max_funding_short"): 
            cost_tol = abs(params.max_funding_short)
        
        tol_long = 0.0005
        if hasattr(params, "min_funding_long"): 
            tol_long = abs(params.min_funding_long)
            
        # Masks for eligibility
        # Long Eligible: Funding < Cost Tolerance (e.g. < 5bps)
        # Short Eligible: Funding > -Cost Tolerance (e.g. > -5bps)
        
        if funding_vals is not None:
            long_eligible = mask & (funding_vals < tol_long)
            short_eligible = mask & (funding_vals > -cost_tol)
        else:
            long_eligible = mask
            short_eligible = mask
            
        # Dispatch weighting method
        method = getattr(params, "weighting_method", "rank_optimized")
        
        if method == "frazzini_pedersen":
            return self._weights_frazzini_pedersen(beta, long_eligible, short_eligible, params, bundle, volatility)
        else:
            return self._weights_rank_optimized(beta, long_eligible, short_eligible, params, bundle, idx)

    def _weights_frazzini_pedersen(
        self, 
        beta: np.ndarray, 
        long_mask: np.ndarray, 
        short_mask: np.ndarray, 
        params: BettingAgainstBetaParams, 
        bundle: FundingDataBundle,
        volatility: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rank-based weighting (Frazzini & Pedersen style) with optional Volatility Scaling.
        """
        # Rank considered over the union of eligible assets to establish "Market Ranks"
        union_mask = long_mask | short_mask
        if np.sum(union_mask) < 2:
            return np.zeros(len(beta))
        
        # Calculate Ranks (z_i) for all assets
        # We rank all assets that are vaguely valid to get proper distribution
        # Then we zero out the ones we can't trade on specific sides
        
        beta_subset = beta[union_mask]
        indices = np.where(union_mask)[0]
        
        # Rank betas from 1 to N
        ranks = np.argsort(np.argsort(beta_subset)) + 1
        z_bar = np.mean(ranks)
        
        # Raw Weights: z_bar - z_i 
        # Low Beta (Low Rank) -> Positive Weight (Long Candidate)
        # High Beta (High Rank) -> Negative Weight (Short Candidate)
        
        raw_w = z_bar - ranks
        
        # Map back to full size
        w_long_raw = np.zeros(len(beta))
        w_short_raw = np.zeros(len(beta))
        
        # Assign raw weights to indices
        # w_long_raw takes positive values (Low Beta)
        # w_short_raw takes negative values (High Beta) -> convert to absolute
        
        # LONG SIDE
        # Condition: Raw weight > 0 AND Long Eligible
        is_positive = raw_w > 0
        long_cand_indices = indices[is_positive]
        long_cand_weights = raw_w[is_positive]
        
        # Apply Long Eligibility Mask
        # We need to map subset indices back to global to check mask
        valid_longs = []
        valid_long_w = []
        
        for idx, w in zip(long_cand_indices, long_cand_weights):
            if long_mask[idx]:
                valid_longs.append(idx)
                valid_long_w.append(w)
        
        # Restrict to Top K (Largest weights -> Lowest Betas)
        K = params.portfolio_size_each_side
        if K > 0 and len(valid_longs) > K:
            # Sort by weight descending (Largest absolute weight = Most Extreme Rank)
            combined = sorted(zip(valid_long_w, valid_longs), key=lambda x: x[0], reverse=True)
            combined = combined[:K]
            valid_long_w = [x[0] for x in combined]
            valid_longs = [x[1] for x in combined]

        # SHORT SIDE
        # Condition: Raw weight < 0 AND Short Eligible
        is_negative = raw_w < 0
        short_cand_indices = indices[is_negative]
        short_cand_weights = -raw_w[is_negative] # Make positive
        
        valid_shorts = []
        valid_short_w = []
        
        for idx, w in zip(short_cand_indices, short_cand_weights):
            if short_mask[idx]:
                valid_shorts.append(idx)
                valid_short_w.append(w)

        # Restrict to Top K (Largest weights -> Highest Betas)
        if K > 0 and len(valid_shorts) > K:
            # Sort by weight descending
            combined = sorted(zip(valid_short_w, valid_shorts), key=lambda x: x[0], reverse=True)
            combined = combined[:K]
            valid_short_w = [x[0] for x in combined]
            valid_shorts = [x[1] for x in combined]

        # OPTIONAL: Volatility Scaling
        # Divide weights by volatility (inverse vol weighting)
        # w_i_new = w_i / (sigma_i)^2 or sigma_i. Frazzini uses 1/sigma^2 often, but 1/sigma is more robust.
        
        use_vol_scaling = getattr(params, "volatility_scaling", False)
        
        if use_vol_scaling and volatility is not None:
             # Scale Longs
             if valid_longs:
                v_idxs_l = np.array(valid_longs)
                vols_l = volatility[v_idxs_l]
                # Avoid division by zero
                vols_l[vols_l < 1e-6] = np.mean(vols_l) 
                
                # Apply inverse vol scaling
                # Raw weights are rank-based. We tilt them by stability.
                # W_new = W_rank / Var(beta) = W_rank / (Vol^2)
                ws_l = np.array(valid_long_w)
                ws_l_scaled = ws_l / (vols_l ** 2)
                valid_long_w = list(ws_l_scaled)

             # Scale Shorts
             if valid_shorts:
                v_idxs_s = np.array(valid_shorts)
                vols_s = volatility[v_idxs_s]
                vols_s[vols_s < 1e-6] = np.mean(vols_s) 
                
                ws_s = np.array(valid_short_w)
                ws_s_scaled = ws_s / (vols_s ** 2)
                valid_short_w = list(ws_s_scaled)
                
        # SCALING
        final_weights = np.zeros(len(beta))
        
        # Scale Long
        if valid_longs:
            v_idxs = np.array(valid_longs)
            v_ws = np.array(valid_long_w)
            v_betas = beta[v_idxs]
            
            # We want sum(w * beta) = Target
            # w_final = k * w_raw
            # k * sum(w_raw * beta) = Target
            numerator = params.target_side_beta
            denominator = np.sum(v_ws * v_betas)
            
            if denominator > 1e-6:
                k_long = numerator / denominator
                final_weights[v_idxs] = v_ws * k_long
        
        # Scale Short
        if valid_shorts:
            v_idxs = np.array(valid_shorts)
            v_ws = np.array(valid_short_w)
            v_betas = beta[v_idxs]
            
            numerator = params.target_side_beta
            denominator = np.sum(v_ws * v_betas)
            
            if denominator > 1e-6:
                k_short = numerator / denominator
                final_weights[v_idxs] = -v_ws * k_short # Negative for short
        
        # Gross Exposure Cap
        # If Frazzini (Rank Based), we use leverage_cap (default 5.0) to allow Beta=1 target to be met.
        # If Rank Optimized (Solver), we use gross_exposure_limit (default 1.0) because solver is constrained by it.
        
        gross_exp = np.sum(np.abs(final_weights))
        
        # Determine strictness of cap
        limit = params.gross_exposure_limit if getattr(params, "weighting_method", "") != "frazzini_pedersen" else getattr(params, "leverage_cap", 5.0)
        
        if gross_exp > limit:
            final_weights *= (limit / gross_exp)
            
        return final_weights

    def _weights_rank_optimized(
        self,
        beta: np.ndarray,
        long_mask: np.ndarray,
        short_mask: np.ndarray,
        params: BettingAgainstBetaParams,
        bundle: FundingDataBundle,
        idx: int = -1,
    ) -> np.ndarray:
        
        # Identify available assets
        # For rank-optimized (Top-K), we usually start with strict ranking
        # We need to filter indices first.
        
        # Strategy:
        # 1. Take top K lowest beta that are Long Eligible
        # 2. Take top K highest beta that are Short Eligible
        
        # Sort ALL valid betas (using union mask to find candidates, then checking speciic mask)
        union_mask = long_mask | short_mask
        available_indices = np.nonzero(union_mask)[0]
        
        k = params.portfolio_size_each_side
        
        if len(available_indices) < 2:
             return np.zeros(len(beta))

        beta_sub = beta[available_indices]
        sorted_args = np.argsort(beta_sub) # Ascending
        
        # Walk from left (Low Beta) to find K long candidates
        long_indices = []
        for loc in sorted_args:
            idx_map = available_indices[loc]
            if len(long_indices) < k and long_mask[idx_map]:
                long_indices.append(idx_map)
        
        # Walk from right (High Beta) to find K short candidates
        short_indices = []
        for loc in sorted_args[::-1]:
            idx_map = available_indices[loc]
            if len(short_indices) < k and short_mask[idx_map]:
                short_indices.append(idx_map)
        
        long_indices = np.array(long_indices)
        short_indices = np.array(short_indices)
        
        if len(long_indices) == 0 or len(short_indices) == 0:
            return np.zeros(len(beta))

        # --- OPTIMIZATION ---
        beta_long = beta[long_indices]
        beta_short = beta[short_indices]
        
        full_weights = np.zeros(len(beta))
        target_beta = params.target_side_beta
        tol = params.beta_tolerance
        
        # Covariance for Min Variance
        cov_long_mat = None
        cov_short_mat = None
        
        is_min_var = getattr(params, "optimization_objective", "diversification") == "min_variance"
        
        if is_min_var and idx > 0:
            from beta_shrink_func import shrink_covariance_bayes
            
            # Windows
            win_l = getattr(params, "covariance_window", 90)
            win_s = max(10, win_l // 3) # Short window approx 1/3 of long
            
            # Start indices
            start_l = max(0, idx - win_l)
            start_s = max(0, idx - win_s)
            
            # Slices
            hist_l = bundle.returns_df.iloc[start_l:idx]
            hist_s = bundle.returns_df.iloc[start_s:idx]
            
            if not hist_l.empty and not hist_s.empty:
                # --- LONG LEG ---
                rets_l_long = hist_l.iloc[:, long_indices]
                rets_l_short = hist_s.iloc[:, long_indices]
                
                # Raw Covariances
                c_l_long = rets_l_long.cov().to_numpy()
                c_l_short = rets_l_short.cov().to_numpy()
                
                # Handle NaNs and regularization
                c_l_long = np.nan_to_num(c_l_long, 0.0)
                c_l_short = np.nan_to_num(c_l_short, 0.0)
                
                # Shrink
                n_l = len(hist_l)
                n_s = len(hist_s)
                cov_long_mat = shrink_covariance_bayes(c_l_short, c_l_long, n_s, n_l)
                
                # Add slight regularization for solver stability
                cov_long_mat += np.eye(len(long_indices)) * 1e-6
                
                # --- SHORT LEG ---
                rets_s_long = hist_l.iloc[:, short_indices]
                rets_s_short = hist_s.iloc[:, short_indices]
                
                c_s_long = rets_s_long.cov().to_numpy()
                c_s_short = rets_s_short.cov().to_numpy()
                
                c_s_long = np.nan_to_num(c_s_long, 0.0)
                c_s_short = np.nan_to_num(c_s_short, 0.0)
                
                cov_short_mat = shrink_covariance_bayes(c_s_short, c_s_long, n_s, n_l)
                cov_short_mat += np.eye(len(short_indices)) * 1e-6

        # Check feasibility (approx)
        # We want to solve, so let solver decide feasibility usually, but simple checks help speed
        
        lev_cap = getattr(params, "leverage_cap", 5.0)

        try:
            # LONG
            n_long = len(long_indices)
            w_long = cvx.Variable(n_long, nonneg=True)
            
            if cov_long_mat is not None:
                obj_l = cvx.Minimize(cvx.quad_form(w_long, cov_long_mat))
            else:
                obj_l = cvx.Minimize(cvx.sum_squares(w_long))
            
            # Constraints
            # 1. Target Beta: sum(w * beta) = 1 (approx)
            # 2. Max Weight per asset
            # 3. Min Weight per asset
            # 4. Leverage Cap: sum(w) <= Cap
            
            # Relax beta constraint slightly if needed, but BAB usually targets it strictly
            constr_l = [
                beta_long @ w_long >= target_beta - tol,
                beta_long @ w_long <= target_beta + tol,
                cvx.sum(w_long) <= lev_cap,
            ]
            
            if params.max_weight <= 1.0:
                 constr_l.append(w_long <= params.max_weight)
                 
            if params.min_weight > 0: 
                constr_l.append(w_long >= params.min_weight)
            
            cvx.Problem(obj_l, constr_l).solve(solver=cvx.CLARABEL, verbose=False)
            
            if w_long.value is None: 
                # Fallback to pure equal weight or skip? Skip for safety.
                return np.zeros(len(beta))
            weights_long = w_long.value
            
            # SHORT
            n_short = len(short_indices)
            w_short = cvx.Variable(n_short, nonneg=True)
            
            if cov_short_mat is not None:
                obj_s = cvx.Minimize(cvx.quad_form(w_short, cov_short_mat))
            else:
                obj_s = cvx.Minimize(cvx.sum_squares(w_short))
                
            constr_s = [
                beta_short @ w_short >= target_beta - tol,
                beta_short @ w_short <= target_beta + tol,
                cvx.sum(w_short) <= lev_cap,
            ]
            
            if params.max_weight <= 1.0:
                constr_s.append(w_short <= params.max_weight)
                
            if params.min_weight > 0: 
                constr_s.append(w_short >= params.min_weight)
            
            cvx.Problem(obj_s, constr_s).solve(solver=cvx.CLARABEL, verbose=False)
            
            if w_short.value is None: return np.zeros(len(beta))
            weights_short = -w_short.value
            
            full_weights[long_indices] = weights_long
            full_weights[short_indices] = weights_short
            
            # We do NOT rescale by gross_exposure_limit here if we want to respect the Beta Target.
            # The leverage_cap inside solver ensures we are within safety limits.
            
            full_weights[np.abs(full_weights) < EPS] = 0.0
            return full_weights
            
        except Exception:
            return np.zeros(len(beta))



# --- BAB Backtest Engine ---

class BABBacktestEngine:
    """
    Backtest engine for Betting Against Beta strategy.
    
    Total Return = Price Return + Funding Return (funding is still applicable for perps)
    """
    
    def __init__(
        self,
        bundle: FundingDataBundle,
        strategy: Strategy,
        weighting: WeightingModel,
        params: BettingAgainstBetaParams,
    ):
        self.bundle = bundle
        self.strategy = strategy
        self.weighting = weighting
        self.params = params
        
        self.strategy.prepare(bundle)
    
    def _universe_mask(self, idx: int, next_date: pd.Timestamp) -> np.ndarray:
        """Create mask for tradeable universe."""
        current_date = self.bundle.price_df.index[idx]
        p_curr = self.bundle.price_df.iloc[idx].to_numpy()
        
        if next_date not in self.bundle.price_df.index:
            return np.zeros(len(self.bundle.tickers), dtype=bool)
        
        p_next = self.bundle.price_df.loc[next_date].to_numpy()
        has_price = np.isfinite(p_curr) & np.isfinite(p_next)
        
        # Volume Filter
        if self.params.volume_filter_threshold > 0 and self.bundle.volume_df is not None:
            if current_date in self.bundle.volume_df.index:
                v_curr = self.bundle.volume_df.loc[current_date].to_numpy()
                v_curr = np.nan_to_num(v_curr, 0.0)
                
                # Determine threshold based on current universe
                # e.g. 0.8 means keep top 80% (exclude bottom 20%)
                # q = 1 - 0.8 = 0.2
                
                valid_vols = v_curr[has_price]
                if len(valid_vols) > 0:
                    q = 1.0 - self.params.volume_filter_threshold
                    cutoff = np.quantile(valid_vols, q)
                    has_volume = (v_curr >= cutoff)
                    has_price = has_price & has_volume

        # Funding data is optional for BAB but still contributes to returns
        if next_date in self.bundle.funding_df.index:
            f_next = self.bundle.funding_df.loc[next_date].to_numpy()
            has_funding = np.isfinite(f_next)
        else:
            has_funding = np.ones(len(self.bundle.tickers), dtype=bool)
        
        return has_price & has_funding.astype(bool)
    
    def run(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Run the backtest.
        
        Returns:
            equity_series, return_series, price_return_series, funding_return_series,
            turnover_series, positions_df, detailed_df
        """
        b = self.bundle
        n_dates = len(b.dates)
        if end_idx is None or end_idx >= n_dates:
            end_idx = n_dates - 1
        
        equity_path = []
        return_path = []
        price_pnl_path = []
        funding_pnl_path = []
        turnover_path = []
        date_path = []
        position_records = []
        detailed_records = []
        
        equity = 1.0
        prev_w = np.zeros(len(b.tickers))
        
        for i in range(start_idx, end_idx):
            current_date = pd.Timestamp(b.dates[i])
            next_date = current_date + pd.Timedelta(days=1)
            
            price_exists = next_date in b.price_df.index
            funding_exists = next_date in b.funding_df.index
            
            if not price_exists:
                equity_path.append(equity)
                return_path.append(0.0)
                price_pnl_path.append(0.0)
                funding_pnl_path.append(0.0)
                turnover_path.append(0.0)
                date_path.append(next_date)
                continue
            
            uni_mask = self._universe_mask(i, next_date)
            
            # Get Signals
            sigs = self.strategy.signals(i, b)
            
            # Get Weights
            weights = self.weighting.weights(i, sigs, b, uni_mask, self.params)
            
            # Calculate Turnover
            turnover = np.sum(np.abs(weights - prev_w))
            cost = turnover * (self.params.tc_bps / 10000.0)
            
            # Calculate Returns
            active = (weights != 0)
            if not np.any(active):
                pnl_price = 0.0
                pnl_funding = 0.0
                daily_ret = 0.0 - cost
            else:
                w_active = weights[active]
                
                # Price Return
                p_curr = b.price_df.iloc[i].to_numpy()[active]
                p_next = b.price_df.loc[next_date].to_numpy()[active]
                r_price = (p_next - p_curr) / p_curr
                r_price = np.nan_to_num(r_price, 0.0)
                pnl_price = np.sum(w_active * r_price)
                
                # Funding Return (if available)
                if funding_exists:
                    f_next = b.funding_df.loc[next_date].to_numpy()[active]
                    f_next = np.nan_to_num(f_next, 0.0)
                    pnl_funding = np.sum(-w_active * f_next)
                else:
                    pnl_funding = 0.0
                
                daily_ret = pnl_price + pnl_funding - cost
                
                # Record positions
                long_idx = np.where(weights > 0)[0]
                short_idx = np.where(weights < 0)[0]
                
                # Calculate realized portfolio betas
                beta_arr = sigs["beta"]
                long_beta = np.sum(weights[long_idx] * beta_arr[long_idx]) if len(long_idx) > 0 else 0.0
                short_beta = np.sum(weights[short_idx] * beta_arr[short_idx]) if len(short_idx) > 0 else 0.0
                net_beta = long_beta + short_beta
                
                position_records.append({
                    "date": next_date,
                    "long_tickers": "|".join([b.tickers[j] for j in long_idx]),
                    "short_tickers": "|".join([b.tickers[j] for j in short_idx]),
                    "long_allocations": "|".join([f"{b.tickers[j]}:{weights[j]:.4f}" for j in long_idx]),
                    "short_allocations": "|".join([f"{b.tickers[j]}:{weights[j]:.4f}" for j in short_idx]),
                    "long_positions": len(long_idx),
                    "short_positions": len(short_idx),
                    "total_long": np.sum(weights[long_idx]),
                    "total_short": np.sum(weights[short_idx]),
                    "long_beta": long_beta,
                    "short_beta": short_beta,
                    "net_beta": net_beta,
                    "daily_return": daily_ret,
                    "price_return": pnl_price,
                    "funding_return": pnl_funding,
                    "turnover": turnover
                })
                
                # Detailed per-symbol records
                f_next_full = b.funding_df.loc[next_date].to_numpy() if funding_exists else np.zeros(len(b.tickers))
                p_curr_full = b.price_df.iloc[i].to_numpy()
                p_next_full = b.price_df.loc[next_date].to_numpy()
                
                for j in np.where(weights != 0)[0]:
                    sym = b.tickers[j]
                    w_j = weights[j]
                    price_chg = (p_next_full[j] - p_curr_full[j]) / p_curr_full[j] if p_curr_full[j] != 0 else 0.0
                    detailed_records.append({
                        'date': next_date,
                        'trade_date': current_date,
                        'symbol': sym,
                        'weight': w_j,
                        'beta': beta_arr[j] if np.isfinite(beta_arr[j]) else np.nan,
                        'actual_funding': f_next_full[j] if np.isfinite(f_next_full[j]) else np.nan,
                        'price_change': price_chg,
                        'price_return_contrib': w_j * price_chg,
                        'funding_return_contrib': -w_j * (f_next_full[j] if np.isfinite(f_next_full[j]) else 0.0),
                        'actual_return_total': price_chg + (f_next_full[j] if np.isfinite(f_next_full[j]) else 0.0),
                    })
            
            equity *= (1.0 + daily_ret)
            
            equity_path.append(equity)
            return_path.append(daily_ret)
            price_pnl_path.append(pnl_price)
            funding_pnl_path.append(pnl_funding)
            turnover_path.append(turnover)
            date_path.append(next_date)
            
            prev_w = weights
        
        # Compile Results
        idx = pd.to_datetime(date_path)
        eq_s = pd.Series(equity_path, index=idx, name="equity")
        ret_s = pd.Series(return_path, index=idx, name="return")
        price_ret_s = pd.Series(price_pnl_path, index=idx, name="price_return")
        funding_ret_s = pd.Series(funding_pnl_path, index=idx, name="funding_return")
        turn_s = pd.Series(turnover_path, index=idx, name="turnover")
        
        pos_df = pd.DataFrame(position_records)
        if not pos_df.empty:
            pos_df['date'] = pd.to_datetime(pos_df['date'])
        
        detailed_df = pd.DataFrame(detailed_records)
        if not detailed_df.empty:
            detailed_df['date'] = pd.to_datetime(detailed_df['date'])
            detailed_df['trade_date'] = pd.to_datetime(detailed_df['trade_date'])
        
        return eq_s, ret_s, price_ret_s, funding_ret_s, turn_s, pos_df, detailed_df


# --- BAB Walk Forward Runner ---

class BABWalkForwardRunner:
    """
    Walk-Forward Runner for Betting Against Beta strategy.
    """
    
    def __init__(
        self,
        bundle: FundingDataBundle,
        params_grid: List[BettingAgainstBetaParams],
        train_span: int,
        test_span: int,
        step_span: int,
        score_mode: str = "sharpe",
        mode: str = "expanding",
        periods_per_year: float = PERIODS_PER_YEAR,
    ):
        self.bundle = bundle
        self.params_grid = params_grid
        self.train_span = train_span
        self.test_span = test_span
        self.step_span = step_span
        self.score_mode = score_mode
        self.mode = mode.lower()
        self.periods_per_year = periods_per_year
        
        self.strategy_cls = BettingAgainstBetaStrategy
        self.weighting_cls = BettingAgainstBetaWeighting
    
    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        all_dates = self.bundle.dates
        total = len(all_dates)
        
        results = []
        oos_returns_list = []
        oos_price_returns_list = []
        oos_funding_returns_list = []
        oos_dates_list = []
        all_positions_list = []
        all_detailed_records_list = []
        
        current_end = self.train_span
        iteration = 0
        
        while current_end < total:
            iteration += 1
            
            if self.mode == "rolling":
                train_start = max(0, current_end - self.train_span)
            else:
                train_start = 0
            
            train_end = current_end
            test_start = current_end
            test_end_inclusive = min(current_end + self.test_span, total) - 1
            
            if test_end_inclusive < test_start:
                break
            
            print(f"BAB Iteration {iteration} ({self.mode.title()}): Train [{train_start}:{train_end}], Test [{test_start}:{test_end_inclusive}]")
            
            # 1. Optimize Params (Train)
            best_score = -np.inf
            best_params = None
            best_sharpe = np.nan
            
            for params in self.params_grid:
                strat = self.strategy_cls(params)
                weight = self.weighting_cls()
                engine = BABBacktestEngine(self.bundle, strat, weight, params)
                
                _, ret, _, _, _, _, _ = engine.run(start_idx=train_start, end_idx=train_end)
                
                score = select_score(ret, mode=self.score_mode, periods_per_year=self.periods_per_year)
                sharpe = compute_sharpe(ret, periods_per_year=self.periods_per_year)
                
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params
                    best_sharpe = sharpe
            
            if best_params is None:
                best_params = self.params_grid[0]
            
            # 2. Run OOS (Test)
            strat = self.strategy_cls(best_params)
            weight = self.weighting_cls()
            engine = BABBacktestEngine(self.bundle, strat, weight, best_params)
            
            _, ret_oos, price_ret_oos, funding_ret_oos, _, pos_oos, detailed_oos = engine.run(
                start_idx=test_start, end_idx=test_end_inclusive
            )
            
            if not ret_oos.empty:
                oos_returns_list.extend(ret_oos.values)
                oos_price_returns_list.extend(price_ret_oos.values)
                oos_funding_returns_list.extend(funding_ret_oos.values)
                oos_dates_list.extend(ret_oos.index)
                
                oos_score = select_score(ret_oos, mode=self.score_mode, periods_per_year=self.periods_per_year)
                oos_sharpe = compute_sharpe(ret_oos, periods_per_year=self.periods_per_year)
                
                if not pos_oos.empty:
                    pos_oos = pos_oos.copy()
                    pos_oos['iteration'] = iteration
                    all_positions_list.append(pos_oos)
                
                if not detailed_oos.empty:
                    detailed_oos = detailed_oos.copy()
                    detailed_oos['iteration'] = iteration
                    all_detailed_records_list.append(detailed_oos)
            else:
                oos_score = np.nan
                oos_sharpe = np.nan
            
            results.append({
                "iteration": iteration,
                "train_start": all_dates[train_start],
                "train_end": all_dates[train_end],
                "test_start": all_dates[test_start],
                "test_end": all_dates[test_end_inclusive],
                "best_params": best_params,
                "is_score": best_score,
                "is_sharpe": best_sharpe,
                "oos_score": oos_score,
                "oos_sharpe": oos_sharpe
            })
            
            current_end += self.step_span
        
        # Aggregate
        oos_s = pd.Series(oos_returns_list, index=oos_dates_list).sort_index()
        oos_s = oos_s[~oos_s.index.duplicated(keep='first')]
        oos_equity = (1 + oos_s).cumprod()
        
        oos_price_s = pd.Series(oos_price_returns_list, index=oos_dates_list).sort_index()
        oos_price_s = oos_price_s[~oos_price_s.index.duplicated(keep='first')]
        
        oos_funding_s = pd.Series(oos_funding_returns_list, index=oos_dates_list).sort_index()
        oos_funding_s = oos_funding_s[~oos_funding_s.index.duplicated(keep='first')]
        
        wf_df = pd.DataFrame(results)
        positions_df = pd.concat(all_positions_list, ignore_index=True) if all_positions_list else pd.DataFrame()
        detailed_df = pd.concat(all_detailed_records_list, ignore_index=True) if all_detailed_records_list else pd.DataFrame()
        
        return wf_df, oos_s, oos_equity, oos_price_s, oos_funding_s, positions_df, detailed_df
    
    def report(
        self,
        wf_df: pd.DataFrame,
        oos_returns: pd.Series,
        oos_equity: pd.Series,
        oos_price_returns: Optional[pd.Series] = None,
        oos_funding_returns: Optional[pd.Series] = None,
        detailed_df: Optional[pd.DataFrame] = None,
        plot: bool = True,
        fig_dir: Optional[str] = None,
        title_suffix: str = "",
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive performance report for BAB strategy.
        """
        per_year = self.periods_per_year
        fig_path = Path(fig_dir) if fig_dir else None
        if fig_path:
            fig_path.mkdir(parents=True, exist_ok=True)
        
        out: Dict[str, Any] = {}
        
        # 1. Aggregate OOS performance
        if oos_returns is not None and not oos_returns.empty:
            agg_sharpe = compute_sharpe(oos_returns, periods_per_year=per_year)
            agg_sortino = compute_sortino_ratio(oos_returns, periods_per_year=per_year)
            agg_calmar = compute_calmar_ratio(oos_returns, oos_equity, periods_per_year=per_year)
            agg_comp = compute_composite_score(oos_returns, oos_equity, periods_per_year=per_year)
            agg_score = select_score(oos_returns, oos_equity, mode=self.score_mode, periods_per_year=per_year)
            agg_total_ret = oos_equity.iloc[-1] - 1
            agg_cagr = (oos_equity.iloc[-1] ** (per_year / len(oos_equity)) - 1) if len(oos_equity) else float("nan")
            
            drawdown = (oos_equity / oos_equity.cummax() - 1.0)
            max_dd = drawdown.min()

            # IR Analysis
            rolling_ir = compute_rolling_ir(oos_returns, window=30, periods_per_year=per_year)
            avg_rolling_ir = rolling_ir.mean()
            
            # IC Analysis
            daily_ic = pd.Series(dtype=float)
            rolling_icir = pd.Series(dtype=float)
            mean_ic = np.nan
            icir = np.nan
            
            if detailed_df is not None and not detailed_df.empty:
                # We forecast 'beta' and want it to correlate with future returns?
                # Actually for BAB: 
                #   We Long Low Beta -> We want Low Beta to have High Return? No, BAB says Low Beta has higher risk-adjusted return.
                #   The standard BAB factor is Long Low / Short High. 
                #   So effective "Signal" is -Beta (Negative Beta). 
                #   If Beta is high, we short (negative weight). If Beta is low, we long (positive weight).
                #   So we check correlation between -Beta and Returns.
                
                # Make a copy to avoid modifying original
                df_ic = detailed_df.copy()
                df_ic['neg_beta'] = -df_ic['beta']
                
                daily_ic = compute_daily_ic(
                    df_ic, 
                    date_col='date', 
                    forecast_col='neg_beta', 
                    target_col='actual_return_total'
                )
                
                if not daily_ic.empty:
                    mean_ic = daily_ic.mean()
                    icir = compute_icir(daily_ic)
                    rolling_icir = compute_rolling_icir(daily_ic, window=30)
            
            # Additive PnL decomposition
            if oos_price_returns is not None and oos_funding_returns is not None:
                total_pnl_additive = oos_returns.cumsum()
                price_pnl_additive = oos_price_returns.cumsum()
                funding_pnl_additive = oos_funding_returns.cumsum()
            else:
                total_pnl_additive = pd.Series(dtype=float)
                price_pnl_additive = pd.Series(dtype=float)
                funding_pnl_additive = pd.Series(dtype=float)
            
            suffix_str = f" ({title_suffix})" if title_suffix else ""
            print("=" * 80)
            print(f"AGGREGATED OUT-OF-SAMPLE PERFORMANCE (Betting Against Beta){suffix_str}")
            print("=" * 80)
            print(f"Mode: {self.mode.upper()}")
            print(f"Total OOS Bars: {len(oos_equity)}")
            if not oos_equity.empty:
                print(f"Date Range: {oos_equity.index[0].date()} to {oos_equity.index[-1].date()}")
            print("--- Risk-Adjusted ---")
            print(f"Selected Score ({self.score_mode}): {agg_score:.3f}")
            print(f"Sharpe:   {agg_sharpe:.3f}")
            print(f"Sortino:  {agg_sortino:.3f}")
            print(f"Calmar:   {agg_calmar:.3f}")
            print(f"Composite:{agg_comp:.3f}")
            print(f"Avg Rolling IR (30d): {avg_rolling_ir:.3f}")
            print("--- Forecasting Skill ---")
            print(f"Mean IC:  {mean_ic:.4f}")
            print(f"ICIR:     {icir:.3f}")
            print("--- Absolute ---")
            print(f"Total Return: {agg_total_ret*100:.2f}%")
            print(f"CAGR:         {agg_cagr*100:.2f}%")
            print(f"Max Drawdown: {max_dd*100:.2f}%")
            
            out.update({
                "agg_sharpe": agg_sharpe,
                "agg_sortino": agg_sortino,
                "agg_calmar": agg_calmar,
                "agg_composite": agg_comp,
                "agg_score": agg_score,
                "agg_total_return": agg_total_ret,
                "agg_cagr": agg_cagr,
                "agg_max_dd": max_dd,
                "mean_ic": mean_ic,
                "icir": icir,
                "daily_ic": daily_ic,
                "rolling_ir": rolling_ir,
                "rolling_icir": rolling_icir,
                "combined_equity": oos_equity,
                "combined_drawdown": drawdown,
                "total_pnl_additive": total_pnl_additive,
                "price_pnl_additive": price_pnl_additive,
                "funding_pnl_additive": funding_pnl_additive,
            })
        else:
            print("No OOS returns available")
        
        # 2. Parameter selection summary
        if wf_df is not None and not wf_df.empty:
            best_params_df = wf_df.copy()
            best_params_df["best_beta_window"] = best_params_df["best_params"].apply(
                lambda p: getattr(p, "beta_window", None)
            )
            best_params_df["best_portfolio_size"] = best_params_df["best_params"].apply(
                lambda p: getattr(p, "portfolio_size_each_side", None)
            )
            best_params_df["best_beta_type"] = best_params_df["best_params"].apply(
                lambda p: getattr(p, "beta_type", None)
            )
            best_params_df["best_target_beta"] = best_params_df["best_params"].apply(
                lambda p: getattr(p, "target_side_beta", None)
            )
            best_params_df["best_use_shrinkage"] = best_params_df["best_params"].apply(
                lambda p: getattr(p, "use_shrinkage", None)
            )
            
            print("\nParameter Selection Counts:")
            print("Beta window selection:\n", best_params_df["best_beta_window"].value_counts().sort_index())
            print("Portfolio size selection:\n", best_params_df["best_portfolio_size"].value_counts().sort_index())
            print("Beta type selection:\n", best_params_df["best_beta_type"].value_counts().sort_index())
            print("Target beta selection:\n", best_params_df["best_target_beta"].value_counts().sort_index())
            print("Use shrinkage selection:\n", best_params_df["best_use_shrinkage"].value_counts().sort_index())
            
            out["best_params_df"] = best_params_df
        else:
            print("wf_df is empty; no parameter summary available.")
        
        # 3. Plots
        if plot or fig_dir:
            try:
                import matplotlib.pyplot as plt
                plt.style.use("seaborn-v0_8")
                
                suffix_display = f" - {title_suffix}" if title_suffix else ""
                clean_suffix = title_suffix.replace(" ", "_").lower() if title_suffix else ""
                
                # A. Equity & Drawdown
                if not oos_equity.empty:
                    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, 
                                             gridspec_kw={"height_ratios": [2, 1, 1, 1]})
                    
                    # 1. Equity
                    axes[0].plot(oos_equity.index, oos_equity.values, 
                                label="BAB Equity", color="tab:green", linewidth=1.5)
                    
                    # Add BTC Benchmark
                    if self.bundle.btc_ret is not None:
                        btc_ret_oos = self.bundle.btc_ret.reindex(oos_equity.index).fillna(0)
                        btc_equity = (1 + btc_ret_oos).cumprod()
                        axes[0].plot(btc_equity.index, btc_equity.values, 
                                    label="BTC Buy & Hold", color="tab:gray", linestyle="--", alpha=0.6)
                    
                    axes[0].set_ylabel("Equity")
                    axes[0].set_title(f"Walk-Forward Equity Curve{suffix_display}")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # 2. Drawdown
                    dd_pct = (oos_equity / oos_equity.cummax() - 1.0) * 100
                    axes[1].plot(dd_pct.index, dd_pct.values, label="Drawdown %", color="tab:red")
                    axes[1].fill_between(dd_pct.index, dd_pct.values, 0, color="tab:red", alpha=0.2)
                    axes[1].set_ylabel("DD (%)")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                    
                    # 3. Rolling ICIR (Replaces Rolling IR)
                    if "rolling_icir" in out and not out["rolling_icir"].empty:
                        ricir = out["rolling_icir"]
                        axes[2].plot(ricir.index, ricir.values, label="Rolling ICIR (30d)", color="tab:purple")
                        axes[2].axhline(0, color="black", linestyle="--", alpha=0.3)
                        axes[2].set_ylabel("Rolling ICIR")
                        axes[2].legend()
                        axes[2].grid(True, alpha=0.3)
                    
                    # 4. Rolling IC (30d)
                    if "daily_ic" in out and not out["daily_ic"].empty:
                        dic = out["daily_ic"]
                        roll_ic = dic.rolling(30).mean()
                        axes[3].plot(roll_ic.index, roll_ic.values, label="Rolling IC (30d)", color="tab:orange")
                        
                        axes[3].axhline(0, color="black", linestyle="--", alpha=0.3)
                        axes[3].set_ylabel("Rolling IC")
                        axes[3].set_xlabel("Date")
                        axes[3].legend()
                        axes[3].grid(True, alpha=0.3)

                    plt.tight_layout()
                    
                    if fig_dir:
                        fname = f"bab_performance_summary_{clean_suffix}.png" if clean_suffix else "bab_performance_summary.png"
                        fig.savefig(fig_path / fname, dpi=150)
                        
                    if plot:
                        plt.show()
                    
                    plt.close(fig)
                
                # B. Parameter Selection
                if "best_params_df" in out:
                    bp = out["best_params_df"]
                    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
                    plots = [
                        ("best_beta_window", "Beta Window"),
                        ("best_portfolio_size", "Portfolio Size"),
                        ("best_beta_type", "Beta Type"),
                        ("best_target_beta", "Target Beta"),
                        ("best_use_shrinkage", "Use Shrinkage"),
                    ]
                    for ax, (col, title) in zip(axes.flatten(), plots):
                        if col in bp.columns:
                            bp[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="tab:green")
                            ax.set_title(title)
                            ax.grid(alpha=0.3)
                    # Hide unused subplot
                    axes.flatten()[-1].axis('off')
                    plt.tight_layout()
                    
                    if fig_dir:
                        fname = f"bab_parameter_counts_{clean_suffix}.png" if clean_suffix else "bab_parameter_counts.png"
                        fig.savefig(fig_path / fname, dpi=150)
                        
                    if plot:
                        plt.show()
                        
                    plt.close(fig)
                
                # C. IS vs OOS Sharpe
                if wf_df is not None and not wf_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(wf_df["iteration"], wf_df["is_sharpe"], 
                           label="IS Sharpe", marker="o", linestyle="-", alpha=0.7)
                    ax.plot(wf_df["iteration"], wf_df["oos_sharpe"], 
                           label="OOS Sharpe", marker="o", linestyle="-", alpha=0.7)
                    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Sharpe Ratio")
                    ax.set_title(f"BAB: In-Sample vs Out-of-Sample Sharpe{suffix_display}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    if fig_dir:
                        fname = f"bab_is_oos_sharpe_{clean_suffix}.png" if clean_suffix else "bab_is_oos_sharpe.png"
                        fig.savefig(fig_path / fname, dpi=150)
                    
                    if plot:
                        plt.show()
                        
                    plt.close(fig)
                
            except ImportError:
                print("matplotlib not available; skipping plots.")
        
        return out
