import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import json
import datetime
import time
import os

def get_binance_klines(symbol, interval, start_time, end_time):
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    all_data = []
    
    current_start = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    print(f"Fetching {symbol} data...")
    
    while current_start < end_ts:
        url = f"{base_url}?symbol={symbol}&interval={interval}&startTime={current_start}&endTime={end_ts}&limit={limit}"
        
        try:
            with urllib.request.urlopen(url) as response:
                if response.status != 200:
                    print(f"Error fetching data: {response.reason}")
                    break
                
                data = json.loads(response.read().decode())
                if not data:
                    break
                
                all_data.extend(data)
                last_open_time = data[-1][0]
                current_start = last_open_time + 86400000 
                
                time.sleep(0.1) 
                
        except Exception as e:
            print(f"Exception fetching data: {e}")
            break
            
    return all_data

def main():
    # 1. Load Strategy Data
    if not os.path.exists('detailed_records_bab_wf.csv'):
        print("Error: detailed_records_bab_wf.csv not found.")
        return

    print("Loading detailed records...")
    df = pd.read_csv('detailed_records_bab_wf.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate to daily stats
    # We want: 
    # - Strategy Daily Return (sum of contributions)
    # - Long Leg Return Contribution
    # - Short Leg Return Contribution
    # - Allocation (sum of weights)
    
    daily_stats = df.groupby('date').apply(lambda x: pd.Series({
        'strat_return': x['price_return_contrib'].sum() + x['funding_return_contrib'].sum(),
        'long_ret_contrib': x.loc[x['weight'] > 0, 'price_return_contrib'].sum() + x.loc[x['weight'] > 0, 'funding_return_contrib'].sum(),
        'short_ret_contrib': x.loc[x['weight'] < 0, 'price_return_contrib'].sum() + x.loc[x['weight'] < 0, 'funding_return_contrib'].sum(),
        'long_exposure': x.loc[x['weight'] > 0, 'weight'].sum(),
        'short_exposure': x.loc[x['weight'] < 0, 'weight'].sum(),
    }))
    
    daily_stats = daily_stats.sort_index()
    print(f"Strategy data range: {daily_stats.index.min().date()} to {daily_stats.index.max().date()}")

    # 2. Fetch BTC Data
    # We need data starting a bit before the strategy start to calculate the first return
    start_fetch = daily_stats.index.min() - datetime.timedelta(days=5)
    end_fetch = daily_stats.index.max() + datetime.timedelta(days=1)
    
    btc_raw = get_binance_klines("BTCUSDT", "1d", start_fetch, end_fetch)
    
    if not btc_raw:
        print("Failed to fetch BTC data. Cannot perform comparative analysis.")
        return

    btc_df = pd.DataFrame(btc_raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    # Binance open_time is ms timestamp
    btc_df['date'] = pd.to_datetime(btc_df['open_time'], unit='ms').dt.normalize()
    btc_df['close'] = btc_df['close'].astype(float)
    btc_df = btc_df.set_index('date').sort_index()
    
    # Calculate BTC daily returns
    btc_df['btc_return'] = btc_df['close'].pct_change()
    
    # 3. Merge
    combined = daily_stats.join(btc_df[['btc_return']], how='inner')
    
    if combined.empty:
        print("No overlapping dates between Strategy and BTC data.")
        return

    print(f"Combined data points: {len(combined)}")

    # 4. Analysis
    
    # A. Correlations
    corr_concurrent = combined['strat_return'].corr(combined['btc_return'])
    corr_lead = combined['strat_return'].corr(combined['btc_return'].shift(-1))
    corr_lag = combined['strat_return'].corr(combined['btc_return'].shift(1))
    
    print("-" * 40)
    print("CORRELATION ANALYSIS")
    print("-" * 40)
    print(f"Corr(Strat_t, BTC_t):   {corr_concurrent:.4f}")
    print(f"Corr(Strat_t, BTC_t+1): {corr_lead:.4f} (Strategy predicts Market?)")
    print(f"Corr(Strat_t, BTC_t-1): {corr_lag:.4f} (Market predicts Strategy?)")
    
    # B. Rolling Realized Beta Calculation
    window = 60
    print(f"Calculating {window}-day Rolling Realized Betas...")
    
    rolling_cov_net = combined['strat_return'].rolling(window).cov(combined['btc_return'])
    rolling_cov_long = combined['long_ret_contrib'].rolling(window).cov(combined['btc_return'])
    rolling_cov_short = combined['short_ret_contrib'].rolling(window).cov(combined['btc_return'])
    rolling_var_btc = combined['btc_return'].rolling(window).var()
    
    combined['realized_beta_net'] = rolling_cov_net / rolling_var_btc
    combined['realized_beta_long'] = rolling_cov_long / rolling_var_btc
    combined['realized_beta_short'] = rolling_cov_short / rolling_var_btc

    print(f"Mean Realized Net Beta:   {combined['realized_beta_net'].mean():.4f}")
    print(f"Mean Realized Long Beta:  {combined['realized_beta_long'].mean():.4f}")
    print(f"Mean Realized Short Beta: {combined['realized_beta_short'].mean():.4f}")

    # C. Statistics
    print("-" * 40)
    print("RETURN STATISTICS")
    print("-" * 40)
    print(f"Strategy Mean Daily Ret: {combined['strat_return'].mean():.5f}")
    print(f"BTC Mean Daily Ret:      {combined['btc_return'].mean():.5f}")
    print(f"Strategy Daily Std:      {combined['strat_return'].std():.5f}")
    print(f"BTC Daily Std:           {combined['btc_return'].std():.5f}")
    print(f"Strategy Sharpe (Ann):   {(combined['strat_return'].mean()/combined['strat_return'].std())*np.sqrt(365):.2f}")
    
    # 5. Plotting
    
    # Set style
    plt.style.use('ggplot')
    
    # Plot 1: Cumulative Returns
    plt.figure(figsize=(12, 6))
    (1 + combined['strat_return']).cumprod().plot(label='Strategy (BAB)')
    (1 + combined['btc_return']).cumprod().plot(label='BTC', linestyle='--', alpha=0.7)
    plt.title('Cumulative Returns: BAB Strategy vs BTC')
    plt.legend()
    plt.ylabel('Growth of $1')
    plt.tight_layout()
    plt.savefig('bab_vs_btc_equity.png')
    print("Saved bab_vs_btc_equity.png")
    
    # Plot 2: Return Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(combined['strat_return'], bins=50, alpha=0.6, label='Strategy', density=True)
    plt.hist(combined['btc_return'], bins=50, alpha=0.6, label='BTC', density=True)
    plt.title('Daily Return Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('return_distribution.png')
    print("Saved return_distribution.png")
    
    # Plot 3: Long/Short Exposure (Gross)
    plt.figure(figsize=(12, 6))
    combined['long_exposure'].plot(label='Long Exposure', color='green', alpha=0.7)
    combined['short_exposure'].plot(label='Short Exposure', color='red', alpha=0.7)
    (combined['long_exposure'] + combined['short_exposure']).plot(label='Net Gross Exposure', color='black', linewidth=1)
    plt.title('Portfolio Allocation (Weights)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('exposure.png')
    print("Saved exposure.png")
    
    # Plot 4: Realized Portfolio Beta
    plt.figure(figsize=(12, 6))
    combined['realized_beta_long'].plot(label='Realized Long Beta', color='green', alpha=0.6)
    combined['realized_beta_short'].plot(label='Realized Short Beta', color='red', alpha=0.6)
    combined['realized_beta_net'].plot(label='Realized Net Beta', color='blue', linewidth=1.5)
    plt.title(f'Realized Portfolio Beta vs BTC ({window}d Rolling)')
    plt.ylabel('Beta')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_beta.png')
    print("Saved portfolio_beta.png")

if __name__ == "__main__":
    main()