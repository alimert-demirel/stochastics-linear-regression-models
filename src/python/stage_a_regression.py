import os
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

def initialize_directories():
    """Ensure data and output directories exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def fetch_financial_data(tickers, start_date="2021-01-01", end_date="2026-01-01"):
    """Download historical closing prices using yf.Ticker to avoid MultiIndex bugs."""
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    df_list = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        price_series = hist['Close'].rename(ticker)
        df_list.append(price_series)
        
    data = pd.concat(df_list, axis=1)
    
    data.index = data.index.tz_localize(None)
    data.dropna(inplace=True)
    
    data.to_csv("data/raw_prices.csv")
    return data

def run_baseline_regression(df, y_ticker="JPM", x_ticker="BAC"):
    """
    Run OLS Multiple Linear Regression to find the Hedge Ratio (Beta).
    Y = Beta * X + Alpha + Residual (Spread)
    """
    Y = df[y_ticker]
    X = df[x_ticker]
    X_with_constant = sm.add_constant(X)
    
    model = sm.OLS(Y, X_with_constant).fit()
    
    alpha = model.params['const']
    beta = model.params[x_ticker]
    r_squared = model.rsquared
    
    print("\n--- Baseline Regression Results ---")
    print(f"Hedge Ratio (Beta): {beta:.4f}")
    print(f"Intercept (Alpha):  {alpha:.4f}")
    print(f"R-squared Score:    {r_squared:.4f}")
    
    df['Spread'] = Y - (beta * X + alpha)
    df.to_csv("data/processed_spread.csv")
    
    return beta, alpha, df

def generate_readme_plots(df, y_ticker="JPM", x_ticker="BAC", beta=1.0, alpha=0.0):
    """Generate high-quality visualizations for the repository README."""
    # Plot 1: Historical Normalized Price Comparison
    plt.figure()
    normalized_df = df[[y_ticker, x_ticker]] / df[[y_ticker, x_ticker]].iloc[0]
    plt.plot(normalized_df[y_ticker], label=f"{y_ticker} (Normalized)", color="#1f77b4")
    plt.plot(normalized_df[x_ticker], label=f"{x_ticker} (Normalized)", color="#ff7f0e")
    plt.title("Normalized Price Trajectory: JPM vs BAC")
    plt.xlabel("Date")
    plt.ylabel("Normalized Scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/price_trajectory.png", dpi=300)
    plt.close()

    # Plot 2: The generated Arbitrage Spread.
    plt.figure()
    plt.plot(df['Spread'], color="#2ca02c", label="Residual Spread")
    plt.axhline(df['Spread'].mean(), color="black", linestyle="--", label="Mean (0)")
    plt.title(f"Statistical Arbitrage Spread: {y_ticker} - ({beta:.2f} * {x_ticker} + {alpha:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Spread Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/arbitrage_spread.png", dpi=300)
    plt.close()
    print("\nVisualizations successfully saved to the 'outputs/' directory.")

if __name__ == "__main__":
    initialize_directories()
    
    price_data = fetch_financial_data(["JPM", "BAC"], start_date="2021-01-01", end_date="2026-01-01")
    
    hedge_ratio, intercept, updated_df = run_baseline_regression(price_data, "JPM", "BAC")
    
    generate_readme_plots(updated_df, "JPM", "BAC", hedge_ratio, intercept)