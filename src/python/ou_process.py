import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

def calibrate_ou_process(spread_series, dt=1/252):
    """
    Calibrates the Ornstein-Uhlenbeck process using Linear Regression.
    x_{t} = a + b * x_{t-1} + error
    """
    y = spread_series.values[1:]
    x = spread_series.values[:-1]
    
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    
    a = model.params[0]
    b = model.params[1]
    residuals = model.resid
    
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.std(residuals) / np.sqrt(dt)
    
    half_life = np.log(2) / theta
    
    print("\n--- Ornstein-Uhlenbeck Calibration ---")
    print(f"Long-term Mean (Mu):       {mu:.4f}")
    print(f"Speed of Reversion (Theta): {theta:.4f}")
    print(f"Volatility (Sigma):         {sigma:.4f}")
    print(f"Half-Life (Trading Days):   {half_life:.2f} days")
    
    return mu, theta, sigma

def generate_trading_signals(df, mu, sigma, dt=1/252):
    """Calculates Z-scores to define dynamic entry and exit thresholds."""
    eq_sigma = sigma / np.sqrt(2 * theta) if 'theta' in locals() else np.std(df['Spread'])
    
    df['Z_Score'] = (df['Spread'] - mu) / eq_sigma
    
    entry_threshold = 2.0  
    exit_threshold = 0.5   
    
    df.to_csv("data/trading_signals.csv")
    return df, entry_threshold, exit_threshold

def plot_trade_signals(df, entry_thresh, exit_thresh):
    """Plots the Z-Score of the spread with trade execution bands."""
    plt.figure()
    plt.plot(df['Z_Score'], color="#9467bd", label="Spread Z-Score")
    
    # Plot thresholds
    plt.axhline(entry_thresh, color="red", linestyle="--", alpha=0.7, label=f"Short Spread (+{entry_thresh} SD)")
    plt.axhline(-entry_thresh, color="green", linestyle="--", alpha=0.7, label=f"Buy Spread (-{entry_thresh} SD)")
    plt.axhline(exit_thresh, color="black", linestyle=":", alpha=0.5, label="Exit Threshold")
    plt.axhline(-exit_thresh, color="black", linestyle=":", alpha=0.5)
    plt.axhline(0, color="black", linewidth=1.5, label="Mean (0)")
    
    plt.title("Statistical Arbitrage Signals: JPM/BAC Spread Z-Score")
    plt.xlabel("Date")
    plt.ylabel("Standard Deviations from Mean")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/ou_trade_signals.png", dpi=300)
    plt.close()
    print("\nTrade signal visualization saved to 'outputs/ou_trade_signals.png'.")

if __name__ == "__main__":
    try:

        data = pd.read_csv("data/processed_spread.csv", index_col="Date", parse_dates=True)


        mu, theta, sigma = calibrate_ou_process(data['Spread'])


        signal_df, enter, exit = generate_trading_signals(data, mu, sigma)
   

        plot_trade_signals(signal_df, enter, exit)
        
    except FileNotFoundError:
        print("Error: Could not find 'data/processed_spread.csv'. Please run the regression script first.")