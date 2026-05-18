import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (14, 7)

def price_put_option(paths_df, strike_price, risk_free_rate=0.04, T=1.0):
    """
    Prices a European Put Option using Monte Carlo final states.
    Payoff = max(Strike - S_T, 0)
    """

    final_prices = paths_df.iloc[:, -1]
    
    payoffs = np.maximum(strike_price - final_prices, 0)
    
    option_price = np.exp(-risk_free_rate * T) * np.mean(payoffs)
    return option_price

def plot_simulations(gbm_df, merton_df, strike_price):
    """Generates the final comparative visualization."""
    fig, axes = plt.subplots(1, 2, sharey=True)
    
    # Plotting the first 100 paths to avoid massive file sizes
    for i in range(100):
        axes[0].plot(gbm_df.iloc[i, :], color="#1f77b4", alpha=0.1)
        axes[1].plot(merton_df.iloc[i, :], color="#d62728", alpha=0.1)

    axes[0].set_title("Standard Model (Geometric Brownian Motion)")
    axes[0].set_xlabel("Trading Days")
    axes[0].set_ylabel("Simulated Price ($)")
    axes[0].axhline(strike_price, color="black", linestyle="--", label=f"Strike Price (${strike_price})")
    axes[0].legend()

    axes[1].set_title("Risk of Collapse (Merton's Jump-Diffusion)")
    axes[1].set_xlabel("Trading Days")
    axes[1].axhline(strike_price, color="black", linestyle="--", label=f"Strike Price (${strike_price})")
    axes[1].legend()

    plt.suptitle("Monte Carlo Simulations: Pricing Tail Risk in Statistical Arbitrage", fontsize=16)
    plt.tight_layout()
    plt.savefig("../../outputs/monte_carlo_simulations.png", dpi=300)
    plt.close()
    print("\nSimulation visualization saved to 'outputs/monte_carlo_simulations.png'.")

if __name__ == "__main__":
    try:
        # load the c++ generated data
        print("Loading C++ simulation matrices...")
        gbm_data = pd.read_csv("../../data/gbm_paths.csv", header=None)
        merton_data = pd.read_csv("../../data/merton_paths.csv", header=None)
        
        STRIKE = 35.0 
        
        # Calculatng Option Prices
        gbm_price = price_put_option(gbm_data, STRIKE)
        merton_price = price_put_option(merton_data, STRIKE)
        
        print("\n--- Derivatives Pricing Engine (Protective Put) ---")
        print(f"Strike Price: ${STRIKE:.2f}")
        print(f"Price under standard GBM model:    ${gbm_price:.2f}")
        print(f"Price under Merton's Jump model: ${merton_price:.2f}")
        print("-" * 50)
        
        difference = ((merton_price - gbm_price) / gbm_price) * 100
        print(f"Conclusion: Standard models UNDERPRICE tail-risk insurance by {difference:.1f}%")
        
        plot_simulations(gbm_data, merton_data, STRIKE)
        
    except FileNotFoundError:
        print("Error: Could not find the C++ CSV files. Make sure to compile and run 'monte_carlo_engine.cpp' first.")