import sys
import os

# --- BRUTE FORCE FIX: Add Project Root to Path ---
# Get the absolute path of the 'trading-rl-mle' folder
# We go up 3 levels: src/models/train.py -> src/models -> src -> trading-rl-mle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force add it to Python's path
if project_root not in sys.path:
    sys.path.append(project_root)

import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig):
    # 1. Load Data
    root_path = Path(hydra.utils.get_original_cwd())
    data_path = root_path / cfg.env.data_dir / f"{cfg.env.ticker}_processed.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"Loaded data: {len(df)} bars")

    # 2. Recreate Environment (Raw, no Monitor needed for inference)
    env = gym.make(
        cfg.env.id,
        df=df,
        window_size=cfg.env.window_size,
        initial_balance=cfg.env.initial_balance,
        commission=cfg.env.commission
    )
    
    # 3. Load Trained Model
    # We look for the model saved at the end of training
    # NOTE: Ensure this filename matches what your train.py saved!
    model_name = f"ppo_{cfg.env.ticker}_{cfg.agent.total_timesteps}_steps.zip"
    model_path = root_path / model_name
    
    if not model_path.exists():
        print(f"Warning: Could not find {model_name}.")
        # Try finding the latest checkpoint if the main model is missing
        checkpoints = list((root_path / "checkpoints").glob("*.zip"))
        if checkpoints:
            model_path = sorted(checkpoints)[-1] # Take the latest
            print(f"Defaulting to latest checkpoint: {model_path.name}")
        else:
            raise FileNotFoundError(f"No models found in {root_path}")
        
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # 4. Run Backtest Loop
    obs, _ = env.reset()
    done = False
    
    portfolio_values = []
    
    # Calculate Buy & Hold Baseline for comparison
    initial_price = df.iloc[cfg.env.window_size]['Close']
    initial_balance = cfg.env.initial_balance
    shares_buy_hold = initial_balance / initial_price
    
    print("Starting Backtest...")
    while not done:
        # Predict action (Deterministic for evaluation, no exploration)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step Environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info['net_worth'])
        
    # 5. Results
    portfolio_values = np.array(portfolio_values)
    
    # Calculatio Buy & Hold Curve (Approximate based on Close prices)
    # We slice the DF to matchthe steps the agent actually took
    # Agent starts at index 'window_size', and runs until the end
    
    relevant_prices = df['Close'].iloc[cfg.env.window_size : cfg.env.window_size + len(portfolio_values)].values
    buy_hold_values = shares_buy_hold * relevant_prices
    
    # Returns Calculation
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    # Annualized Sharpe: Mean / Std * Sqrt(Trading Steps per Year)
    # Assuming 30min bars -> 13 bars/day -> 252 days ~ 3276 steps/year
    # Adjust this scalar based on your actual data frequency!
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 13)
    
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    bh_return = (buy_hold_values[-1] - initial_balance) / initial_balance * 100
    
    print(f"--- Results ---")
    print(f"Final Balance:    ${portfolio_values[-1]:.2f}")
    print(f"Total Return:     {total_return:.2f}%")
    print(f"Buy & Hold Ret:   {bh_return:.2f}%")
    print(f"Sharpe Ratio:     {sharpe:.4f}")
    
    # 6. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='RL Agent', color='blue')
    plt.plot(buy_hold_values, label='Buy & Hold', color='gray', linestyle='--', alpha=0.6)
    plt.axhline(y=initial_balance, color='red', linestyle=':', label='Initial Balance')
    
    plt.title(f"Backtest: RL Agent vs Buy & Hold ({cfg.env.ticker})")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = root_path / "backtest_result.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    # plt.show() # Uncomment if running locally with a display
    
if __name__ == "__main__":
    evaluate()