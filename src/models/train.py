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
import gymnasium as gym

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Local Imports
import src.envs  # Ensure envs are registered

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    """""
    Main Training Loop
    """""
    # 1. Path resolution
    root_path = Path(hydra.utils.get_original_cwd())
    data_path = root_path / cfg.env.data_dir / f"{cfg.env.ticker}_processed.parquet"
    
    print(f"Loading data from: {data_path}")
    
    # 2. Load Data
    print("data_path:", data_path)
    actual_path = "/Users/zachary.tchir/dev/personal/trading-rl-mle/data/processed/AAPL_processed.parqiuet"
    print(actual_path)
    print(data_path == actual_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_parquet(data_path)
    
    # 3. Initialize Environment
    env = DummyVecEnv([lambda: Monitor(gym.make(
        cfg.env.id,
        df=df,
        window_size=cfg.env.window_size,
        initial_balance=cfg.env.initial_balance,
        commission=cfg.env.commission
    ))])
    
    # 4. Initialize Agent
    # We map Hydra config params directly to the PPO class
    print(f"Initializing PPO Agent on device: {cfg.device}")
    model = PPO(
        policy=cfg.agent.policy,
        env=env,
        verbose=cfg.agent.verbose,
        learning_rate=cfg.agent.learning_rate,
        n_steps=cfg.agent.n_steps,
        batch_size=cfg.agent.batch_size,
        n_epochs=cfg.agent.n_epochs,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_range=cfg.agent.clip_range,
        ent_coef=cfg.agent.ent_coef,
        device=cfg.device, # Important: 'mps' for Mac, 'cuda' for Nvidia
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 5. Callbacks
    # Save the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ppo_trading"
    )
    
    # 6. Train
    print("Starting training...")
    total_timesteps = cfg.agent.total_timesteps
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    model_name = f"ppo_trading_{cfg.env.ticker}_{total_timesteps}_steps"
    model.save(model_name)
    print(f"Training complete. Model saved as {model_name}.zip")
    
if __name__ == "__main__":
    train()