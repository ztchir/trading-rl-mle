import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces   
from typing import Dict, Any

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size: int = 30, initial_balance: float = 10000, commission: float = 0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Action Space 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)  # e.g., Buy, Hold, Sell
        
        # Observation Space
        # Shape: (window_size, number_of_features + 2)
        # Two additional features for balance and "Shares Held")
        self.n_features = df.shape[1]
        self.obs_shape = (self.window_size, self.n_features + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )
        
    def _load_data(self):
        # Placeholder for data loading logic
        return np.random.rand(100, 10)  # Example data
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset Internal State
        self.balance = self.initial_balance
        self.shares_held = 0
        self.networth = self.initial_balance
        self.max_net_woth = self.initial_balance
        
        # Pointer to current time step
        # Start at window_size to have enough data for first observation
        self.current_step = self.window_size
        
        return self._get_observation(), {}
    
    def _get_observation(self):
            # 1. Get Market Data Window
            # Shape: (30, n_features)
            market_window = self.df.iloc[self.current_step - self.window_size : self.current_step].values

            # 2. Add Account State (Repeated for the whole window to match shape)
            # This helps the Conv1D/LSTM layers "see" the account status at every time step
            account_state = np.array([self.balance, self.shares_held])
            
            # Expand account state to match window length: (30, 2)
            account_matrix = np.tile(account_state, (self.window_size, 1))
            
            # 3. Concatenate
            obs = np.hstack((market_window, account_matrix))
            
            return obs.astype(np.float32)
        
    def step(self, action):
        # Current Market Price
        current_price = self.df.iloc[self.current_step]['Close']
        
        # 1. Execute Trade Logic
        if action == 1:  # Buy
            # Can we afford at least one share?
            max_shares = self.balance // current_price
            if max_shares > 0:
                shares_bought = max_shares
                cost = shares_bought * current_price * (1 + self.commission)
                self.balance -= cost
                self.shares_held += shares_bought
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_sold = self.shares_held
                revenue = shares_sold * current_price * (1 - self.commission)
                self.balance += revenue
                self.shares_held = 0
        
        
        # Action 0 (Hold) does nothing
        
        # 2. Update Step and Net Worth
        self.current_step += 1

        new_net_worth = self.balance + self.shares_held * current_price
        
        # 3. Calculate Reward
        # ----------------------------------------
        # Reward = Log Return of Portfolio
        # R_t = ln(V_t / V_{t-1})
        
        reward = np.log(new_net_worth / self.networth) if self.networth > 0 else 0
        
        self.networth = new_net_worth
        if self.networth > self.max_net_woth:
            self.max_net_woth = self.networth   
        
            
        # 4. Check Termination
        # ----------------------------------------
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if self.networth < self.initial_balance * 0.1:
            terminated = True  
        
        # 5. Return
        info = {'net_worth': self.networth, 'shares': self.shares_held}
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Shares: {self.shares_held}")