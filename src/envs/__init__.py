from gymnasium.envs.registration import register
from .trading_env import TradingEnv

register(
    id="TradingEnv-v0",
    entry_point="src.envs.trading_env:TradingEnv",
)