from gymnasium.envs.registration import register
from .single_area_lfc_env import SingleAreaLFCEnv

register(
    id="SingleAreaLFC-v0",
    entry_point="lfc_gym.single_area_lfc_env:SingleAreaLFCEnv",
)
