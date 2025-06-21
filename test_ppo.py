import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import lfc_gym


def run_test():
    env = gym.make("SingleAreaLFC-v0", max_time=300.0)
    model = PPO.load("ppo_single_area_lfc")

    obs, _ = env.reset()
    env.unwrapped.disturbance_schedule = {100.0: (90.0, 0.02), 200.0: (100.0, -0.04)}

    time_log, df_log, dpload_log, dpm_log, u_log, reward_log = [], [], [], [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        time_log.append(env.unwrapped.t)
        df_log.append(obs[0])
        dpload_log.append(env.unwrapped.current_dPload)
        dpm_log.append(env.unwrapped.current_dPm)
        u_log.append(action[0] if isinstance(action, (np.ndarray, list)) else action)
        reward_log.append(reward)

    time_log = np.array(time_log)
    plt.figure(figsize=(10, 11))

    plt.subplot(5, 1, 1)
    plt.plot(time_log, df_log, label='Δf (pu)')
    plt.grid(), plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(time_log, dpload_log, label='ΔPload (pu)', color='orange')
    plt.grid(), plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(time_log, dpm_log, label='ΔPm (pu)', color='green')
    plt.grid(), plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(time_log, u_log, label='Control signal (pu)', color='purple')
    plt.grid(), plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(time_log, reward_log, label='Reward', color='red')
    plt.grid(), plt.legend()

    plt.xlabel('Time (s)')
    plt.suptitle('PPO Test: Disturbance from 30s to 120s with Reward Plot')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test()