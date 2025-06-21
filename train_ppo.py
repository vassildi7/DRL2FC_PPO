import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import lfc_gym  # Custom env registration


def make_env():
    return gym.make(
        "SingleAreaLFC-v0",
        max_time=300.0,
        disturbance_segment_duration=40.0,
        disturbance_range=(-0.06, 0.06)
    )


def evaluate_model(model, env, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)


class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_rewards.append(reward)
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Episode reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Episode Rewards")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=dict(net_arch=[dict(pi=[128, 64], vf=[128, 64])]),
    )

    reward_tracker = RewardTrackingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_lfc_checkpoint'
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ppo_best_model/",
        log_path="./ppo_logs/",
        eval_freq=3000,
        n_eval_episodes=5,
        deterministic=True
    )

    model.learn(
        total_timesteps=200_000,
        callback=[reward_tracker, eval_callback, checkpoint_callback]
    )

    model.save("ppo_single_area_lfc")

    mean_reward, std_reward = evaluate_model(model, eval_env)
    print("\nTraining Complete.")
    print(f"Evaluation over 10 episodes:\n  Mean Reward: {mean_reward:.2f}\n  Std Reward: {std_reward:.2f}")

    reward_tracker.plot_rewards()


if __name__ == "__main__":
    main()