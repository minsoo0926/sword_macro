import time
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from rl.config import *
from rl.env import SwordEnv 
from rl.inference import SwordAI


def make_env(seed=None):
    def _init():
        env = SwordEnv()
        env = ActionMasker(env, lambda env: env.action_masks())
        return env
    return _init

def run_test(mode = 'ai'):
    env = DummyVecEnv([make_env()])
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(MODEL_PATH)

    print("\n=== Test start (Ctrl+C to stop) ===")
    obs = env.reset()
    print("Initial Funds:", env.envs[0].env.state[0])
    
    total_reward = 0
    step_count = 0
    budget_history = []
    level_history = []
    reward_history = []
    fail_count_history = []

    try:
        while True:
            action_masks = get_action_masks(env)
            if mode == 'ai':
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                ai = SwordAI()
                action = ai.heuristic(
                    fund=env.envs[0].env.state[0],
                    level=env.envs[0].env.state[1],
                    fail_count=env.envs[0].env.state[3]
                )
                action = [action if action != -1 else 0]
            obs, rewards, dones, info = env.step(action)
            budget_history.append(env.envs[0].env.state[0])
            level_history.append(env.envs[0].env.state[1])
            fail_count_history.append(env.envs[0].env.state[3])
            reward = rewards[0]
            reward_history.append(reward)
            total_reward += reward
            step_count += 1

            if step_count % 100 == 0:
                print(f"Step: {step_count} | Action: {action[0]} | Reward: {reward:.2f}")
                env.render(mode='human') 
            
            if dones[0]:
                print(f"--- Episode finished,Total reward: {total_reward:.2f} ---")
                break
                obs = env.reset()
                total_reward = 0
                step_count = 0
                budget_history = []
                level_history = []
                time.sleep(1.0) # Pause briefly to observe results
    # Plot budget and level history
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 2, 1)
        plt.plot(budget_history[:-1], label='Budget (G)')
        plt.xlabel('Steps')
        plt.ylabel('Budget (G)')
        plt.title('Budget Over Time')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(level_history[:-1], label='Sword Level (+)', color='orange')
        plt.xlabel('Steps')
        plt.ylabel('Sword Level (+)')
        plt.title('Sword Level Over Time')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(reward_history[:-1], label='Reward', color='green')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Reward Over Time')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(fail_count_history[:-1], label='Fail Count', color='red')
        # plt.hist(fail_count_history, bins=20, color='purple', alpha=0.7)
        plt.xlabel('Steps')
        plt.ylabel('Fail Count')
        plt.title('Fail Count Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()


    except KeyboardInterrupt:
        print("\nTest stopped by user.")

if __name__ == "__main__":
    run_test('ai')