import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from env import SwordEnv
import rl.test

LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = "./models/sword_ppo_final.zip"
STATS_PATH = "./models/vec_normalize.pkl"

def make_env():
    env = SwordEnv()
    check_env(env)
    env = Monitor(env, LOG_DIR)
    return env

def main():
    # create environment
    env = DummyVecEnv([make_env])
    if os.path.exists(STATS_PATH):
        env = VecNormalize.load(STATS_PATH, env)
        env.training = True
        env.norm_reward = True
        loaded = True
    else:
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=True, 
            clip_obs=10.
        )
        loaded = False

    # create model
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64
        )

    # checkpoint save callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=MODEL_DIR, 
        name_prefix="sword_ppo"
    )

    print("Start training...")
    model.learn(total_timesteps=1000000, reset_num_timesteps=not loaded, callback=checkpoint_callback)

    # save model
    model.save(f"{MODEL_DIR}/sword_ppo_final")
    env.save(f"{MODEL_DIR}/vec_normalize.pkl")
    
    print("Training completed and saved successfully!")

if __name__ == "__main__":
    main()
    rl.test.test()