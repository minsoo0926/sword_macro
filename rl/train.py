import os
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from rl.env import SwordEnv
import rl.test
from rl.config import *

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = SwordEnv()
check_env(env)

def make_env(rank, seed=0):
    def _init():
        env = SwordEnv()
        env = ActionMasker(env, lambda env: env.action_masks())
        env = Monitor(env, LOG_DIR)
        env.reset(seed=seed+rank)
        return env
    set_random_seed(seed)
    return _init

def main(timesteps=TRAINING_TIMESTEPS):
    # create environment
    env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
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
        model = MaskablePPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model...")
        model = MaskablePPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA
        )

    # checkpoint save callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=MODEL_DIR, 
        name_prefix="sword_ppo"
    )

    print("Start training...")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=not loaded, callback=checkpoint_callback)

    # save model
    model.save(f"{MODEL_DIR}/sword_ppo_final")
    env.save(f"{MODEL_DIR}/vec_normalize.pkl")
    
    print("Training completed and saved successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, default=TRAINING_TIMESTEPS, help='Total training timesteps')
    args = parser.parse_args()
    main(timesteps=args.t)
    rl.test.run_test()