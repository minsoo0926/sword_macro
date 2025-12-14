# File directiries and paths
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
MODEL_PATH = "./models/sword_ppo_final.zip"
STATS_PATH = "./models/vec_normalize.pkl"

# Macro config
LEVEL_THRESHOLD = 10
CHAT_OUTPUT_COORD = (200, 500)  # x, y coordinates of the chat output box
CHAT_INPUT_COORD = (200, 900)   # x, y coordinates of the chat input box

# Training config
TRAINING_TIMESTEPS = 1000000
N_ENVS = 8
N_STEPS = 2048
BATCH_SIZE = 64
LEARNING_RATE = 0.0003

# Game Env config
MAX_STEPS = 1000
TARGET_RATE = 5
MINIMUM_FUND = 10000
MINIMUM_SELL_LEVEL = 5
REWARD_COEFF = 0.001