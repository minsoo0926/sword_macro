from gymnasium import Env, spaces
import numpy as np
import pandas as pd
from rl.config import *

level_summary = pd.read_csv('./level_summary.csv')
level_summary_dict = level_summary.to_dict('index')
level_cost = {
    0: 10,
    1: 20,
    2: 50,
    3: 100,
    4: 200,
    5: 500,
    6: 1000,
    7: 2000,
    8: 5000,
    9: 10000,
    10: 20000,
    11: 30000,
    12: 40000,
    13: 50000,
    14: 70000,
    15: 100000,
    16: 150000,
    17: 200000,
    18: 300000,
    19: 400000,
    20: 500000,
}

class SwordEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        low_limits = np.array([0, 0, 0, 0], dtype=np.int32) # fund, sword level, cost to enhance, fail count
        high_limits = np.array([1e8, 20, 1e6, 50], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_limits, high=high_limits, shape=(4,), dtype=np.int32)
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.target_rate = TARGET_RATE
        # self.minimum_fund = MINIMUM_FUND
        self.minimum_sell_level = MINIMUM_SELL_LEVEL
        self.reward_coeff = REWARD_COEFF
        self.level_data = level_summary_dict

    def action_masks(self):
        masks = [True, True]
        level = self.state[1]
        cost = level_cost[level]

        if self.state[0] < cost:
            masks[0] = False  # cannot enhance
        if level < self.minimum_sell_level:
            masks[1] = False  # cannot sell
        if level >= 20:
            masks[0] = False  # cannot enhance beyond level 20
        return np.array(masks, dtype=bool)

    def get_sell_price(self, level):
        sell_avg = self.level_data[level]['AVG_sell']
        sell_std = self.level_data[level]['STD_sell']
        sell_price = max(0, int(self.np_random.normal(sell_avg, sell_std)))
        return sell_price
    
    def avg_value(self, level):
        return self.level_data[level]['AVG_sell']
    
    def sell(self):
        if self.state[1] == 0:
            return 0
        sell_price = self.get_sell_price(self.state[1])
        self.state[0] += sell_price
        self.state[1] = 0
        self.state[3] = 0
        return sell_price

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random start level and fund for diverse start
        start_level = int(self.np_random.integers(0, 20))
        start_cost = level_cost[start_level]
        start_fund = int((start_level+1) * self.np_random.integers(1, 5) * 100000)

        self.state = np.array([start_fund, start_level, start_cost, 0], dtype=np.int32)
        self.current_step = 0
        self.target_fund = start_fund * self.target_rate
        self.minimum_fund = start_fund / self.target_rate
        return self.state, {}
    
    def step(self, action):
        reward = 0.0
        done = False
        truncated = False
        level = self.state[1]
        cost = level_cost[level]
        self.current_step += 1

        # Enhance
        if action == 0:
            self.state[0] -= cost
            # multinomial probability for success/remain/break
            success_prob = self.level_data[level]['성공'] / 100
            remain_prob = self.level_data[level]['유지'] / 100
            break_prob = self.level_data[level]['파괴'] / 100
            outcome = self.np_random.choice(['success', 'remain', 'break'], p=[success_prob, remain_prob, break_prob])

            reward -= cost * self.reward_coeff  # cost penalty
            if outcome == 'success':
                self.state[1] += 1
                reward += (self.avg_value(level + 1) - self.avg_value(level)) * self.reward_coeff
            elif outcome == 'break':
                self.state[1] = 0
                # reward -= 10
                reward -= self.avg_value(level) * self.reward_coeff * 0.5  # penalty for breakage
            
            if outcome == 'remain':
                self.state[3] += 1
            else:
                self.state[3] = 0

        # Sell
        elif action == 1:
            sell_price = self.sell()
        else:
            raise ValueError("Invalid Action")

        if self.state[0] < self.minimum_fund and self.state[1] == 0:
            reward -= 1000  # penalty for running out of fund
            done = True

        next_cost = level_cost[self.state[1]]
        can_enhance = self.state[0] >= next_cost
        can_sell = self.state[1] >= self.minimum_sell_level
        if not can_enhance and not can_sell:
            done = True
            reward -= 1000 # penalty for no possible actions

        self.state[2] = next_cost

        if not done and self.current_step >= self.max_steps:
            truncated = True
            sell_price = self.sell()

        return self.state, reward, done, truncated, {}

    def render(self):
        print(f"Current Fund: {self.state[0]}G, Current Sword Level: +{self.state[1]}")

    def close(self):
        pass