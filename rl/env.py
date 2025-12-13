from gymnasium import Env, spaces
import numpy as np
import pandas as pd

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
    12: 50000
}

class SwordEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        low_limits = np.array([0, 0], dtype=np.int32)
        high_limits = np.array([1e8, 20], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_limits, high=high_limits, shape=(2,), dtype=np.int32)
        self.max_steps = 1000
        self.current_step = 0
        self.target_fund = 1e7
        self.minimum_fund = 1000
        self.level_data = level_summary_dict

    def get_sell_price(self, level):
        sell_avg = self.level_data[level]['AVG_sell']
        sell_std = self.level_data[level]['STD_sell']
        sell_price = max(0, int(np.random.normal(sell_avg, sell_std)))
        return sell_price

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.state = np.array([100000, 0], dtype=np.int32)
        self.current_step = 0
        return self.state, {}
    
    def step(self, action):
        # Enhance
        if action == 0: 
            cost = level_cost[self.state[1]]
            done = False
            if self.state[0] >= cost:
                self.state[0] -= cost
                reward = -cost / 1000
                # multinomial probability for success/remain/break
                success_prob = self.level_data[self.state[1]]['성공'] / 100
                remain_prob = self.level_data[self.state[1]]['유지'] / 100
                break_prob = self.level_data[self.state[1]]['파괴'] / 100
                outcome = np.random.choice(['success', 'remain', 'break'], p=[success_prob, remain_prob, break_prob])
                if outcome == 'success':
                    self.state[1] += 1
                    reward = 0 # ignore cost penalty on success
                    if self.state[1] > 12:
                        sell_price = self.get_sell_price(12)
                        self.state[0] += sell_price
                        reward += sell_price / 1000
                        self.state[1] = 0

                elif outcome == 'break':
                    self.state[1] = 0
                    reward -= 10
                    if self.state[0] < self.minimum_fund:
                        reward -= 90  # additional penalty for low fund after break
                        done = True

                self.current_step += 1
                done = bool(self.state[0] >= self.target_fund) or done
                truncated = self.current_step >= self.max_steps
            else:
                # give penalty for invalid action
                done = False
                truncated = False
                reward = -10
                
        # Sell
        elif action == 1:
            if self.state[1] == 0:
                # give penalty for invalid action
                done = False
                truncated = False
                reward = -10
                return self.state, reward, done, truncated, {}
            sell_price = self.get_sell_price(self.state[1])
            self.state[0] += sell_price
            self.state[1] = 0

            self.current_step += 1
            done = bool(self.state[0] >= self.target_fund)
            truncated = self.current_step >= self.max_steps
            reward = sell_price / 1000

        if action not in [0, 1]:
            raise ValueError("Invalid Action")

        return self.state, reward, done, truncated, {}

    def render(self):
        print(f"Current Fund: {self.state[0]}G, Current Sword Level: +{self.state[1]}")

    def close(self):
        pass