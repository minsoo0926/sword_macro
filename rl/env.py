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
        low_limits = np.array([0, 0, 0], dtype=np.int32) # fund, sword level, cost to enhance
        high_limits = np.array([1e8, 20, 1e6], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_limits, high=high_limits, shape=(3,), dtype=np.int32)
        self.max_steps = 1000
        self.current_step = 0
        self.target_fund = 1e6
        self.minimum_fund = 10000
        self.minimum_sell_level = 5
        self.reward_coeff = 0.001
        self.level_data = level_summary_dict

    def get_sell_price(self, level):
        sell_avg = self.level_data[level]['AVG_sell']
        sell_std = self.level_data[level]['STD_sell']
        sell_price = max(0, int(self.np_random.normal(sell_avg, sell_std)))
        return sell_price
    
    def sell(self):
        if self.state[1] == 0:
            return 0
        sell_price = self.get_sell_price(self.state[1])
        self.state[0] += sell_price
        self.state[1] = 0
        return sell_price

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([100000, 0, 10], dtype=np.int32)
        self.current_step = 0
        return self.state, {}
    
    def step(self, action):
        reward = 0.0
        done = False
        truncated = False

        # Enhance
        if action == 0 or self.state[1] < self.minimum_sell_level: 
            cost = level_cost[self.state[1]]
            if self.state[0] >= cost:
                self.state[0] -= cost
                # multinomial probability for success/remain/break
                success_prob = self.level_data[self.state[1]]['성공'] / 100
                remain_prob = self.level_data[self.state[1]]['유지'] / 100
                break_prob = self.level_data[self.state[1]]['파괴'] / 100
                outcome = self.np_random.choice(['success', 'remain', 'break'], p=[success_prob, remain_prob, break_prob])

                if outcome == 'success':
                    self.state[1] += 1
                    if self.state[1] > 12:
                        sell_price = self.sell()
                        reward += sell_price * self.reward_coeff    
                elif outcome == 'break':
                    self.state[1] = 0
            else:
                sell_price = self.sell()
                reward += sell_price * self.reward_coeff
        # Sell
        elif action == 1:
            sell_price = self.sell()
            reward = sell_price * self.reward_coeff

        if action not in [0, 1]:
            raise ValueError("Invalid Action")

        if self.state[0] >= self.target_fund:
            reward += 1000  # bonus for reaching target fund
            done = True
        elif self.state[0] < self.minimum_fund and self.state[1] == 0:
            reward -= 1000  # penalty for running out of fund
            done = True

        self.current_step += 1
        self.state[2] = level_cost[self.state[1]]

        if self.current_step >= self.max_steps or done:
            truncated = True
            sell_price = self.sell()
            reward += sell_price * self.reward_coeff
        
        return self.state, reward, done, truncated, {}

    def render(self):
        print(f"Current Fund: {self.state[0]}G, Current Sword Level: +{self.state[1]}")

    def close(self):
        pass