import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from rl.env import SwordEnv, level_cost
from rl.config import *

class SwordAI:
    def __init__(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        print("Loading AI Model & Stats...")
        self.dummy_env = DummyVecEnv([lambda: SwordEnv()])
        
        try:
            self.vec_norm = VecNormalize.load(stats_path, self.dummy_env)
            self.vec_norm.training = False 
            self.vec_norm.norm_reward = False
        except FileNotFoundError:
            raise Exception("cannot find VecNormalize stats file (.pkl)!")

        try:
            self.model = MaskablePPO.load(model_path)
        except FileNotFoundError:
            raise Exception("cannot find model file (.zip)!")
            
        print("AI Ready!")

    def _normalize_obs(self, raw_obs):
        obs_tensor = self.vec_norm.normalize_obs(np.array([raw_obs]))
        return obs_tensor

    def _get_mask(self, fund, level):
        cost = level_cost.get(level, 0)
        can_enhance = fund >= cost
        can_sell = level >= 5
        
        return np.array([can_enhance, can_sell])

    def predict(self, fund: int, level: int):
        cost = level_cost.get(level, 0)
        raw_obs = np.array([fund, level, cost], dtype=np.int32)
        action_masks = self._get_mask(fund, level)

        if not any(action_masks):
            return -1
            
        norm_obs = self._normalize_obs(raw_obs)
        
        action, _ = self.model.predict(
            norm_obs, 
            action_masks=action_masks, 
            deterministic=True
        )
        
        return int(action[0])

if __name__ == "__main__":
    ai = SwordAI(MODEL_PATH, STATS_PATH)

    test_cases = [
        (500000, 0),   # 돈 많음, 0강 -> 당연히 강화(0)
        (50, 5),      # 돈 없음, 5강 -> 팔아야 함(1)
        (1000000, 5),  # 돈 많음, 5강 -> AI의 선택은? (강화/판매)
        (10, 1),       # 돈 없음, 0강 -> 행동 불능(-1)
    ]

    print("\n=== AI Inference Test ===")
    for fund, level in test_cases:
        action = ai.predict(fund, level)
        
        act_str = "강화(Enhance)" if action == 0 else \
                  "판매(Sell)" if action == 1 else "행동 불가(Dead End)"
                  
        print(f"상태 [돈: {fund:>7}, 레벨: {level:>2}] -> AI 선택: {act_str}")