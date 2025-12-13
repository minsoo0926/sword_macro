## Sword Growing RL/Macro

### Description
- Data parsing tool for Sword Growing game in KakaoTalk Gamebot
- RL environment to build agent
- Macro support

### Quick start
```
pip install -r requirement.txt
make macro
```

### RL support
- Used PPO algorithm using SB3
- Makefile script for convenient training
    - `Make train`: start training
    - `Make test`: test 1000 timesteps and plot

### Data Parsing
1. Export Kakaotalk chat log
2. run `process_data.ipynb`

### AI/Heuristic inference support
- Trained RL agent is used for inference
- Rule-based policy
    - Sell when `LEVEL_THRESHOLD` achieved
    - refer to `rl/config.py`

### Macro (Hotkey) Support
- `F1`: 강화
- `F2`: 판매
- `F3`: AI inference (loop)
- `F4`: Heuristic inference (loop)
- `F5`: 매크로 종료
