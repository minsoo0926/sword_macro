## Sword Growing RL/Macro

### Description
- Data parsing tool for Sword Growing game in KakaoTalk Gamebot
- RL environment to build agent
- Macro support

### RL support
- Used PPO algorithm using SB3
- Makefile script for convenient training
    - `Make train`: start training
    - `Make test`: test 1000 timesteps and plot

### Data Parsing
1. Export Kakaotalk chat log
2. run `process_data.ipynb`

### Macro (Hotkey) Support
- `F1`: 강화
- `F2`: 판매
- `F3`: 매크로 종료
