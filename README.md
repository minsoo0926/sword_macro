## Sword Growing RL/Macro

### Description
- end-to-end Data parsing / RL / macro tool for Sword Growing (검키우기) game in KakaoTalk Gamebot
- RL environment to build agent
- Macro support (on Macbook yet)

### Quick start
0. Preliminaries: `uv`
1. Sync your python env
    ```bash
    make init
    ```
1. Set rl/config.py
    - `CHAT_OUTPUT_COORD`: coordinate of chatbot output
    - `CHAT_INPUT_COORD`: coordinate of your input box
2. Run macro
    ```bash
    make macro
    ```
3. Functions
    - `F1`, `F2` is hotkey for "강화", "판매" respectively
    - `F3` starts loop based on AI inference
    - `F4` starts loop based on rule-based strategy
    - `F5` quits program

### RL support
- Used PPO algorithm using SB3
- Makefile script for convenient training
    - `Make train`: start training
    - `Make test`: test 1000 timesteps and plot

### Data Parsing
1. Export Kakaotalk chat log
2. put .csv file to `./data/*`
3. run `process_data.ipynb`

### AI/Heuristic inference support
- Trained RL agent is used for inference
- Rule-based policy
    - If `LEVEL_THRESHOLD` achieved, if fail count surpasses `FAIL_COUNT_THRESHOLD`, sell current sword
    - refer to `rl/config.py`

### Macro (Hotkey) Support
- `F1`: 강화
- `F2`: 판매
- `F3`: AI inference (loop)
- `F4`: Heuristic inference (loop)
- `F5`: 매크로 종료

### TODOS
- Windows support
- More training
- More data collection (for better environment modeling)