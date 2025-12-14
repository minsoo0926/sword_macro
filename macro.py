from pynput import keyboard, mouse
import time
import pyperclip
import re
import threading
from rl.inference import SwordAI
from rl.config import CHAT_OUTPUT_COORD, CHAT_INPUT_COORD, ACTION_DELAY

is_running = True
pressed_keys = set()
controller = keyboard.Controller()
mouse_controller = mouse.Controller()
ai = SwordAI()

running_mode = None  # 'ai' or 'heuristic' or None
fail_count = 0

def worker_loop():
    global running_mode
    while True:
        if running_mode == 'ai':
            act_inference('ai')
            time.sleep(ACTION_DELAY)  # wait before next inference
        elif running_mode == 'heuristic':
            act_inference('heuristic')
            time.sleep(ACTION_DELAY)
        else:
            time.sleep(0.1)

t = threading.Thread(target=worker_loop, daemon=True)
t.start()

def _click_mouse(x, y):
    mouse_controller.position = (x, y)
    time.sleep(0.1)
    mouse_controller.click(mouse.Button.left, 1)
    time.sleep(0.1)

def _copy_message():
    _click_mouse(*CHAT_OUTPUT_COORD)
    controller.press(keyboard.Key.cmd)
    controller.press('a')
    time.sleep(0.1)
    controller.release('a')
    time.sleep(0.1)
    controller.press('c')
    time.sleep(0.1)
    controller.release('c')
    controller.release(keyboard.Key.cmd)
    time.sleep(0.1)
    _click_mouse(*CHAT_INPUT_COORD)
    # controller.press(keyboard.Key.esc)
    # controller.release(keyboard.Key.esc)
    # time.sleep(0.1)
    text = pyperclip.paste()
    # text = _parse_message(text)
    # print(text)
    return text

def _parse_message(message):
    global fail_count
    message = message.split('@')[-1]
    enhance_pattern = re.findall(r'강화 (\w+)', message)
    result = enhance_pattern[0] if enhance_pattern else None

    if result == '유지':
        fail_count += 1
    else:
        fail_count = 0

    print("="*30)
    print(message)
    print("="*30)
    level_pattern = re.findall(r'\+(\d+)', message)
    level = int(level_pattern[-1]) if level_pattern \
        else 0 if result == '파괴' else None
    gold_pattern = re.findall(r'(?:남은|현재\s*보유|보유)\s*골드:\s*([\d,]+)\s*G', message)
    fund = int(gold_pattern[0].replace(',', '')) if gold_pattern else None    
        
    return fund, level

def act_enhance():
    # print("강화 매크로 실행")
    
    controller.press('/')
    time.sleep(0.2)
    controller.press('ㄱ')
    time.sleep(0.2)
    controller.press(keyboard.Key.enter)
    time.sleep(0.2)
    controller.press(keyboard.Key.enter)

def act_sell():
    # print("판매 매크로 실행")
    
    controller.press('/')
    time.sleep(0.2)
    controller.press('ㅍ')
    time.sleep(0.2)
    controller.press(keyboard.Key.enter)
    time.sleep(0.2)
    controller.press(keyboard.Key.enter)

def act_inference(mode='ai'):
    text = _copy_message()
    fund, level = _parse_message(text)
    
    if fund is None or level is None:
        print("Unable to parse fund or level from message.")
        return
    if mode == 'ai':
        print(f"Current Fund: {fund}, Level: {level}, Fail Count: {fail_count}")
        inference_result = ai.predict(fund, level, fail_count)
    else:
        print(f"Current Fund: {fund}, Level: {level}, Fail Count: {fail_count}")
        inference_result = ai.heuristic(fund, level, fail_count)

    print(f"{mode.capitalize()} Inference Result (0: 강화, 1: 판매, -1: 행동 불가): {inference_result}")
    if inference_result == 0:
        act_enhance()
    elif inference_result == 1:
        act_sell()
    else:
        print("No valid action can be taken.")

def on_press(key):
    global running_mode
    try:
        if key in pressed_keys:
            return
        pressed_keys.add(key)
        
        if key == keyboard.Key.f1:
            act_enhance()
        elif key == keyboard.Key.f2:
            act_sell()
        elif key == keyboard.Key.f3:
            running_mode = 'ai'
        elif key == keyboard.Key.f4:
            running_mode = 'heuristic'
        elif key == keyboard.Key.f5:
            running_mode = None
            print("Bye!")
            return False
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key)
    except:
        pass

if __name__ == "__main__":
    print("매크로 실행 중... (F1: 강화, F2: 판매, F3: AI 추론 toggle, F4: heuristic toggle, F5: 종료)")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()