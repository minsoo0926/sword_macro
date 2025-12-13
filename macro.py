from pynput import keyboard, mouse
import time
import pyperclip
import re
from rl.inference import SwordAI

is_running = True
pressed_keys = set()
controller = keyboard.Controller()
mouse_controller = mouse.Controller()
ai = SwordAI()

def _click_mouse(x, y):
    mouse_controller.position = (x, y)
    time.sleep(0.1)
    mouse_controller.click(mouse.Button.left, 1)
    time.sleep(0.1)

def _copy_message():
    _click_mouse(200, 550)
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
    _click_mouse(200, 900)
    # controller.press(keyboard.Key.esc)
    # controller.release(keyboard.Key.esc)
    # time.sleep(0.1)
    text = pyperclip.paste()
    # text = _parse_message(text)
    # print(text)
    return text

def _parse_message(message):
    message = message.split('@')[-1]
    enhance_pattern = re.findall(r'강화 (\w+)', message)
    result = enhance_pattern[0] if enhance_pattern else None
    print("="*30)
    print(message)
    print("="*30)
    level_pattern = re.findall(r'\+(\d+)', message)
    level = int(level_pattern[-1]) if level_pattern \
        else 0 if result == '파괴' else None
    gold_pattern = re.findall(r'(?:남은|현재\s*보유)\s*골드: ([\d,]+)G', message)
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
    print(f"Current Fund: {fund}, Level: {level}")
    if fund is None or level is None:
        print("Unable to parse fund or level from message.")
        return
    if mode == 'ai':
        inference_result = ai.predict(fund, level)
    else:
        inference_result = ai.heuristic(fund, level)

    print(f"AI Inference Result (0: 강화, 1: 판매, -1: 행동 불가): {inference_result}")
    if inference_result == 0:
        act_enhance()
    elif inference_result == 1:
        act_sell()
    else:
        print("No valid action can be taken.")

def on_press(key):
    try:
        if key in pressed_keys:
            return
        pressed_keys.add(key)
        
        if key == keyboard.Key.f1:
            act_enhance()
        elif key == keyboard.Key.f2:
            act_sell()
        elif key == keyboard.Key.f3:
            while True:
                act_inference('ai')
                time.sleep(3)  # wait before next inference
        elif key == keyboard.Key.f4:
            while True:
                act_inference(mode='heuristic')
                time.sleep(3)  # wait before next inference
        elif key == keyboard.Key.f5:
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