from pynput import keyboard
import time

is_running = True
pressed_keys = set()
controller = keyboard.Controller()


def act_enhance():
    print("강화 매크로 실행")
    
    controller.press('/')
    time.sleep(0.5)
    controller.press('ㄱ')
    time.sleep(0.5)
    controller.press(keyboard.Key.enter)
    time.sleep(0.5)
    controller.press(keyboard.Key.enter)

def act_sell():
    print("판매 매크로 실행")
    
    controller.press('/')
    time.sleep(0.5)
    controller.press('ㅍ')
    time.sleep(0.5)
    controller.press(keyboard.Key.enter)
    time.sleep(0.5)
    controller.press(keyboard.Key.enter)

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
            return False
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key)
    except:
        pass

if __name__ == "__main__":
    print("매크로 실행 중... (F1: 강화, F2: 판매, F3: 종료)")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()