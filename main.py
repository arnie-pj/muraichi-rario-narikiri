# -*- coding: utf_8 -*-
#!/usr/bin/env python3

import cv2
import numpy as np
import pyaudio
import threading
import time
import os
from PIL import Image, ImageDraw, ImageFont

# ===== 日本語フォント =====
_JP_FONT_PATHS = [
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/msmincho.ttc",
]
_font_cache = {}

def _get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    for p in _JP_FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f

def put_text_jp(frame, text, x, y, color_bgr=(220, 220, 220), size=14):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    draw.text((x, y), text, font=_get_font(size), fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ===== 設定 =====
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DISPLAY_WIDTH  = 600
DISPLAY_HEIGHT = 800
PENGUIN_SIZE   = 400
MOUTH_INTERVAL = 1 / 8   # 口アニメ間隔（秒）
MIC_THRESHOLD  = 200      # マイク感度閾値

# ===== 色定義 =====
PANEL_COLOR          = (60, 60, 60)
TEXT_COLOR           = (220, 220, 220)
CHECK_ACTIVE_COLOR   = (80, 200, 120)
CHECK_INACTIVE_COLOR = (100, 100, 100)
BORDER_COLOR         = (120, 120, 120)

# ===== グローバル状態 =====
state = {
    "mouth": "auto",   # auto, mic, open, close
}

mic_active   = False
mic_volume   = 0.0
mic_running  = False
mic_error    = False      # デバイスが見つからない／初期化失敗
mic_device   = 0          # 現在使用中のデバイスインデックス
mic_selected = 0          # UIで選択中のデバイスインデックス
mic_lock     = threading.Lock()

mouth_open = False

images = {}


def load_images():
    if not os.path.exists(IMAGE_DIR):
        print(f"画像フォルダが見つかりません: {IMAGE_DIR}")
        return
    for key in ("0", "1"):
        path = os.path.join(IMAGE_DIR, f"{key}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images[key] = img
        else:
            print(f"画像が読み込めません: {path}")


def get_image_key():
    mode = state["mouth"]
    if mode == "open":
        return "1"
    elif mode == "close":
        return "0"
    elif mode == "auto":
        return "1" if mouth_open else "0"
    elif mode == "mic":
        return "1" if (mic_active and mouth_open) else "0"
    return "0"


def mic_worker(device_index):
    """指定デバイスでマイク入力を監視するスレッド"""
    global mic_active, mic_volume, mic_running, mic_error
    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        with mic_lock:
            mic_error = False   # 開通したのでエラー解除
        while mic_running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16)
                vol = np.abs(arr).mean()
                with mic_lock:
                    mic_volume = vol
                    mic_active = vol > MIC_THRESHOLD
            except Exception:
                pass
        stream.stop_stream()
        stream.close()
        pa.terminate()
    except Exception as e:
        print(f"マイク初期化エラー (device={device_index}): {e}")
        with mic_lock:
            mic_error  = True
            mic_active = False
            mic_volume = 0.0
    finally:
        with mic_lock:
            mic_active = False
            mic_volume = 0.0


def start_mic(device_index):
    """マイクスレッドを起動する"""
    global mic_running, mic_device
    mic_running = True
    mic_device  = device_index
    t = threading.Thread(target=mic_worker, args=(device_index,), daemon=True)
    t.start()
    return t


def stop_mic():
    """マイクスレッドを停止する"""
    global mic_running, mic_error
    mic_running = False
    mic_error   = False
    time.sleep(0.15)   # スレッド終了を少し待つ


# ===== UI =====

class CheckBox:
    def __init__(self, label, x, y, w=20, h=20):
        self.label   = label
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.checked = False

    def draw(self, frame):
        color = CHECK_ACTIVE_COLOR if self.checked else CHECK_INACTIVE_COLOR
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), BORDER_COLOR, 1)
        if self.checked:
            cv2.rectangle(frame,
                          (self.x + 3, self.y + 3),
                          (self.x + self.w - 3, self.y + self.h - 3),
                          color, -1)
        put_text_jp(frame, self.label,
                    self.x + self.w + 4, self.y + 1,
                    color_bgr=TEXT_COLOR, size=14)

    def hit(self, mx, my):
        return (self.x <= mx <= self.x + self.w + len(self.label) * 9
                and self.y <= my <= self.y + self.h)


class RadioGroup:
    def __init__(self, label, options, x, y, spacing=90):
        self.label    = label
        self.options  = options
        self.x, self.y = x, y
        self.selected = 0
        self.boxes    = []
        cx = x + 80
        for opt in options:
            self.boxes.append(CheckBox(opt, cx, y, 18, 18))
            cx += spacing
        self.boxes[0].checked = True

    def draw(self, frame):
        put_text_jp(frame, self.label, self.x, self.y + 2,
                    color_bgr=TEXT_COLOR, size=14)
        for b in self.boxes:
            b.draw(frame)

    def click(self, mx, my):
        for i, b in enumerate(self.boxes):
            if b.hit(mx, my):
                self.selected = i
                for j, bb in enumerate(self.boxes):
                    bb.checked = (j == i)
                return True
        return False

    def get_value(self):
        return self.options[self.selected]


class MicSelector:
    """1～10のラジオボタンでマイクデバイスを選択する行"""
    def __init__(self, x, y, spacing=52):
        self.x, self.y = x, y
        self.selected  = 0   # 0-indexed → デバイス番号は1-indexed表示
        self.boxes     = []
        cx = x + 80
        for i in range(1, 11):
            self.boxes.append(CheckBox(str(i), cx, y, 18, 18))
            cx += spacing
        self.boxes[0].checked = True

    def draw(self, frame):
        put_text_jp(frame, "マイク", self.x, self.y + 2,
                    color_bgr=TEXT_COLOR, size=14)
        for b in self.boxes:
            b.draw(frame)

    def click(self, mx, my):
        """クリックされた場合 True を返す。選択が変わった場合のみ True"""
        for i, b in enumerate(self.boxes):
            if b.hit(mx, my) and i != self.selected:
                self.selected = i
                for j, bb in enumerate(self.boxes):
                    bb.checked = (j == i)
                return True
        return False

    def get_device_index(self):
        """PyAudio に渡すデバイスインデックス（0-indexed）を返す"""
        return self.selected


def build_ui():
    y0 = 12
    mic_selector = MicSelector(10, y0, spacing=52)
    y1 = y0 + 34
    mouth_group  = RadioGroup("口", ["自動", "マイク", "開く", "閉じる"], 10, y1, spacing=90)
    panel_y = y1 + 58   # 音量バー＋ERRORメッセージ分の余白を確保
    return mic_selector, mouth_group, panel_y


def render_penguin(frame, img_key, panel_y, display_w, display_h):
    if img_key not in images:
        img_key = next(iter(images), None)
        if img_key is None:
            return

    img  = images[img_key]
    size = min(display_w - 40, display_h - panel_y - 20, PENGUIN_SIZE)
    resized = cv2.resize(img, (size, size))

    px = (display_w - size) // 2
    py = panel_y + (display_h - panel_y - size) // 2

    frame[py:py + size, px:px + size] = (0, 255, 0)

    if resized.shape[2] == 4:
        alpha   = resized[:, :, 3:4].astype(np.float32) / 255.0
        rgb     = resized[:, :, :3].astype(np.float32)
        bg      = np.full((size, size, 3), (0, 255, 0), dtype=np.float32)
        blended = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        frame[py:py + size, px:px + size] = blended
    else:
        frame[py:py + size, px:px + size] = resized


def draw_mic_indicator(frame, display_w, panel_y, device_index):
    bar_w, bar_h = 180, 10
    bx = (display_w - bar_w) // 2
    by = panel_y - 28   # バー位置：パネル下端の少し上

    with mic_lock:
        vol    = mic_volume
        active = mic_active
        error  = mic_error

    # MIC#ラベルは左端固定（UIと被らない）
    label = f"MIC#{device_index + 1}"
    put_text_jp(frame, label, 10, by, color_bgr=TEXT_COLOR, size=12)

    if error:
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (0, 0, 160), -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (0, 0, 220), 1)
        msg = "ERROR: Microphone not found"
        cv2.putText(frame, msg,
                    (bx - 40, by + bar_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 60, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), BORDER_COLOR, 1)
        fill_w = int(bar_w * min(vol / 3000.0, 1.0))
        color  = (0, 200, 100) if active else (100, 100, 200)
        if fill_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), color, -1)


def main():
    global mic_running, mouth_open

    load_images()
    print(f"読み込んだ画像数: {len(images)}  キー: {sorted(images.keys())}")

    display_w = DISPLAY_WIDTH
    display_h = DISPLAY_HEIGHT

    mic_selector, mouth_group, panel_y = build_ui()

    # 初期マイク起動（デバイス0）
    start_mic(mic_selector.get_device_index())

    cv2.namedWindow("rario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rario", display_w, display_h)

    init_frame = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    cv2.imshow("rario", init_frame)
    cv2.waitKey(1)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # マイク選択が変わったらスレッドを再起動
            if mic_selector.click(mx, my):
                stop_mic()
                start_mic(mic_selector.get_device_index())
            mouth_group.click(mx, my)

    cv2.setMouseCallback("rario", on_mouse)

    last_mouth = time.time()

    while True:
        frame = np.full((display_h, display_w, 3), (0, 255, 0), dtype=np.uint8)

        cv2.rectangle(frame, (0, 0), (display_w, panel_y), PANEL_COLOR, -1)
        cv2.line(frame, (0, panel_y), (display_w, panel_y), BORDER_COLOR, 1)

        mic_selector.draw(frame)
        mouth_group.draw(frame)

        state["mouth"] = {
            "自動": "auto", "マイク": "mic",
            "開く": "open", "閉じる": "close"
        }[mouth_group.get_value()]

        now = time.time()
        if now - last_mouth >= MOUTH_INTERVAL:
            mouth_open = not mouth_open
            last_mouth = now

        draw_mic_indicator(frame, display_w, panel_y, mic_selector.get_device_index())

        img_key = get_image_key()
        render_penguin(frame, img_key, panel_y, display_w, display_h)

        cv2.imshow("rario", frame)

        k = cv2.waitKey(33)
        if k == 27 or k == ord('q'):
            break
        if cv2.getWindowProperty("rario", cv2.WND_PROP_VISIBLE) < 1:
            break

    stop_mic()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
