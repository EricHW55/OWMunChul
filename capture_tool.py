import os
import time
from datetime import datetime

import tkinter as tk
from tkinter import messagebox

import mss
import mss.tools
import keyboard
import pygetwindow as gw


# ---------------- 기본 설정 ----------------
BASE_DIR = "dataset"
BLUE_DIR = os.path.join(BASE_DIR, "blue")
RED_DIR = os.path.join(BASE_DIR, "red")

os.makedirs(BLUE_DIR, exist_ok=True)
os.makedirs(RED_DIR, exist_ok=True)


# ---------------- 기능 함수들 ----------------
def focus_overwatch():
    """
    '오버워치' 라는 이름을 가진 창을 찾아서 포커스를 줌.
    여러 개면 첫 번째 것 사용.
    못 찾으면 False 리턴.
    """
    windows = gw.getWindowsWithTitle("오버워치")
    if not windows:
        print("[WARN] '오버워치' 창을 찾지 못했습니다.")
        return False

    win = windows[0]
    try:
        if win.isMinimized:
            win.restore()
        win.activate()
        # 포커스가 넘어갈 시간을 조금 줌
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"[ERROR] 오버워치 창 활성화 실패: {e}")
        return False


def bring_ui_front(root: tk.Tk):
    """Tk 창을 다시 맨 앞으로 올리기"""
    try:
        root.lift()
        root.attributes("-topmost", True)
        # 잠깐 topmost로 올린 뒤 다시 해제 (다른 창 사용 방해 안 하게)
        root.after(300, lambda: root.attributes("-topmost", False))
    except Exception as e:
        print(f"[ERROR] Tk 창 앞으로 가져오기 실패: {e}")


def capture_screen(team: str, root: tk.Tk | None = None):
    """
    team: 'blue' or 'red'
    Overwatch 창으로 포커스 넘긴 뒤 전체 모니터를 캡쳐해서 저장
    캡쳐 후 Tk UI를 다시 앞으로 가져옴 (root가 주어졌을 때)
    """
    ok = focus_overwatch()
    if not ok:
        print("[INFO] 오버워치 창이 없어 캡쳐를 취소합니다.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = team
    save_dir = BLUE_DIR if team == "blue" else RED_DIR
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)

    with mss.mss() as sct:
        # 모니터 1 전체 (멀티 모니터면 필요시 인덱스 바꿔도 됨)
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        mss.tools.to_png(img.rgb, img.size, output=filepath)

    print(f"[CAPTURE] {team.upper()} 저장: {filepath}")

    # 캡쳐 후 Tk 창 다시 앞으로
    if root is not None:
        bring_ui_front(root)


# ---------------- Tkinter UI ----------------
def main():
    root = tk.Tk()
    root.title("OW 캡쳐 도우미")

    # 창 크기/위치 적당히
    root.geometry("300x160")

    info_label = tk.Label(
        root,
        text=(
            "Overwatch 스코어보드 캡쳐\n\n"
            "BLUE : Ctrl+Alt+B\n"
            "RED  : Ctrl+Alt+R\n\n"
            "버튼을 눌러도 캡쳐됩니다."
        ),
        justify="left"
    )
    info_label.pack(pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)

    blue_btn = tk.Button(
        btn_frame,
        text="BLUE 캡쳐",
        width=12,
        command=lambda: capture_screen("blue", root)
    )
    blue_btn.grid(row=0, column=0, padx=5)

    red_btn = tk.Button(
        btn_frame,
        text="RED 캡쳐",
        width=12,
        command=lambda: capture_screen("red", root)
    )
    red_btn.grid(row=0, column=1, padx=5)

    # 종료 버튼
    quit_btn = tk.Button(
        root,
        text="종료",
        width=10,
        command=root.destroy
    )
    quit_btn.pack(pady=5)

    # 전역 핫키 등록 (백그라운드에서 동작)
    keyboard.add_hotkey("ctrl+alt+b", lambda: capture_screen("blue", root))
    keyboard.add_hotkey("ctrl+alt+r", lambda: capture_screen("red", root))

    print("=== OW 캡쳐 도우미 실행 중 ===")
    print("Ctrl+Alt+B : BLUE 캡쳐 (dataset/blue)")
    print("Ctrl+Alt+R : RED 캡쳐 (dataset/red)")
    print("Tk 창의 버튼을 클릭해도 동일하게 캡쳐됩니다.")
    print("창을 닫으면 프로그램이 종료됩니다.")

    # Tk 메인 루프
    root.mainloop()

    keyboard.unhook_all_hotkeys()
    print("프로그램 종료.")


if __name__ == "__main__":
    main()
