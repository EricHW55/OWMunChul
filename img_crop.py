import cv2
import numpy as np

# -----------------------------
# 1. (기존 상수들은 일단 냅둬도 되고, 나중에 채워넣을 때 씀)
# -----------------------------

# Hero
# BOARD_X0 = 602.64
# BOARD_Y0 = 230.36
# BOARD_X1 = 670.66
# BOARD_Y1 = 297.55

# 처치 도움 죽음
# BOARD_X0 = 970.78
# BOARD_Y0 = 230.36
# BOARD_X1 = 1146.85
# BOARD_Y1 = 297.55

# 킬 어시 데스
# BOARD_X0 = 1147.65
# BOARD_Y0 = 230.36
# BOARD_X1 = 1478.98
# BOARD_Y1 = 297.55


REF_W, REF_H = 2048, 1151          # 기준 스샷 해상도 (원하면 바꿔도 됨)

# -----------------------------
# 2. 메인: 드래그해서 좌표 얻기
# -----------------------------

if __name__ == "__main__":
    img_path = r"dataset/blue/2025-11-25 030028.png"  # 실제 파일 경로
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")

    h, w = img.shape[:2]
    print(f"이미지 크기: {w} x {h}")

    # 윈도우 띄우고 ROI 선택 (마우스로 드래그 후 엔터 / 스페이스, ESC는 취소)
    roi = cv2.selectROI("Drag ROI and press ENTER", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w_roi, h_roi = map(int, roi)
    x0, y0 = x, y
    x1, y1 = x + w_roi, y + h_roi

    print("\n=== 픽셀 좌표 ===")
    print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")

    print("\n=== 현재 이미지 기준 비율 ===")
    print(f"x0_n: {x0 / w:.6f}, y0_n: {y0 / h:.6f}")
    print(f"x1_n: {x1 / w:.6f}, y1_n: {y1 / h:.6f}")

    # 기준 해상도(REF_W, REF_H)를 쓰고 싶으면 이렇게도 출력 가능
    print("\n=== 기준 해상도(REF_W, REF_H) 기준 환산 좌표 ===")
    print(f"BOARD_X0 = {x0 / w * REF_W:.2f}")
    print(f"BOARD_Y0 = {y0 / h * REF_H:.2f}")
    print(f"BOARD_X1 = {x1 / w * REF_W:.2f}")
    print(f"BOARD_Y1 = {y1 / h * REF_H:.2f}")

    # 선택 영역이 제대로 맞는지 확인하고 싶으면:
    board = img[y0:y1, x0:x1].copy()
    cv2.imshow("Cropped", board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
