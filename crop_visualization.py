import cv2
import numpy as np

# ---------------------------------
# 0. 기준 해상도 & 네가 찍은 ROI 좌표
#    (2048x1151 기준으로 드래그한 값)
# ---------------------------------
REF_W, REF_H = 2048, 1151

# 파란 팀 맨 위 영웅 아이콘 영역
HERO_X0, HERO_Y0 = 602.64, 230
HERO_X1, HERO_Y1 = 670.66, 298

# 처치 / 도움 / 죽음 묶음 (같은 줄 기준)
KDA_X0, KDA_Y0 = 970.78, 230
KDA_X1, KDA_Y1 = 1146.85, 298

# 피해 / 치유 / 경감 묶음 (같은 줄 기준)
DMG_X0, DMG_Y0 = 1147.65, 230
DMG_X1, DMG_Y1 = 1478.98, 298

N_BLUE_ROWS = 5  # 위 팀 5명
N_RED_ROWS  = 5  # 아래 팀 5명

# 빨간 팀 맨 위 영웅 아이콘 y0 (네가 새로 뽑은 값)
RED_HERO_Y0 = 660.69
# 높이는 파란 팀이랑 동일하게 사용
RED_HERO_Y1 = RED_HERO_Y0 + (HERO_Y1 - HERO_Y0)


# ---------------------------------
# 1. 좌표 정규화 / 복원 함수
# ---------------------------------
def norm_roi(x0, y0, x1, y1):
    """기준 해상도(REF_W, REF_H)에 대해 0~1 비율로 변환"""
    return x0 / REF_W, y0 / REF_H, x1 / REF_W, y1 / REF_H


def denorm_roi(roi_n, img_w, img_h):
    """정규화된 ROI를 실제 이미지 크기로 복원"""
    x0_n, y0_n, x1_n, y1_n = roi_n
    x0 = int(x0_n * img_w)
    y0 = int(y0_n * img_h)
    x1 = int(x1_n * img_w)
    y1 = int(y1_n * img_h)
    return x0, y0, x1, y1


def split_three(x0, x1):
    """가로 방향을 1/3씩 세 칸으로 나누기"""
    w = x1 - x0
    step = w / 3.0
    xs = []
    for i in range(3):
        sx0 = int(x0 + i * step)
        sx1 = int(x0 + (i + 1) * step)
        xs.append((sx0, sx1))
    return xs  # [(x0_0,x1_0), (x0_1,x1_1), (x0_2,x1_2)]


# 기준 ROI를 비율로 저장 (파란팀 / 빨간팀)
HERO_N      = norm_roi(HERO_X0,      HERO_Y0,      HERO_X1,      HERO_Y1)
RED_HERO_N  = norm_roi(HERO_X0,      RED_HERO_Y0,  HERO_X1,      RED_HERO_Y1)
KDA_N       = norm_roi(KDA_X0,       KDA_Y0,       KDA_X1,       KDA_Y1)
DMG_N       = norm_roi(DMG_X0,       DMG_Y0,       DMG_X1,       DMG_Y1)


# ---------------------------------
# 2. 메인: 파란 + 빨간 팀 박스 그려서 디버그
# ---------------------------------
if __name__ == "__main__":
    img_path = r"dataset/blue/2025-11-25 030028.png"  # 스샷 경로
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")

    H, W = img.shape[:2]
    print("image size:", W, H)

    # 기준 ROI들을 현재 이미지 크기에 맞게 스케일링
    hero_x0, hero_y0, hero_x1, hero_y1         = denorm_roi(HERO_N,     W, H)
    red_hero_x0, red_hero_y0, red_hero_x1, red_hero_y1 = denorm_roi(RED_HERO_N, W, H)
    kda_x0,  kda_y0,  kda_x1,  kda_y1          = denorm_roi(KDA_N,      W, H)
    dmg_x0,  dmg_y0,  dmg_x1,  dmg_y1          = denorm_roi(DMG_N,      W, H)

    # 한 행의 높이 = 영웅 셀 전체 높이 (파란 팀 기준)
    row_h = hero_y1 - hero_y0

    # 숫자 줄(y) 위치는 KDA 박스 기준으로 조금 내려간 곳 (파란 팀 기준)
    num_y_offset = kda_y0 - hero_y0        # 영웅 셀 위쪽에서 숫자 줄까지 오프셋
    num_box_h    = kda_y1 - kda_y0         # 숫자 줄의 실제 높이

    # 가로 방향 1/3 분할 (K/A/D, 피해/치유/경감)
    kda_cols = split_three(kda_x0, kda_x1)  # [kills, assists, deaths]
    dmg_cols = split_three(dmg_x0, dmg_x1)  # [damage, heal, mitig]

    debug = img.copy()

    # 색 정의 (BGR)
    COLOR_HERO_BLUE = (255, 255, 0)
    COLOR_HERO_RED  = (0, 165, 255)
    COLOR_KILL = (0, 255, 0)
    COLOR_AST  = (0, 200, 255)
    COLOR_DEAD = (0, 0, 255)
    COLOR_DMG  = (0, 255, 255)
    COLOR_HEAL = (255, 0, 255)
    COLOR_MIT  = (255, 255, 255)

    # -------- 파란 팀 5명 --------
    for i in range(N_BLUE_ROWS):
        hero_row_y0 = int(hero_y0 + i * row_h)
        hero_row_y1 = int(hero_row_y0 + row_h)

        num_row_y0 = int(hero_row_y0 + num_y_offset)
        num_row_y1 = int(num_row_y0 + num_box_h)

        # 영웅 아이콘
        cv2.rectangle(debug, (hero_x0, hero_row_y0), (hero_x1, hero_row_y1),
                      COLOR_HERO_BLUE, 2)

        # K / A / D
        kx0, kx1 = kda_cols[0]
        ax0, ax1 = kda_cols[1]
        dx0, dx1 = kda_cols[2]

        cv2.rectangle(debug, (kx0, num_row_y0), (kx1, num_row_y1), COLOR_KILL, 2)
        cv2.rectangle(debug, (ax0, num_row_y0), (ax1, num_row_y1), COLOR_AST,  2)
        cv2.rectangle(debug, (dx0, num_row_y0), (dx1, num_row_y1), COLOR_DEAD, 2)

        # 피해 / 치유 / 경감
        dmgx0, dmgx1   = dmg_cols[0]
        healx0, healx1 = dmg_cols[1]
        mitx0, mitx1   = dmg_cols[2]

        cv2.rectangle(debug, (dmgx0,  num_row_y0), (dmgx1,  num_row_y1), COLOR_DMG,  2)
        cv2.rectangle(debug, (healx0, num_row_y0), (healx1, num_row_y1), COLOR_HEAL, 2)
        cv2.rectangle(debug, (mitx0,  num_row_y0), (mitx1, num_row_y1), COLOR_MIT,  2)

    # -------- 빨간 팀 5명 --------
    for i in range(N_RED_ROWS):
        hero_row_y0 = int(red_hero_y0 + i * row_h)
        hero_row_y1 = int(hero_row_y0 + row_h)

        num_row_y0 = int(hero_row_y0 + num_y_offset)
        num_row_y1 = int(num_row_y0 + num_box_h)

        # 영웅 아이콘
        cv2.rectangle(debug, (hero_x0, hero_row_y0), (hero_x1, hero_row_y1),
                      COLOR_HERO_RED, 2)

        # K / A / D
        kx0, kx1 = kda_cols[0]
        ax0, ax1 = kda_cols[1]
        dx0, dx1 = kda_cols[2]

        cv2.rectangle(debug, (kx0, num_row_y0), (kx1, num_row_y1), COLOR_KILL, 2)
        cv2.rectangle(debug, (ax0, num_row_y0), (ax1, num_row_y1), COLOR_AST,  2)
        cv2.rectangle(debug, (dx0, num_row_y0), (dx1, num_row_y1), COLOR_DEAD, 2)

        # 피해 / 치유 / 경감
        dmgx0, dmgx1   = dmg_cols[0]
        healx0, healx1 = dmg_cols[1]
        mitx0, mitx1   = dmg_cols[2]

        cv2.rectangle(debug, (dmgx0,  num_row_y0), (dmgx1,  num_row_y1), COLOR_DMG,  2)
        cv2.rectangle(debug, (healx0, num_row_y0), (healx1, num_row_y1), COLOR_HEAL, 2)
        cv2.rectangle(debug, (mitx0,  num_row_y0), (mitx1, num_row_y1), COLOR_MIT,  2)

    cv2.imshow("debug_boxes_all", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
