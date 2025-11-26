import cv2
import numpy as np


class OWScoreboardCropper:
    def __init__(self):
        # 기준 해상도
        self.REF_W, self.REF_H = 2048, 1151

        # 파란 팀 맨 위 영웅 아이콘 영역 (기준 해상도 기준)
        self.HERO_X0, self.HERO_Y0 = 602.64, 230
        self.HERO_X1, self.HERO_Y1 = 670.66, 298

        # 처치 / 도움 / 죽음 묶음 (같은 줄 기준)
        self.KDA_X0, self.KDA_Y0 = 970.78, 230
        self.KDA_X1, self.KDA_Y1 = 1146.85, 298

        # 피해 / 치유 / 경감 묶음 (같은 줄 기준)
        self.DMG_X0, self.DMG_Y0 = 1147.65, 230
        self.DMG_X1, self.DMG_Y1 = 1478.98, 298

        # 빨간 팀 맨 위 영웅 아이콘 y0 (네가 드래그한 값)
        self.RED_HERO_Y0 = 660.69
        self.RED_HERO_Y1 = self.RED_HERO_Y0 + (self.HERO_Y1 - self.HERO_Y0)

        self.N_BLUE_ROWS = 5
        self.N_RED_ROWS = 5

        # 정규화된 ROI 미리 계산
        self.HERO_N = self._norm_roi(self.HERO_X0, self.HERO_Y0,
                                     self.HERO_X1, self.HERO_Y1)
        self.RED_HERO_N = self._norm_roi(self.HERO_X0, self.RED_HERO_Y0,
                                         self.HERO_X1, self.RED_HERO_Y1)
        self.KDA_N = self._norm_roi(self.KDA_X0, self.KDA_Y0,
                                    self.KDA_X1, self.KDA_Y1)
        self.DMG_N = self._norm_roi(self.DMG_X0, self.DMG_Y0,
                                    self.DMG_X1, self.DMG_Y1)

    # ---------- 기본 유틸 ----------

    def _norm_roi(self, x0, y0, x1, y1):
        return (x0 / self.REF_W, y0 / self.REF_H,
                x1 / self.REF_W, y1 / self.REF_H)

    def _denorm_roi(self, roi_n, img_w, img_h):
        x0_n, y0_n, x1_n, y1_n = roi_n
        x0 = int(x0_n * img_w)
        y0 = int(y0_n * img_h)
        x1 = int(x1_n * img_w)
        y1 = int(y1_n * img_h)
        return x0, y0, x1, y1

    def _split_three(self, x0, x1):
        w = x1 - x0
        step = w / 3.0
        xs = []
        for i in range(3):
            sx0 = int(x0 + i * step)
            sx1 = int(x0 + (i + 1) * step)
            xs.append((sx0, sx1))
        return xs  # [(x0_0,x1_0), (x0_1,x1_1), (x0_2,x1_2)]

    # ---------- 메인: 좌표 계산 ----------

    def get_player_boxes(self, img):
        """
        img: BGR 이미지 (numpy array)
        return:
            {
              "blue": [ {slot1 dict}, ..., {slot5 dict} ],
              "red":  [ {slot1 dict}, ..., {slot5 dict} ]
            }

        각 slot dict:
            {
              "hero":   (x0, y0, x1, y1),
              "kills":  (..),
              "assists":(..),
              "deaths": (..),
              "damage": (..),
              "heal":   (..),
              "mitig":  (..)
            }
        """
        h, w = img.shape[:2]

        # 기준 ROI를 현재 이미지 크기로 복원
        hero_x0, hero_y0, hero_x1, hero_y1 = self._denorm_roi(self.HERO_N, w, h)
        red_hero_x0, red_hero_y0, red_hero_x1, red_hero_y1 = self._denorm_roi(
            self.RED_HERO_N, w, h
        )
        kda_x0, kda_y0, kda_x1, kda_y1 = self._denorm_roi(self.KDA_N, w, h)
        dmg_x0, dmg_y0, dmg_x1, dmg_y1 = self._denorm_roi(self.DMG_N, w, h)

        # 행 높이 및 숫자줄 오프셋
        row_h = hero_y1 - hero_y0
        num_y_offset = kda_y0 - hero_y0
        num_box_h = kda_y1 - kda_y0

        # 가로 방향 1/3 분할
        kda_cols = self._split_three(kda_x0, kda_x1)  # [kills, assists, deaths]
        dmg_cols = self._split_three(dmg_x0, dmg_x1)  # [damage, heal, mitig]

        result = {
            "blue": [],
            "red": []
        }

        # ----- 파란 팀 -----
        for i in range(self.N_BLUE_ROWS):
            hero_row_y0 = int(hero_y0 + i * row_h)
            hero_row_y1 = int(hero_row_y0 + row_h)

            num_row_y0 = int(hero_row_y0 + num_y_offset)
            num_row_y1 = int(num_row_y0 + num_box_h)

            # hero
            slot = {}
            slot["hero"] = (hero_x0, hero_row_y0, hero_x1, hero_row_y1)

            # kills / assists / deaths
            (kx0, kx1), (ax0, ax1), (dx0, dx1) = kda_cols
            slot["kills"]   = (kx0, num_row_y0, kx1, num_row_y1)
            slot["assists"] = (ax0, num_row_y0, ax1, num_row_y1)
            slot["deaths"]  = (dx0, num_row_y0, dx1, num_row_y1)

            # damage / heal / mitig
            (dmgx0, dmgx1), (healx0, healx1), (mitx0, mitx1) = dmg_cols
            slot["damage"] = (dmgx0,  num_row_y0, dmgx1,  num_row_y1)
            slot["heal"]   = (healx0, num_row_y0, healx1, num_row_y1)
            slot["mitig"]  = (mitx0,  num_row_y0, mitx1,  num_row_y1)

            result["blue"].append(slot)

        # ----- 빨간 팀 -----
        for i in range(self.N_RED_ROWS):
            hero_row_y0 = int(red_hero_y0 + i * row_h)
            hero_row_y1 = int(hero_row_y0 + row_h)

            num_row_y0 = int(hero_row_y0 + num_y_offset)
            num_row_y1 = int(num_row_y0 + num_box_h)

            slot = {}
            slot["hero"] = (hero_x0, hero_row_y0, hero_x1, hero_row_y1)

            (kx0, kx1), (ax0, ax1), (dx0, dx1) = kda_cols
            slot["kills"]   = (kx0, num_row_y0, kx1, num_row_y1)
            slot["assists"] = (ax0, num_row_y0, ax1, num_row_y1)
            slot["deaths"]  = (dx0, num_row_y0, dx1, num_row_y1)

            (dmgx0, dmgx1), (healx0, healx1), (mitx0, mitx1) = dmg_cols
            slot["damage"] = (dmgx0,  num_row_y0, dmgx1,  num_row_y1)
            slot["heal"]   = (healx0, num_row_y0, healx1, num_row_y1)
            slot["mitig"]  = (mitx0,  num_row_y0, mitx1,  num_row_y1)

            result["red"].append(slot)

        return result

    # (옵션) 디버그로 박스를 그려보고 싶을 때
    def draw_debug(self, img, boxes):
        debug = img.copy()

        color_team = {
            "blue": (255, 255, 0),
            "red":  (0, 165, 255),
        }
        color_stat = {
            "kills":  (0, 255, 0),
            "assists":(0, 200, 255),
            "deaths": (0, 0, 255),
            "damage": (0, 255, 255),
            "heal":   (255, 0, 255),
            "mitig":  (255, 255, 255),
        }

        for team in ["blue", "red"]:
            for slot in boxes[team]:
                # hero
                x0, y0, x1, y1 = slot["hero"]
                cv2.rectangle(debug, (x0, y0), (x1, y1), color_team[team], 2)

                # stats
                for k, c in color_stat.items():
                    x0, y0, x1, y1 = slot[k]
                    cv2.rectangle(debug, (x0, y0), (x1, y1), c, 2)

        return debug


# -------- 사용 예시 --------
if __name__ == "__main__":
    # img_path = r"dataset/blue/2025-11-25 030028.png"
    img_path = r"KakaoTalk_20251124_212103298.png"
    img = cv2.imread(img_path)

    cropper = OWScoreboardCropper()
    boxes = cropper.get_player_boxes(img)

    # 예: 파란 2번의 damage 좌표
    # print("blue[1].damage:", boxes["blue"][1]["damage"])
    print(boxes['blue'][0])

    # 디버그 이미지 보기
    debug_img = cropper.draw_debug(img, boxes)
    cv2.imshow("debug", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
