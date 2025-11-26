import cv2
import numpy as np


class OWScoreboardCropper:
    def __init__(self):
        # 기준 해상도 (1K FHD)
        self.REF_W, self.REF_H = 1920, 1080

        # 파란 팀 맨 위 영웅 아이콘 (1080p 기준)
        self.HERO_X0, self.HERO_Y0 = 565.00, 216.00
        self.HERO_X1, self.HERO_Y1 = 628.00, 280.00

        # KDA 영역
        self.KDA_X0, self.KDA_Y0 = 910.00, 216.00
        self.KDA_X1, self.KDA_Y1 = 1075.00, 280.00

        # DMG / HEAL / MIT 영역
        self.DMG_X0, self.DMG_Y0 = 1075.00, 216.00
        self.DMG_X1, self.DMG_Y1 = 1385.00, 280.00

        # 빨간팀 첫 영웅 Y 위치
        self.RED_HERO_Y0 = 619.0
        self.RED_HERO_Y1 = self.RED_HERO_Y0 + (self.HERO_Y1 - self.HERO_Y0)

        self.N_BLUE_ROWS = 5
        self.N_RED_ROWS = 5

        # 정규화된 ROI
        self.HERO_N = self._norm_roi(self.HERO_X0, self.HERO_Y0,
                                     self.HERO_X1, self.HERO_Y1)
        self.RED_HERO_N = self._norm_roi(self.HERO_X0, self.RED_HERO_Y0,
                                         self.HERO_X1, self.RED_HERO_Y1)
        self.KDA_N = self._norm_roi(self.KDA_X0, self.KDA_Y0,
                                    self.KDA_X1, self.KDA_Y1)
        self.DMG_N = self._norm_roi(self.DMG_X0, self.DMG_Y0,
                                    self.DMG_X1, self.DMG_Y1)

    # ============================================================
    #  기본 유틸
    # ============================================================
    def normalize_image(self, img):
        """모든 입력 이미지를 1920×1080 기준으로 맞춘다."""
        h, w = img.shape[:2]
        if (w, h) == (self.REF_W, self.REF_H):
            return img
        return cv2.resize(img, (self.REF_W, self.REF_H),
                          interpolation=cv2.INTER_AREA)

    def _norm_roi(self, x0, y0, x1, y1):
        return (x0 / self.REF_W, y0 / self.REF_H,
                x1 / self.REF_W, y1 / self.REF_H)

    def _denorm_roi(self, roi_n, img_w, img_h):
        x0_n, y0_n, x1_n, y1_n = roi_n
        return (int(x0_n * img_w), int(y0_n * img_h),
                int(x1_n * img_w), int(y1_n * img_h))

    def _split_three(self, x0, x1):
        w = x1 - x0
        step = w / 3.0
        xs = []
        for i in range(3):
            sx0 = int(x0 + i * step)
            sx1 = int(x0 + (i + 1) * step)
            xs.append((sx0, sx1))
        return xs

    # ============================================================
    #  메인 로직
    # ============================================================
    def get_player_boxes(self, img):
        """
        입력 이미지(2K/1K/등 상관없음)를 1080p로 맞춘 뒤 좌표 반환.
        반환되는 좌표는 반드시 normalize_image(img)에 적용해야 함.
        """
        img_norm = self.normalize_image(img)
        h, w = img_norm.shape[:2]

        hero_x0, hero_y0, hero_x1, hero_y1 = self._denorm_roi(self.HERO_N, w, h)
        red_hero_x0, red_hero_y0, red_hero_x1, red_hero_y1 = \
            self._denorm_roi(self.RED_HERO_N, w, h)

        kda_x0, kda_y0, kda_x1, kda_y1 = self._denorm_roi(self.KDA_N, w, h)
        dmg_x0, dmg_y0, dmg_x1, dmg_y1 = self._denorm_roi(self.DMG_N, w, h)

        row_h = hero_y1 - hero_y0
        num_y_offset = kda_y0 - hero_y0
        num_box_h = kda_y1 - kda_y0

        kda_cols = self._split_three(kda_x0, kda_x1)
        dmg_cols = self._split_three(dmg_x0, dmg_x1)

        result = {"blue": [], "red": []}

        # --------- BLUE ----------
        for i in range(self.N_BLUE_ROWS):
            slot = {}
            hero_row_y0 = int(hero_y0 + i * row_h)
            hero_row_y1 = int(hero_row_y0 + row_h)
            num_row_y0 = int(hero_row_y0 + num_y_offset)
            num_row_y1 = int(num_row_y0 + num_box_h)

            slot["hero"] = (hero_x0, hero_row_y0, hero_x1, hero_row_y1)

            (kx0, kx1), (ax0, ax1), (dx0, dx1) = kda_cols
            slot["kills"]   = (kx0, num_row_y0, kx1, num_row_y1)
            slot["assists"] = (ax0, num_row_y0, ax1, num_row_y1)
            slot["deaths"]  = (dx0, num_row_y0, dx1, num_row_y1)

            (dx0_, dx1_), (hx0_, hx1_), (mx0_, mx1_) = dmg_cols
            slot["damage"] = (dx0_, num_row_y0, dx1_, num_row_y1)
            slot["heal"]   = (hx0_, num_row_y0, hx1_, num_row_y1)
            slot["mitig"]  = (mx0_, num_row_y0, mx1_, num_row_y1)

            result["blue"].append(slot)

        # --------- RED ----------
        for i in range(self.N_RED_ROWS):
            slot = {}
            hero_row_y0 = int(red_hero_y0 + i * row_h)
            hero_row_y1 = int(hero_row_y0 + row_h)
            num_row_y0 = int(hero_row_y0 + num_y_offset)
            num_row_y1 = int(num_row_y0 + num_box_h)

            slot["hero"] = (hero_x0, hero_row_y0, hero_x1, hero_row_y1)

            (kx0, kx1), (ax0, ax1), (dx0, dx1) = kda_cols
            slot["kills"]   = (kx0, num_row_y0, kx1, num_row_y1)
            slot["assists"] = (ax0, num_row_y0, ax1, num_row_y1)
            slot["deaths"]  = (dx0, num_row_y0, dx1, num_row_y1)

            (dx0_, dx1_), (hx0_, hx1_), (mx0_, mx1_) = dmg_cols
            slot["damage"] = (dx0_, num_row_y0, dx1_, num_row_y1)
            slot["heal"]   = (hx0_, num_row_y0, hx1_, num_row_y1)
            slot["mitig"]  = (mx0_, num_row_y0, mx1_, num_row_y1)

            result["red"].append(slot)

        return result

    # ============================================================
    #  디버그
    # ============================================================
    def draw_debug(self, img, boxes):
        debug = self.normalize_image(img).copy()

        team_color = {"blue": (255,255,0), "red": (0,165,255)}
        stat_color = {
            "kills": (0,255,0),
            "assists": (0,200,255),
            "deaths": (0,0,255),
            "damage": (0,255,255),
            "heal": (255,0,255),
            "mitig": (255,255,255),
        }

        for team in ["blue", "red"]:
            for slot in boxes[team]:

                (x0,y0,x1,y1) = slot["hero"]
                cv2.rectangle(debug, (x0,y0), (x1,y1),
                              team_color[team], 2)

                for k, c in stat_color.items():
                    (x0,y0,x1,y1) = slot[k]
                    cv2.rectangle(debug, (x0,y0), (x1,y1),
                                  c, 2)

        return debug


# -------- 사용 예시 --------
if __name__ == "__main__":
    img_path = r"testdata/test1.png"  # 2K든 1K든 상관 없음
    img = cv2.imread(img_path)

    cropper = OWScoreboardCropper()

    # get_player_boxes는 이제 boxes만 리턴
    boxes = cropper.get_player_boxes(img)

    print("blue[0]:", boxes["blue"][0])

    debug_img = cropper.draw_debug(img, boxes)
    cv2.imshow("debug", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
