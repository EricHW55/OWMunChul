# hero_classification.py
import cv2
import numpy as np
import re
import os
from glob import glob

from crop_coordinates import OWScoreboardCropper


class OWHeroTemplateClassifier:
    def __init__(
        self,
        cropper=None,
        template_dir="hero_templates",
        target_size=(64, 64),
        threshold=0.7,
    ):
        """
        cropper      : OWScoreboardCropper 인스턴스 (없으면 내부에서 생성)
        template_dir : 영웅 아이콘 템플릿 폴더 (kiriko.png, genji.png, ...)
        target_size  : 템플릿/크롭 이미지를 리사이즈할 크기 (w, h)
        threshold    : 매칭 점수가 이 값보다 작으면 'unknown' 처리
        """
        self.cropper = cropper or OWScoreboardCropper()
        self.template_dir = template_dir
        self.target_size = target_size
        self.threshold = threshold

        # 템플릿 로드
        self.templates = self._load_templates()

    # ---------- 내부 유틸 ----------

    def _load_templates(self):
        """
        hero_templates 폴더의 png/jpg를 전부 읽어서
        { "kiriko": template_img, ... } 형태로 반환
        """
        templates = {}

        exts = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for e in exts:
            paths.extend(glob(os.path.join(self.template_dir, e)))

        if not paths:
            raise RuntimeError(
                f"No template images found in template_dir={self.template_dir}"
            )

        for path in paths:
            # 파일 이름에서 영웅 이름 추출 (kiriko.png -> kiriko)
            name = os.path.splitext(os.path.basename(path))[0]

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Failed to read template: {path}")
                continue

            tmpl = self._preprocess(img)
            templates[name] = tmpl

        print(f"[HeroTemplate] Loaded {len(templates)} templates: {list(templates.keys())}\n")
        return templates

    def _preprocess(self, img):
        """리사이즈 + 그레이스케일 전처리"""
        # 리사이즈
        resized = cv2.resize(
            img, self.target_size, interpolation=cv2.INTER_AREA
        )
        # 그레이스케일
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray

    def _crop_box(self, img, box):
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]

    # ---------- 템플릿 매칭 핵심 ----------

    def match_single(self, hero_crop):
        """
        hero_crop: BGR crop 이미지 (한 칸의 영웅 아이콘)
        return: (best_name, best_score)
        """
        if hero_crop is None or hero_crop.size == 0:
            return "unknown", 0.0

        patch = self._preprocess(hero_crop)

        best_name = "unknown"
        best_score = -1.0

        for name, tmpl in self.templates.items():
            # 같은 크기로 맞춰뒀으니 matchTemplate 결과는 1x1
            res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.threshold:
            return "unknown", best_score
        # 숫자 제거 (lucio_1 -> lucio_, kiriko2 -> kiriko)
        best_name = re.sub(r'\d+', '', best_name)
        
        return best_name, best_score

    def classify_all(self, img):
        """
        img: 전체 스코어보드 스크린샷 (BGR)
        return:
          {
            "blue": [
               { "hero_name": str, "match_score": float, "bbox": (x0,y0,x1,y1) },
               ... (5명)
            ],
            "red":  [ ... (5명) ]
          }
        """
        boxes = self.cropper.get_player_boxes(img)
        result = {"blue": [], "red": []}

        for team in ["blue", "red"]:
            for slot_boxes in boxes[team]:
                hx0, hy0, hx1, hy1 = slot_boxes["hero"]
                hero_crop = self._crop_box(img, slot_boxes["hero"])

                hero_name, score = self.match_single(hero_crop)

                result[team].append(
                    {
                        "hero_name": hero_name,
                        "match_score": score,
                        "bbox": (hx0, hy0, hx1, hy1),
                    }
                )

        return result

    # 디버그용: 특정 칸만 확인
    def debug_single_hero(self, img, team="blue", player_idx=0):
        boxes = self.cropper.get_player_boxes(img)
        hero_box = boxes[team][player_idx]["hero"]
        hero_crop = self._crop_box(img, hero_box)

        name, score = self.match_single(hero_crop)
        print(f"{team}[{player_idx}] hero => {name} (score={score:.3f})")

        cv2.imshow("hero_crop", hero_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return name, score


# -------- 간단 사용 예시 --------
if __name__ == "__main__":
    img_path = r"dataset/blue/2025-11-25 030028.png"
    img = cv2.imread(img_path)

    cropper = OWScoreboardCropper()
    classifier = OWHeroTemplateClassifier(
        cropper=cropper,
        template_dir="hero_templates",
        target_size=(64, 64),
        threshold=0.7,  # 필요하면 나중에 튜닝
    )

    heroes = classifier.classify_all(img)

    from pprint import pprint
    print("Blue team heroes:")
    pprint(heroes["blue"])
    print("Red team heroes:")
    pprint(heroes["red"])

    # 디버깅: 파란 1번 영웅만 띄워보기
    # classifier.debug_single_hero(img, team="blue", player_idx=0)
