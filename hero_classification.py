# hero_classification.py

import cv2
import numpy as np
import re
import os
from glob import glob

from crop_coordinates_1k import OWScoreboardCropper


class OWHeroTemplateClassifier:
    def __init__(self, cropper=None,
                 template_dir="hero_templates",
                 target_size=(64, 64),
                 threshold=0.65):
        self.cropper = cropper or OWScoreboardCropper()
        self.template_dir = template_dir
        self.target_size = target_size
        self.threshold = threshold

        self.templates = self._load_templates()

    def _load_templates(self):
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(glob(os.path.join(self.template_dir, ext)))

        templates = {}
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p)
            if img is None:
                continue
            templates[name] = self._prep(img)
        return templates

    def _prep(self, img):
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _crop(self, img_norm, box):
        x0, y0, x1, y1 = box
        return img_norm[y0:y1, x0:x1]

    def match_single(self, crop):
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        patch = self._prep(crop)
        best = ("unknown", -1.0)

        for name, tmpl in self.templates.items():
            score = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
            if score > best[1]:
                best = (name, score)

        name, score = best
        if score < self.threshold:
            return "unknown", score

        name = re.sub(r'\d+', '', name)  # lucio_1 → lucio
        return name, score

    def classify_all(self, img):
        boxes = self.cropper.get_player_boxes(img)
        img_norm = self.cropper.normalize_image(img)

        result = {"blue": [], "red": []}

        for team in ["blue", "red"]:
            for slot in boxes[team]:
                crop = self._crop(img_norm, slot["hero"])
                name, score = self.match_single(crop)

                result[team].append({
                    "hero_name": name,
                    "match_score": score,
                    "bbox": slot["hero"],
                })

        return result



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
