import csv
import os
import glob
import cv2

from crop_coordinates import OWScoreboardCropper
from read_number_paddleocr import OWStatsRecognizer
from hero_classification import OWHeroTemplateClassifier


class OWCSVBuilder:
    def __init__(self,
                 base_dir="dataset",
                 csv_path="ow_stats.csv",
                 cropper=None,
                 stats_recognizer=None,
                 hero_classifier=None):
        self.base_dir = base_dir
        self.blue_dir = os.path.join(base_dir, "blue")
        self.red_dir = os.path.join(base_dir, "red")
        self.csv_path = csv_path

        self.cropper = cropper or OWScoreboardCropper()
        self.stats_recognizer = stats_recognizer or OWStatsRecognizer(cropper=self.cropper)
        self.hero_classifier = hero_classifier or OWHeroTemplateClassifier(cropper=self.cropper)

        # CSV 헤더: src_team / src_image / win 추가
        self.header = [
            "src_team", "src_image", "win",
            "team", "slot_index", "hero",
            "kills", "assists", "deaths",
            "damage", "heal", "mitig",
        ]

    def _image_to_rows(self, img_path: str, src_team_label: str):
        """
        한 이미지(한 판 스샷) → 여러 row (최대 10줄)

        - 폴더가 blue  : blue 팀 win=1,  red 팀 win=0
        - 폴더가 red   : blue 팀 win=0,  red 팀 win=1
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 이미지를 읽을 수 없음: {img_path}")
            return []

        stats = self.stats_recognizer.read_all(img)
        heroes = self.hero_classifier.classify_all(img)

        rows = []

        for team in ("blue", "red"):
            # 폴더 기준으로 이긴 팀 결정
            if src_team_label == "blue":
                # blue 폴더 => blue 팀이 이긴 경기
                win = 1 if team == "blue" else 0
            else:
                # red 폴더 => red 팀이 이긴 경기
                win = 1 if team == "red" else 0

            for i, (stat_slot, hero_slot) in enumerate(zip(stats[team], heroes[team])):
                row = [
                    src_team_label,              # 스샷이 있던 폴더 ("blue"/"red")
                    os.path.basename(img_path),  # 파일 이름
                    win,                         # 해당 팀의 승패 라벨
                    team,                        # row의 팀 ("blue"/"red")
                    i,                           # slot_index (0~4)
                    hero_slot["hero_name"],      # hero
                    stat_slot.get("kills", 0),
                    stat_slot.get("assists", 0),
                    stat_slot.get("deaths", 0),
                    stat_slot.get("damage", 0),
                    stat_slot.get("heal", 0),
                    stat_slot.get("mitig", 0),
                ]
                rows.append(row)

        return rows


    def build_csv(self):
        """
        dataset/blue, dataset/red 하위의 모든 이미지를 읽어서
        CSV 파일로 저장 (기존 파일이 있으면 append 방식)
        """
        # png, jpg 다 처리하고 싶으면 패턴 추가
        patterns = ["*.png", "*.jpg", "*.jpeg"]

        all_image_info = []  # (img_path, src_team_label)

        for src_team_dir in [self.blue_dir, self.red_dir]:
            src_team_label = os.path.basename(src_team_dir)  # "blue" or "red"
            file_list = []
            for pat in patterns:
                file_list.extend(glob.glob(os.path.join(src_team_dir, pat)))

            file_list.sort()
            for img_path in file_list:
                all_image_info.append((img_path, src_team_label))

        if not all_image_info:
            print("[INFO] 처리할 이미지가 없습니다.")
            return

        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 새 파일이면 헤더부터 쓰기
            if not file_exists:
                writer.writerow(self.header)

            for img_path, src_team_label in all_image_info:
                print(f"[PROCESS] {src_team_label}  {img_path}")
                try:
                    rows = self._image_to_rows(img_path, src_team_label)
                    for row in rows:
                        writer.writerow(row)
                except Exception as e:
                    print(f"[ERROR] {img_path} 처리 중 에러: {e}")

        print(f"[DONE] CSV 생성/추가 완료: {self.csv_path}")


if __name__ == "__main__":
    print("\n=== CSV 빌드 ===")
    builder = OWCSVBuilder(
        base_dir="dataset",
        csv_path="ow_stats.csv",
    )
    builder.build_csv()
