import os
import glob
import csv
import cv2

from crop_coordinates import OWScoreboardCropper
from read_number_paddleocr import OWStatsRecognizer


class OWNumberDatasetBuilder:
    def __init__(self,
                 base_dir="dataset",
                 out_dir="dataset/img_to_number",
                 cropper=None,
                 stats_recognizer=None):

        self.base_dir = base_dir
        self.blue_dir = os.path.join(base_dir, "blue")
        self.red_dir = os.path.join(base_dir, "red")

        self.out_dir = out_dir
        self.out_img_dir = os.path.join(out_dir, "images")
        self.label_csv = os.path.join(out_dir, "labels.csv")

        os.makedirs(self.out_img_dir, exist_ok=True)

        # 🔹 여기 cropper는 기존 2K 기준 crop_coordinates.py 버전 사용
        self.cropper = cropper or OWScoreboardCropper()

        # 🔹 PaddleOCR 숫자 리더 (라벨 생성용)
        self.stats_recognizer = stats_recognizer or OWStatsRecognizer(cropper=self.cropper)

        # 어떤 스탯 칸을 저장할지
        self.stat_keys = ["kills", "assists", "deaths", "damage", "heal", "mitig"]

        # CSV 헤더
        self.header = [
            "filename",      # 저장된 crop 이미지 파일명
            "value",         # 정수 값
            "stat_key",      # kills / assists / ...
            "team",          # blue / red
            "slot_index",    # 0~4 (위에서부터)
            "src_team_dir",  # 원본 이미지가 있던 폴더 (blue/red)
            "src_image"      # 원본 파일명
        ]

    def _collect_image_paths(self):
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        all_paths = []

        for src_team_dir in [self.blue_dir, self.red_dir]:
            src_team_label = os.path.basename(src_team_dir)  # "blue" or "red"
            file_list = []
            for pat in patterns:
                file_list.extend(glob.glob(os.path.join(src_team_dir, pat)))
            file_list.sort()
            for p in file_list:
                all_paths.append((p, src_team_label))

        return all_paths

    def build_dataset(self):
        image_infos = self._collect_image_paths()
        if not image_infos:
            print("[INFO] 처리할 이미지가 없습니다.")
            return

        # 기존 CSV가 없으면 헤더부터 작성
        file_exists = os.path.exists(self.label_csv)
        f = open(self.label_csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(self.header)

        idx = 0  # 이미지 파일 이름 넘버링용

        for img_path, src_team_label in image_infos:
            print(f"[PROCESS] {src_team_label}  {img_path}")
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                print(f"[WARN] 이미지를 읽을 수 없음: {img_path}")
                continue

            try:
                # 1) 라벨은 2K 원본 기준으로 PaddleOCR 돌려서 읽음
                stats = self.stats_recognizer.read_all(img_orig)

                # 2) 학습용 이미지는 1920x1080(FHD)로 통일
                target_w, target_h = 1920, 1080
                img_1080 = cv2.resize(
                    img_orig, (target_w, target_h),
                    interpolation=cv2.INTER_AREA
                )

                # 3) 1080p 기준으로 crop 좌표 계산
                boxes_1080 = self.cropper.get_player_boxes(img_1080)

                for team in ["blue", "red"]:
                    for slot_idx, slot_boxes in enumerate(boxes_1080[team]):
                        stat_slot = stats[team][slot_idx]  # ← 라벨은 2K에서 읽은 값

                        for stat_key in self.stat_keys:
                            value = stat_slot.get(stat_key, 0)

                            # 4) 1080p 이미지에서 crop
                            x0, y0, x1, y1 = slot_boxes[stat_key]
                            crop = img_1080[y0:y1, x0:x1]

                            if crop.size == 0:
                                continue

                            # 파일명: 000001_blue_0_damage.png 이런 식
                            fname = f"{idx:06d}_{team}_{slot_idx}_{stat_key}.png"
                            out_path = os.path.join(self.out_img_dir, fname)

                            cv2.imwrite(out_path, crop)

                            # 라벨 한 줄
                            row = [
                                fname,
                                int(value),
                                stat_key,
                                team,
                                slot_idx,
                                src_team_label,
                                os.path.basename(img_path)
                            ]
                            writer.writerow(row)

                            idx += 1

            except Exception as e:
                print(f"[ERROR] {img_path} 처리 중 에러: {e}")

        f.close()
        print(f"[DONE] img_to_number 데이터셋 생성 완료")
        print(f"  - 이미지 디렉토리: {self.out_img_dir}")
        print(f"  - 라벨 CSV:        {self.label_csv}")


if __name__ == "__main__":
    builder = OWNumberDatasetBuilder(
        base_dir="dataset",
        out_dir="dataset/img_to_number",
    )
    builder.build_dataset()
