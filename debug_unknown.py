import os
import glob
import cv2

from crop_coordinates import OWScoreboardCropper
from hero_classification import OWHeroTemplateClassifier

BASE_DIR = "dataset"
BLUE_DIR = os.path.join(BASE_DIR, "blue")
RED_DIR = os.path.join(BASE_DIR, "red")


def debug_unknown_heroes():
    unknown_count = 0
    cropper = OWScoreboardCropper()
    hero_classifier = OWHeroTemplateClassifier(cropper=cropper, threshold=0.65)

    # png, jpg 둘 다 처리하고 싶으면 패턴에 추가
    patterns = ["*.png", "*.jpg", "*.jpeg"]

    for src_team_dir in [BLUE_DIR, RED_DIR]:
        src_team_label = os.path.basename(src_team_dir)  # "blue" / "red"

        file_list = []
        for pat in patterns:
            file_list.extend(glob.glob(os.path.join(src_team_dir, pat)))

        file_list.sort()

        for img_path in file_list:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] 이미지를 읽을 수 없음: {img_path}")
                continue

            heroes = hero_classifier.classify_all(img)

            # unknown 있는지 확인
            has_unknown = False
            for team in ("blue", "red"):
                for slot in heroes[team]:
                    if slot["hero_name"] == "unknown":
                        has_unknown = True
                        break
                if has_unknown:
                    break

            if has_unknown:
                unknown_count += 1
                print("\n[UNKNOWN HERO DETECTED]")
                print(f"원본 폴더: {src_team_label}, 파일: {img_path}")

                # 이 이미지의 영웅들을 전부 출력
                for team in ("blue", "red"):
                    print(f"  Team: {team}")
                    for i, slot in enumerate(heroes[team]):
                        name = slot["hero_name"]
                        score = slot.get("match_score", None)
                        if score is not None:
                            print(f"    {i}: {name:15s}  (score={score:.3f})")
                        else:
                            print(f"    {i}: {name:15s}")

    print("\n[완료] unknown 영웅 디버그 검사 끝.")
    print(f"Unknown 개수 : {unknown_count}")



if __name__ == "__main__":
    debug_unknown_heroes()
