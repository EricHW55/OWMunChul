from crop_coordinates import OWScoreboardCropper
from read_number import OWStatsRecognizer
from hero_classification import OWHeroTemplateClassifier
import cv2

img_path = r"dataset/red/red_2025-11-26_01-22-31.png"
img = cv2.imread(img_path)

cropper = OWScoreboardCropper()
stats_recognizer = OWStatsRecognizer(cropper=cropper)
hero_classifier = OWHeroTemplateClassifier(cropper=cropper)

stats = stats_recognizer.read_all(img)
heroes = hero_classifier.classify_all(img)

# 영웅 + 스탯 합치기 

header = ["team", "slot_index", "hero",
          "kills", "assists", "deaths",
          "damage", "heal", "mitig"]

rows = []  # 최종 행렬 (10 x 9 정도)

for team in ("blue", "red"):
    for i, (stat_slot, hero_slot) in enumerate(zip(stats[team], heroes[team])):
        row = [
            team,                      # team
            i,                         # slot_index (0~4)
            hero_slot["hero_name"],    # hero
            stat_slot.get("kills", 0),
            stat_slot.get("assists", 0),
            stat_slot.get("deaths", 0),
            stat_slot.get("damage", 0),
            stat_slot.get("heal", 0),
            stat_slot.get("mitig", 0),
        ]
        rows.append(row)


#  출력 부분 
# 문자열로 변환
str_rows = [[str(x) for x in row] for row in rows]

# 각 컬럼 최대 길이 계산
col_widths = []
for col_idx, col_name in enumerate(header):
    max_len = len(col_name)
    for row in str_rows:
        max_len = max(max_len, len(row[col_idx]))
    col_widths.append(max_len)

# 한 줄 출력용 함수
def format_row(values):
    return "  ".join(
        str(v).ljust(col_widths[i]) for i, v in enumerate(values)
    )

# 출력
print(format_row(header))
for row in str_rows:
    print(format_row(row))
