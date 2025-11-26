# predict_from_image.py

import os
import cv2
import pandas as pd
import joblib

from crop_coordinates import OWScoreboardCropper
from read_number_paddleocr import OWStatsRecognizer
# from read_number_cnn_model import OWStatsRecognizer
from hero_classification import OWHeroTemplateClassifier
from feature_transformer import OWFeatureTransformer


def extract_rows_from_image(img_path: str) -> pd.DataFrame:
    """
    test.png 같은 스코어보드 스샷 한 장을
    -> ow_stats.csv 형식과 동일한 row들로 변환해서 DataFrame 반환

    컬럼:
      src_team, src_image, team, slot_index, hero,
      kills, assists, deaths, damage, heal, mitig
    (win 라벨은 여기서는 모르는 값이므로 넣지 않음)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")

    cropper = OWScoreboardCropper()
    stats_recognizer = OWStatsRecognizer(cropper=cropper)
    hero_classifier = OWHeroTemplateClassifier(cropper=cropper, threshold=0.5)

    stats = stats_recognizer.read_all(img)
    heroes = hero_classifier.classify_all(img)

    rows = []
    src_image = os.path.basename(img_path)
    src_team = "unknown"  # 학습 때만 쓰이던 정보라 여기서는 의미 없음

    for team in ("blue", "red"):
        for slot_idx, (stat_slot, hero_slot) in enumerate(zip(stats[team], heroes[team])):
            row = {
                "src_team": src_team,
                "src_image": src_image,
                "team": team,
                "slot_index": slot_idx,
                "hero": hero_slot["hero_name"],
                "kills": stat_slot.get("kills", 0),
                "assists": stat_slot.get("assists", 0),
                "deaths": stat_slot.get("deaths", 0),
                "damage": stat_slot.get("damage", 0),
                "heal": stat_slot.get("heal", 0),
                "mitig": stat_slot.get("mitig", 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main(_path:str):
    IMG_PATH = _path                     # 테스트할 이미지 경로
    MODEL_PATH = "xgb_win_model.joblib"  # 학습해 둔 모델 아티팩트

    # 1) 이미지에서 raw 스탯 DataFrame 추출
    df_raw = extract_rows_from_image(IMG_PATH)
    
    print(df_raw)

    # 2) 피처 변환 (학습 때 쓰던 것과 동일한 전처리)
    transformer = OWFeatureTransformer()
    features = transformer.transform(df_raw, drop_id_cols=True)

    # 3) XGBoost 모델 로드
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact.get("threshold", 0.5)

    X = features[feature_cols]

    # 4) 각 row별 승리 확률 예측
    win_proba = model.predict_proba(X)[:, 1]  # P(win=1)

    # 5) 영웅 이름 + 예측값 출력 (열 정렬)
    hero_names = df_raw["hero"].astype(str).values
    max_len = max(len(h) for h in hero_names)  # 가장 긴 이름 길이

    print("hero".ljust(max_len), "  win_proba")
    for hero_name, p in zip(hero_names, win_proba):
        # hero_name을 max_len 만큼 left-pad 해서 정렬
        print(f"{hero_name.ljust(max_len)}  {p:.4f}")
        
        
    return pd.DataFrame([hero_names, win_proba]).T

if __name__ == "__main__":
    img_path = 'testdata/test1_1080p.png'
    results = main(img_path)

    blue_sum = results.iloc[:5, 1].sum()
    red_sum  = results.iloc[5:10, 1].sum()

    # 합이 5가 되도록: 값 * (5 / 합)
    results.iloc[:5, 1]  = results.iloc[:5, 1]  * (5 / blue_sum)
    results.iloc[5:10, 1] = results.iloc[5:10, 1] * (5 / red_sum)

    print(results)

    # print("blue total:", results.iloc[:5, 1].sum())
    # print("red total:", results.iloc[5:10, 1].sum())