# predict_from_image.py

import os
import cv2
import pandas as pd
import joblib

from crop_coordinates_1k import OWScoreboardCropper
from read_number_cnn_model import OWStatsRecognizer
from hero_classification import OWHeroTemplateClassifier
from feature_transformer import OWFeatureTransformer


def extract_rows_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read {img_path}")

    cropper = OWScoreboardCropper()
    stats_rec = OWStatsRecognizer(cropper=cropper)
    hero_cls = OWHeroTemplateClassifier(cropper=cropper, threshold=0.5)

    stats = stats_rec.read_all(img)
    heroes = hero_cls.classify_all(img)

    rows = []
    src_img = os.path.basename(img_path)

    for team in ("blue", "red"):
        for idx, (slot_st, slot_he) in enumerate(zip(stats[team], heroes[team])):
            rows.append({
                "src_team": "unknown",
                "src_image": src_img,
                "team": team,
                "slot_index": idx,
                "hero": slot_he["hero_name"],
                "kills": slot_st["kills"],
                "assists": slot_st["assists"],
                "deaths": slot_st["deaths"],
                "damage": slot_st["damage"],
                "heal": slot_st["heal"],
                "mitig": slot_st["mitig"],
            })

    return pd.DataFrame(rows)


def main(path):
    df_raw = extract_rows_from_image(path)
    print(df_raw)

    transformer = OWFeatureTransformer()
    features = transformer.transform(df_raw, drop_id_cols=True)

    artifact = joblib.load("xgb_win_model.joblib")
    model = artifact["model"]
    cols = artifact["feature_cols"]

    win_p = model.predict_proba(features[cols])[:, 1]

    heroes = df_raw["hero"].astype(str).values
    maxlen = max(len(h) for h in heroes)

    print("\nhero".ljust(maxlen), " win_proba")
    for h, p in zip(heroes, win_p):
        print(h.ljust(maxlen), f"{p:.4f}")

    return pd.DataFrame([heroes, win_p]).T


if __name__ == "__main__":
    img_path = "testdata/test1.png"
    out = main(img_path)

    blue_sum = out.iloc[:5, 1].sum()
    red_sum  = out.iloc[5:10, 1].sum()

    out.iloc[:5, 1] *= 5 / blue_sum
    out.iloc[5:10, 1] *= 5 / red_sum

    print(out)
