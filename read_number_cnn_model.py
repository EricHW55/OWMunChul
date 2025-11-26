# read_number_cnn_model.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_coordinates_1k import OWScoreboardCropper


# ====== 하이퍼파라미터 ====== #
MAX_LEN = 5
NUM_CLASSES = 11   # 0~9 + blank
BLANK = 10

TARGET_H = 32
TARGET_W = 128


# ====== CNN 모델 ====== #
class ScoreNumberNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        conv_out_h = TARGET_H // 8
        conv_out_w = TARGET_W // 8
        conv_dim = 128 * conv_out_h * conv_out_w

        self.fc = nn.Sequential(
            nn.Linear(conv_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, MAX_LEN * NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x.view(-1, MAX_LEN, NUM_CLASSES)


# ====== 전처리 ====== #
def preprocess_crop_to_tensor(crop):
    if crop is None or crop.size == 0:
        raise RuntimeError("Empty crop")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = TARGET_H / float(h)
    new_w = max(1, int(w * scale))

    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(gray, (new_w, TARGET_H), interpolation=interp)

    if new_w > TARGET_W:
        resized = cv2.resize(resized, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    else:
        pad_left = (TARGET_W - new_w) // 2
        pad_right = TARGET_W - new_w - pad_left
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=0)

    resized = resized.astype(np.float32) / 255.0
    return torch.from_numpy(resized[None, None])


# ====== 디코더 ====== #
def decode_digits(logits):
    preds = logits.softmax(-1).argmax(-1)[0].tolist()

    digits = [str(p) for p in preds if p != BLANK]
    if not digits:
        return 0

    try:
        return int("".join(digits))
    except:
        return 0


# ====== 메인 OCR 클래스 ====== #
class OWStatsRecognizer:
    def __init__(self, cropper=None, ckpt_path="checkpoints/score_number_net.pt"):
        self.cropper = cropper or OWScoreboardCropper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ScoreNumberNet().to(self.device)
        self.model.eval()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state_dict)

        self.stat_keys = ["kills", "assists", "deaths", "damage", "heal", "mitig"]

        print("[INFO] CNN checkpoint loaded:", ckpt_path)

    def _crop_from_norm(self, img_norm, box):
        x0, y0, x1, y1 = box
        return img_norm[y0:y1, x0:x1]

    def _ocr(self, crop):
        try:
            tensor = preprocess_crop_to_tensor(crop).to(self.device)
        except RuntimeError:
            return 0

        with torch.no_grad():
            logits = self.model(tensor)
        return decode_digits(logits)

    def read_all(self, img):
        boxes = self.cropper.get_player_boxes(img)
        img_norm = self.cropper.normalize_image(img)   # 반드시 여기서 crop!!

        result = {"blue": [], "red": []}

        for team in ["blue", "red"]:
            for slot in boxes[team]:
                slot_res = {}

                for key in self.stat_keys:
                    crop = self._crop_from_norm(img_norm, slot[key])
                    value = self._ocr(crop)
                    slot_res[key] = value

                result[team].append(slot_res)

        return result



# -------- 사용 예시 --------
if __name__ == "__main__":
    img_path = r"testdata/test1.png"
    img = cv2.imread(img_path)

    cropper = OWScoreboardCropper()
    recognizer = OWStatsRecognizer(cropper=cropper,
                                   ckpt_path="checkpoints/score_number_net.pt")

    stats = recognizer.read_all(img)
    print("blue:", stats["blue"])
    print("red:", stats["red"])
    # recognizer.debug_single_cell(img, team="blue", player_idx=0, stat_key="damage")
