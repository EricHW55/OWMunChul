import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crop_coordinates import OWScoreboardCropper


# ====== 학습 때 쓴 하이퍼파라미터와 맞춰야 함 ====== #
MAX_LEN = 5         # 최대 자리수
NUM_CLASSES = 11    # 0~9 + blank
BLANK = 10          # blank 토큰 인덱스

TARGET_H = 32
TARGET_W = 128


# ====== 모델 정의 (train 스크립트와 동일해야 함) ====== #

class ScoreNumberNet(nn.Module):
    """
    작은 CNN 기반 숫자 시퀀스 인식기.
    입력: (B, 1, TARGET_H, TARGET_W)
    출력: (B, MAX_LEN, NUM_CLASSES)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2, W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/4, W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/8, W/8
        )

        # 32x128 기준: 32 -> 16 -> 8 -> 4 / 128 -> 64 -> 32 -> 16
        conv_out_h = TARGET_H // 8  # 4
        conv_out_w = TARGET_W // 8  # 16
        conv_out_dim = 128 * conv_out_h * conv_out_w  # 128*4*16

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, MAX_LEN * NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, MAX_LEN, NUM_CLASSES)
        return x


# ====== 전처리 / 디코딩 유틸 ====== #

def preprocess_crop_to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR crop 이미지를 (1, 1, TARGET_H, TARGET_W) 텐서로 변환.
    - 그레이스케일
    - 높이 TARGET_H로 비율 유지 리사이즈
    - 좌우 패딩으로 폭 TARGET_W 맞추기
    - [0,1] float32
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise RuntimeError("Empty crop image")

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise RuntimeError("Invalid crop size")

    # 높이 기준 리사이즈
    scale = TARGET_H / float(h)
    new_w = max(1, int(w * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(gray, (new_w, TARGET_H), interpolation=interp)

    # 좌우 패딩으로 TARGET_W 맞추기
    if new_w > TARGET_W:
        resized = cv2.resize(resized, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    else:
        pad_left = (TARGET_W - new_w) // 2
        pad_right = TARGET_W - new_w - pad_left
        resized = cv2.copyMakeBorder(
            resized,
            top=0,
            bottom=0,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

    resized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized[None, None, :, :])  # (1,1,H,W)
    return tensor


def decode_digits(logits: torch.Tensor) -> int:
    """
    logits: (1, MAX_LEN, NUM_CLASSES)
    → argmax로 자리수별 클래스 뽑아서 BLANK 제거 후 int 변환
    """
    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)  # (1, L)
    preds = preds[0].tolist()     # 길이 L 리스트

    digits = []
    for p in preds:
        if p == BLANK:
            continue
        digits.append(str(p))

    if not digits:
        return 0
    try:
        return int("".join(digits))
    except ValueError:
        return 0


# ====== 메인 래퍼 클래스 ====== #

class OWStatsRecognizer:
    def __init__(self, cropper=None, ckpt_path="checkpoints/score_number_net.pt"):
        """
        CNN 기반 숫자 리더.
        - cropper로 각 칸 좌표를 구한 뒤
        - crop → (32x128) 텐서 → ScoreNumberNet → 정수 디코딩
        """
        self.cropper = cropper or OWScoreboardCropper()

        # 기준 해상도 (cropper에서 쓰는 값과 통일)
        self.REF_W = self.cropper.REF_W
        self.REF_H = self.cropper.REF_H

        self.stat_keys = ["kills", "assists", "deaths", "damage", "heal", "mitig"]

        # 디바이스 / 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScoreNumberNet().to(self.device)
        self.model.eval()

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state_dict)
        print(f"[INFO] Loaded CNN checkpoint from {ckpt_path}")

    # ---------------- 기본 유틸 ---------------- #

    def _normalize_image(self, img):
        """
        어떤 해상도로 들어오든 간에
        cropper 기준 해상도(REF_W, REF_H)로 통일.
        """
        h, w = img.shape[:2]
        if w == self.REF_W and h == self.REF_H:
            return img

        resized = cv2.resize(
            img,
            (self.REF_W, self.REF_H),
            interpolation=cv2.INTER_AREA
        )
        return resized

    def _crop_box(self, img, box):
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]

    # ---------------- 숫자 인식 ---------------- #

    def _ocr_number(self, crop_bgr):
        """
        하나의 crop(숫자 박스)에 대해 CNN으로 숫자를 읽어 int 리턴.
        실패하면 0 리턴.
        """
        try:
            tensor = preprocess_crop_to_tensor(crop_bgr).to(self.device)  # (1,1,H,W)
        except RuntimeError:
            return 0

        with torch.no_grad():
            logits = self.model(tensor)  # (1, L, C)
        value = decode_digits(logits)
        return value

    # ---------------- 전체 스탯 읽기 ---------------- #

    def read_all(self, img):
        """
        img: BGR 이미지 (원본 해상도 상관 없음)
        return:
            {
              "blue": [ {hero: (x0,y0,x1,y1), kills:int, ...} * 5 ],
              "red":  [ ... ]
            }
        """
        img_norm = self._normalize_image(img)

        boxes = self.cropper.get_player_boxes(img_norm)
        result = {"blue": [], "red": []}

        for team in ["blue", "red"]:
            for slot_boxes in boxes[team]:
                slot_result = {
                    "hero": slot_boxes["hero"]
                }

                for key in self.stat_keys:
                    crop = self._crop_box(img_norm, slot_boxes[key])
                    value = self._ocr_number(crop)
                    slot_result[key] = value

                result[team].append(slot_result)

        return result

    # ---------------- 디버깅용 ---------------- #

    def debug_single_cell(self, img, team="blue", player_idx=0, stat_key="damage"):
        """
        특정 한 칸만 떼서 crop / CNN 결과 확인용.
        """
        img_norm = self._normalize_image(img)
        boxes = self.cropper.get_player_boxes(img_norm)
        box = boxes[team][player_idx][stat_key]
        crop = self._crop_box(img_norm, box)

        value = self._ocr_number(crop)

        cv2.imshow("crop", crop)
        print(f"{team}[{player_idx}] {stat_key} =>", value)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return value


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
