import os
import csv
import random
import cv2
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ========= 하이퍼파라미터 / 상수 ========= #

MAX_LEN = 5         # 최대 자리수 (0~99999까지 커버)
NUM_CLASSES = 11    # 0~9 + blank
BLANK = 10          # blank 토큰 인덱스

TARGET_H = 32       # 입력 높이
TARGET_W = 128      # 입력 너비

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42


# ========= 유틸 함수 ========= #

def encode_number(value: int) -> torch.Tensor:
    """
    정수 value를 길이 MAX_LEN의 시퀀스로 인코딩.
    예: 1214 -> [10, 1, 2, 1, 4] (오른쪽 정렬, 왼쪽 BLANK)
    """
    s = str(int(value))
    s = s.replace(",", "")
    digits = [int(c) for c in s if c.isdigit()]

    if len(digits) > MAX_LEN:
        # 너무 길면 뒤에서부터 MAX_LEN 자리만 사용
        digits = digits[-MAX_LEN:]

    pad = [BLANK] * (MAX_LEN - len(digits))
    encoded = pad + digits
    return torch.tensor(encoded, dtype=torch.long)


def preprocess_image(path: str) -> torch.Tensor:
    """
    crop 이미지를 불러와서 (1, TARGET_H, TARGET_W) 텐서로 변환.
    - 그레이스케일
    - 높이 TARGET_H로 비율 유지 리사이즈
    - 좌우 패딩으로 폭 TARGET_W 맞추기
    - [0,1] 범위로 정규화
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise RuntimeError(f"Empty image: {path}")

    # 높이 기준 리사이즈
    scale = TARGET_H / float(h)
    new_w = max(1, int(w * scale))
    if scale > 1.0:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_AREA

    img_resized = cv2.resize(img, (new_w, TARGET_H), interpolation=interp)

    # 폭 TARGET_W 맞추기 (양 옆 패딩)
    if new_w > TARGET_W:
        # 너무 넓으면 그냥 TARGET_W로 다시 리사이즈
        img_resized = cv2.resize(img_resized, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    else:
        pad_left = (TARGET_W - new_w) // 2
        pad_right = TARGET_W - new_w - pad_left
        img_resized = cv2.copyMakeBorder(
            img_resized,
            top=0,
            bottom=0,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = img_resized[None, :, :]  # (1, H, W)

    return torch.from_numpy(img_resized)


# ========= Dataset 클래스 ========= #

class ImgToNumberDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, stat_filter: List[str] = None):
        """
        csv_path: labels.csv 경로
        img_dir:  crop 이미지들이 들어 있는 디렉토리 (ex. dataset/img_to_number/images)
        stat_filter: ["damage", "heal"] 등 특정 stat만 학습하고 싶을 때 필터
        """
        self.img_dir = img_dir
        self.items: List[Tuple[str, int]] = []  # (파일명, value)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                value = int(row["value"])
                stat_key = row["stat_key"]

                if stat_filter is not None and stat_key not in stat_filter:
                    continue

                self.items.append((filename, value))

        if not self.items:
            raise RuntimeError("No items found in dataset. Check csv_path / stat_filter.")

        print(f"[DATASET] Loaded {len(self.items)} samples from {csv_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, value = self.items[idx]
        img_path = os.path.join(self.img_dir, filename)

        img_tensor = preprocess_image(img_path)  # (1, H, W)
        target = encode_number(value)           # (MAX_LEN,)

        return img_tensor, target


# ========= 모델 정의 ========= #

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

        # 현재 크기 계산 (TARGET_H=32, TARGET_W=128 기준)
        # 32 -> 16 -> 8 -> 4   /  128 -> 64 -> 32 -> 16
        conv_out_h = TARGET_H // 8
        conv_out_w = TARGET_W // 8
        conv_out_dim = 128 * conv_out_h * conv_out_w

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
        x = x.view(-1, MAX_LEN, NUM_CLASSES)  # (B, L, C)
        return x


# ========= 학습 / 평가 유틸 ========= #

def compute_loss(logits, targets):
    # logits: (B, L, C), targets: (B, L)
    B, L, C = logits.shape
    logits_flat = logits.view(B * L, C)
    targets_flat = targets.view(B * L)
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_digits = 0
    correct_digits = 0
    total_seqs = 0
    correct_seqs = 0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            loss = compute_loss(logits, targets)
            total_loss += loss.item() * imgs.size(0)

            # digit accuracy
            preds = logits.argmax(dim=-1)  # (B,L)
            correct_digits += (preds == targets).sum().item()
            total_digits += preds.numel()

            # full sequence accuracy
            seq_equal = (preds == targets).all(dim=1)  # (B,)
            correct_seqs += seq_equal.sum().item()
            total_seqs += imgs.size(0)

    avg_loss = total_loss / total_seqs
    digit_acc = correct_digits / total_digits
    seq_acc = correct_seqs / total_seqs

    return avg_loss, digit_acc, seq_acc


# ========= 메인 학습 루프 ========= #

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    csv_path = "dataset/img_to_number/labels.csv"
    img_dir = "dataset/img_to_number/images"

    # 필요하면 특정 stat만 학습할 수도 있음 (ex. ["damage"])
    stat_filter = None  # 또는 ["damage", "heal"]

    dataset = ImgToNumberDataset(csv_path, img_dir, stat_filter=stat_filter)

    # train / val split
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * (1.0 - VAL_RATIO))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = ScoreNumberNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_seq_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = compute_loss(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        val_loss, val_digit_acc, val_seq_acc = evaluate(model, val_loader, device)

        print(f"[EPOCH {epoch:02d}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_digit_acc={val_digit_acc*100:.2f}%  "
              f"val_seq_acc={val_seq_acc*100:.2f}%")

        # 베스트 모델 저장 (전체 시퀀스 정확도 기준)
        if val_seq_acc > best_val_seq_acc:
            best_val_seq_acc = val_seq_acc
            os.makedirs("checkpoints", exist_ok=True)
            save_path = os.path.join("checkpoints", "score_number_net.pt")
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_seq_acc": val_seq_acc,
                "config": {
                    "MAX_LEN": MAX_LEN,
                    "NUM_CLASSES": NUM_CLASSES,
                    "BLANK": BLANK,
                    "TARGET_H": TARGET_H,
                    "TARGET_W": TARGET_W,
                }
            }, save_path)
            print(f"  -> Saved best model to {save_path}")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
