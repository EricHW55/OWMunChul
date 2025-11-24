import cv2
import numpy as np
import pytesseract
from crop_coordinates import OWScoreboardCropper


class OWStatsRecognizer:
    def __init__(self, cropper=None, ocr_config=None):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        """
        cropper: OWScoreboardCropper 인스턴스 (없으면 내부에서 새로 생성)
        ocr_config: tesseract 설정 문자열
        """
        self.cropper = cropper or OWScoreboardCropper()
        # 숫자 + 쉼표만, 한 줄짜리 숫자 박스라고 가정
        self.ocr_config = ocr_config or r"--psm 7 -c tessedit_char_whitelist=0123456789,"

        # 윈도우에서 tesseract 경로 지정이 필요하면 여기서 설정
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # 읽을 스탯 키 순서
        self.stat_keys = ["kills", "assists", "deaths", "damage", "heal", "mitig"]

    def _crop_box(self, img, box):
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]

    def _preprocess_for_ocr(self, crop):
        """
        숫자 OCR을 위해 간단 전처리: 그레이스케일 + OTSU 이진화
        필요하면 여기에서 blur, dilation 등 추가 튜닝 가능
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 배경색/글자색 테스트해보고 THRESH_BINARY ↔ THRESH_BINARY_INV 필요하면 바꿔보기
        _, thr = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    def _ocr_number(self, crop):
        """
        crop: BGR 또는 그레이/이진 이미지
        return: int 또는 None
        """
        proc = self._preprocess_for_ocr(crop)
        text = pytesseract.image_to_string(proc, config=self.ocr_config)
        text = text.strip()
        text = text.replace(",", "")  # 9,136 → 9136

        if text == "":
            return 0  # 힐량/경감 0 같은 경우가 많아서 0으로 처리 (원하면 None으로 바꿔도 됨)

        # 숫자 아닌 문자가 섞이면 제거
        filtered = "".join(ch for ch in text if ch.isdigit())
        if filtered == "":
            return None

        try:
            return int(filtered)
        except ValueError:
            return None

    def read_all(self, img):
        """
        img: BGR 이미지
        return:
            {
              "blue": [ {hero:..., kills:int, ...}, ... 5명 ],
              "red":  [ ... 5명 ]
            }
        hero는 좌표를 그대로 두고, 스탯만 숫자로 바꿈.
        """
        boxes = self.cropper.get_player_boxes(img)
        result = {"blue": [], "red": []}

        for team in ["blue", "red"]:
            for slot_boxes in boxes[team]:
                slot_result = {"hero": slot_boxes["hero"]}  # hero는 좌표 유지

                for key in self.stat_keys:
                    crop = self._crop_box(img, slot_boxes[key])
                    value = self._ocr_number(crop)
                    slot_result[key] = value

                result[team].append(slot_result)

        return result

    # 디버깅용: 특정 칸 하나만 읽어서 확인
    def debug_single_cell(self, img, team="blue", player_idx=0, stat_key="damage"):
        boxes = self.cropper.get_player_boxes(img)
        box = boxes[team][player_idx][stat_key]
        crop = self._crop_box(img, box)
        proc = self._preprocess_for_ocr(crop)
        value = self._ocr_number(crop)

        cv2.imshow("crop", crop)
        cv2.imshow("proc", proc)
        print(f"{team}[{player_idx}] {stat_key} =>", value)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return value


# -------- 사용 예시 --------
if __name__ == "__main__":
    img_path = r"dataset/blue/2025-11-25 030028.png"
    img = cv2.imread(img_path)

    cropper = OWScoreboardCropper()
    recognizer = OWStatsRecognizer(cropper=cropper)

    # 전체 읽기
    stats = recognizer.read_all(img)
    print("blue 1st player:", stats["blue"][0])
    print("red 5th player:", stats["red"][4])

    # 디버깅: 파란 2번 피해 값만 따로 확인해보기
    # recognizer.debug_single_cell(img, team="blue", player_idx=1, stat_key="damage")
