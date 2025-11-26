import cv2
from paddleocr import PaddleOCR
from crop_coordinates import OWScoreboardCropper


class OWStatsRecognizer:
    def __init__(self, cropper=None):
        """
        PaddleOCR 3.x용 숫자 전용 리더.
        - detection + recognition 파이프라인을 그대로 쓰지만
          우리는 이미 숫자 영역만 crop 해서 넣어줌.
        """
        self.cropper = cropper or OWScoreboardCropper()

        # 기준 해상도 (cropper에서 쓰는 값과 통일)
        self.REF_W = self.cropper.REF_W
        self.REF_H = self.cropper.REF_H

        # 최신 PaddleOCR 3.x 스타일 초기화
        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",              # 배포는 CPU 기준
            text_rec_score_thresh=0.0
        )

        self.stat_keys = ["kills", "assists", "deaths", "damage", "heal", "mitig"]

    # ---------------- 기본 유틸 ---------------- #

    def _normalize_image(self, img):
        """
        어떤 해상도로 들어오든 간에
        cropper 기준 해상도(REF_W, REF_H)로 통일.
        """
        h, w = img.shape[:2]
        if w == self.REF_W and h == self.REF_H:
            return img

        # 전체 이미지 비율 통일 (약간 찌그러져도 숫자/좌표 일관성이 더 중요)
        resized = cv2.resize(
            img,
            (self.REF_W, self.REF_H),
            interpolation=cv2.INTER_AREA
        )
        return resized

    def _crop_box(self, img, box):
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]

    def _preprocess_for_ocr(self, crop):
        """
        PaddleOCR는 자체 전처리를 해주기 때문에
        여기서는:
        1) 모든 숫자 박스를 같은 높이(target_h)로 맞춰서
           2k/1k간 글자 크기 차이를 줄임.
        2) aspect ratio는 유지 (가로 방향 왜곡 최소화)
        """
        h, w = crop.shape[:2]
        target_h = 64  # PaddleOCR가 보기 좋은 대략적인 글자 높이

        if h <= 0 or w <= 0:
            return crop

        scale = target_h / float(h)
        # upsample/downsample 모두 같은 방식으로 처리
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA

        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=interp
        )

        return resized

    # ---------------- 숫자 OCR ---------------- #

    def _parse_int_from_text(self, text: str):
        """
        PaddleOCR가 반환한 문자열에서 정수만 안전하게 파싱.
        예: '9,136' → 9136, 'O9' → 9, '*' 등은 무시
        """
        if not text:
            return None

        # 공백/쉼표 제거
        text = text.strip().replace(",", "")

        # 숫자만 남기기
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits == "":
            return None

        try:
            return int(digits)
        except ValueError:
            return None

    def _ocr_number(self, crop):
        """
        하나의 crop(숫자 박스)에 대해 PaddleOCR로 숫자를 읽어 int 리턴.
        실패하면 0 리턴 (필요하면 None으로 바꿔도 됨).
        """
        img_for_ocr = self._preprocess_for_ocr(crop)

        # PaddleOCR 3.x에서는 predict() 사용
        # result: List[dict], 각 dict에 "rec_texts", "rec_scores", ...
        result = self.ocr.predict(img_for_ocr)

        if not result:
            return 0

        line = result[0]

        # PaddleOCR 3.x: line은 dict 형식
        texts = None
        if isinstance(line, dict) and "rec_texts" in line:
            texts = line.get("rec_texts") or []
        else:
            # 혹시 구버전 API (2.x) 호환용 fallback
            texts = []
            for item in line:
                # 옛날 형식: (box, (text, score))
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _, txt_conf = item
                    if isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 1:
                        texts.append(str(txt_conf[0]))
                    else:
                        texts.append(str(txt_conf))

        if not texts:
            return 0

        raw_text = texts[0]
        value = self._parse_int_from_text(raw_text)

        if value is None:
            return 0
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
        # 1) 먼저 기준 해상도로 통일
        img_norm = self._normalize_image(img)

        # 2) 통일된 이미지 기준으로 좌표 계산
        boxes = self.cropper.get_player_boxes(img_norm)
        result = {"blue": [], "red": []}

        for team in ["blue", "red"]:
            for slot_boxes in boxes[team]:
                slot_result = {
                    "hero": slot_boxes["hero"]  # 영웅은 좌표만 유지
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
        특정 한 칸만 떼서 crop / PaddleOCR 결과 확인용.
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
    img_path = r"testdata/test1.png"  # 1k든 2k든 상관 없음
    img = cv2.imread(img_path)

    cropper = OWScoreboardCropper()
    recognizer = OWStatsRecognizer(cropper=cropper)

    stats = recognizer.read_all(img)
    print("blue:", stats["blue"])
    print("red:", stats["red"])

    # 디버그용
    # recognizer.debug_single_cell(img, team="blue", player_idx=0, stat_key="damage")
