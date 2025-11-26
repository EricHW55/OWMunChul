import cv2

img = cv2.imread("test1.png")

# 1K(FHD) 사이즈
target_w, target_h = 1920, 1080

resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

cv2.imwrite("test1_1080p.png", resized)
print("변환 완료")
