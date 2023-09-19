import cv2
import numpy as np
image = cv2.imread('image1.jpg')

# 大值法、平均值法灰度化
h, w = image.shape[: -1]
gray_max = np.zeros((h, w), dtype=np.uint8)
gray_avg = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        gray_max[i, j] = np.max([image[i, j, 0], image[i, j, 1], image[i, j, 2]])
        gray_avg[i, j] = (int(image[i, j][0]) + int(image[i, j][1]) + int(image[i, j][2])) // 3

# 加权平均法灰度化
gray_weighted = (image[:, :, 0] * 0.11 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.3).astype(np.uint8)

gray_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_cv2.jpg', gray_cv2)
cv2.imwrite('gray_max.jpg', gray_max)
cv2.imwrite('gray_avg.jpg', gray_avg)
cv2.imwrite('gray_weighted.jpg', gray_weighted)
