import cv2
import numpy as np

image = cv2.imread('image2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_image = np.zeros_like(gray_image)
hist = np.zeros(256)

# 计算直方图
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        hist[gray_image[i, j]] += 1

cdf = hist.cumsum() / (gray_image.shape[0] * gray_image.shape[1])

# 应用均衡化
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        equalized_image[i, j] = np.round(255 * cdf[gray_image[i, j]])


equalized_image_cv2 = cv2.equalizeHist(gray_image)
cv2.imwrite('equalized.jpg', equalized_image)
cv2.imwrite('equalized_cv2.jpg', equalized_image_cv2)
