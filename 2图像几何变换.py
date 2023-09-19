import numpy as np
import cv2 as cv

# 读取图像
image = cv.imread('image2.jpg', cv.IMREAD_COLOR)
cv.imshow('Original', image)

# 对image进行平移、缩放和旋转变换，并显示和保存
h, w, c = image.shape[:3]
S1 = np.float32([[1, 0, 100], [0, 1, 100]])  # 平移矩阵
S2 = np.float32([[0.5, 0, w*0.25], [0, 0.5, h*0.25]])  # 缩放矩阵
S3 = cv.getRotationMatrix2D((w/2, h/2), 45, 1)  # 旋转矩阵

Translation = cv.warpAffine(image, S1, (w, h))
Scaling = cv.warpAffine(image, S2, (w, h))
Rotation = cv.warpAffine(image, S3, (w, h))

cv.imshow("Translation", Translation)
cv.imshow("Scaling", Scaling)
cv.imshow("Rotation", Rotation)

cv.imwrite("Translation.jpg", Translation)
cv.imwrite("Scaling.jpg", Scaling)
cv.imwrite("Rotation.jpg", Rotation)

cv.waitKey(0)
cv.destroyAllWindows()

