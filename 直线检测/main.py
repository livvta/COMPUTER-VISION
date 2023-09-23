import cv2
import numpy as np

# 1. 读取图像并将其转换为灰度图像
image = cv2.imread('Hough.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 边缘检测（Canny算子）
edges = cv2.Canny(gray_image, 50, 150)  # 调整阈值根据图像的特性
cv2.imwrite('Cannyedge.jpg', edges)

# 3. 标准霍夫直线检测
lines_standard = cv2.HoughLines(edges, 1, np.pi / 180, 150)

# 创建一张纯白背景的图像
line_image_standard = np.zeros_like(image)

if lines_standard is not None:
    for rho, theta in lines_standard[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image_standard, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 用红色线段标识直线

# 概率霍夫直线检测
lines_probabilistic = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

# 创建一张纯白背景的图像
line_image_probabilistic = np.zeros_like(image)

if lines_probabilistic is not None:
    for line in lines_probabilistic:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image_probabilistic, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 用绿色线段标识直线

# 4. 保存检测直线的图像
cv2.imwrite('HoughLines.jpg', cv2.addWeighted(image, 0.8, line_image_standard, 1, 0))
cv2.imwrite('HoughLinesP.jpg', cv2.addWeighted(image, 0.8, line_image_probabilistic, 1, 0))

# 显示检测结果
cv2.imshow('HoughLines', cv2.addWeighted(image, 0.8, line_image_standard, 1, 0))
cv2.imshow('HoughLinesP', cv2.addWeighted(image, 0.8, line_image_probabilistic, 1, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()
