import cv2
import numpy as np

def RGB2HSI(rgb_img):
    rgb_img = rgb_img.astype(float) / 255
    b, g, r = cv2.split(rgb_img)
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(numerator / (denominator + 1e-5))
    h = theta.copy()
    h[b > g] = 2 * np.pi - h[b > g]
    h /= 2 * np.pi
    s = 1 - 3 * np.minimum(np.minimum(r, g), b) / (r + g + b + 1e-5)
    i = (r + g + b) / 3
    hsi_img = cv2.merge((h, s, i))
    hsi_img = (hsi_img * 255).astype(np.uint8)

    return hsi_img

def RGB2HSV(rgb_img):
    hsv_img = np.zeros_like(rgb_img, dtype=np.uint8)

    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            B = rgb_img[i, j, 0] / 255.0
            G = rgb_img[i, j, 1] / 255.0
            R = rgb_img[i, j, 2] / 255.0
            V = max(B, G, R)
            vmin = min(B, G, R)
            diff = V - vmin
            S = diff / (abs(V) + np.finfo(float).eps)
            diff = 60.0 / (diff + np.finfo(float).eps)
            if V == B:
                H = 240.0 + (R - G) * diff
            elif V == G:
                H = 120.0 + (B - R) * diff
            elif V == R:
                H = (G - B) * diff
            H = H + 360.0 if H < 0.0 else H
            hsv_img[i, j, 0] = int(H / 2)
            hsv_img[i, j, 1] = int(S * 255)
            hsv_img[i, j, 2] = int(V * 255)
    return hsv_img


rgb_img = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
hsi_img = RGB2HSI(rgb_img)
hsv_img = RGB2HSV(rgb_img)
hsv_img_cv2 = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

cv2.imshow("Origin", rgb_img)
cv2.imshow("HSI", hsi_img)
cv2.imshow("HSV", hsv_img)
cv2.imshow("HSV_cv2", hsv_img_cv2)
cv2.imwrite("HSI.jpg", hsi_img)
cv2.imwrite("HSV.jpg", hsv_img)
cv2.imwrite("HSV_cv2.jpg",hsv_img_cv2)
#cv2.cvtColor无法直接转换HSI，故无对比

cv2.waitKey()
cv2.destroyAllWindows()