import cv2
import numpy as np

image = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

# 均值滤波
def mean_filter(image, winsize):
    if winsize % 2 == 0:
        winsize += 1

    winsize_2 = winsize // 2
    winsize_num = winsize * winsize
    image_board = cv2.copyMakeBorder(image, winsize_2, winsize_2, winsize_2, winsize_2, cv2.BORDER_REFLECT)
    row, col = image_board.shape[:2]
    mean_out = np.zeros_like(image, dtype=np.uint8)

    for i in range(winsize_2, row - winsize_2):
        for j in range(winsize_2, col - winsize_2):
            pixel_sum = 0.0

            for y in range(winsize):
                for x in range(winsize):
                    pixel_sum += image_board[i - winsize_2 + y, j - winsize_2 + x]

            mean_out[i - winsize_2, j - winsize_2] = int(pixel_sum / winsize_num + 0.5)

    return mean_out


mean_out = mean_filter(image, winsize=3)
mean_out_cv2 = cv2.blur(image, (3, 3))


# 中值滤波
def median_filter(image):
    median_out = image.copy()
    height, width = image.shape

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            k = 0
            window = [0] * 9

            for y in range(j - 1, j + 2):
                for x in range(i - 1, i + 2):
                    if 0 <= x < width and 0 <= y < height:
                        window[k] = image[y, x]
                    else:
                        window[k] = image[j, i]
                    k += 1

            for m in range(5):
                min_value = m
                for n in range(m + 1, 9):
                    if window[n] < window[min_value]:
                        min_value = n

                temp = window[m]
                window[m] = window[min_value]
                window[min_value] = temp

            median_out[j, i] = window[4]

    return median_out


median_out = median_filter(image)
median_out_cv2 = cv2.medianBlur(image, 3)


# 高斯滤波
def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    return kernel / kernel.sum()


def gaussian_filter(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    rows, cols = image.shape
    gaussian_out = np.zeros((rows, cols))
    pad_size = kernel_size // 2

    for i in range(pad_size, rows - pad_size):
        for j in range(pad_size, cols - pad_size):
            gaussian_out[i, j] = (kernel * image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]).sum()

    return gaussian_out.astype(np.uint8)


kernel_size = 3
sigma = 1.3
gaussian_out = gaussian_filter(image, kernel_size, sigma)
gaussian_out_cv2 = cv2.GaussianBlur(image, (3, 3), 1.3)

cv2.imshow("mean_cv2", mean_out_cv2)
cv2.imshow('mean', mean_out)
cv2.imwrite('mean.jpg', mean_out)
cv2.imwrite("mean_cv2.jpg", mean_out_cv2)

cv2.imshow('median', median_out)
cv2.imshow('median_cv2', median_out_cv2)
cv2.imwrite('median.jpg', median_out)
cv2.imwrite('median_cv2.jpg', median_out_cv2)

cv2.imwrite('gaussian.jpg', gaussian_out)
cv2.imwrite('gaussian_cv2.jpg', gaussian_out_cv2)
cv2.imshow('gaussian', gaussian_out)
cv2.imshow('gaussian_cv2', gaussian_out_cv2)

cv2.waitKey(0)
cv2.destroyAllWindows()
