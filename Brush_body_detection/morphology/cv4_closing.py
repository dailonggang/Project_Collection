# 闭合：closing
# 先膨胀再侵蚀，主要用于闭合主体内的小洞，或者一些黑色的点
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('img/cv.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray.copy(), cmap='gray')

kernel = np.ones((20, 20), dtype=np.int8)
closing1 = cv2.morphologyEx(gray.copy(), cv2.MORPH_CLOSE, kernel)
plt.imshow(closing1, cmap='gray')
plt.show()