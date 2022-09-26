import numpy as np
import matplotlib.pyplot as plt
import cv2
# opening,张开
# 侵蚀+膨胀
# 主要用于清除噪点

img = cv2.imread('img/cv.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray,cmap='gray')

kernel = np.ones((10, 10), dtype=np.int8)
opening1 = cv2.morphologyEx(gray.copy(), cv2.MORPH_OPEN, kernel)

kernel = np.ones((12, 12), dtype=np.int8)
opening2 = cv2.morphologyEx(gray.copy(), cv2.MORPH_OPEN, kernel)

kernel = np.ones((15, 15), dtype=np.int8)
opening3 = cv2.morphologyEx(gray.copy(), cv2.MORPH_OPEN, kernel)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharex=True,sharey=True)

ax1.axis('off')
ax1.imshow(gray.copy(), cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(opening1, cmap='gray')
ax2.set_title('10x10')

ax3.axis('off')
ax3.imshow(opening2, cmap='gray')
ax3.set_title('12x12')

ax4.axis('off')
ax4.imshow(opening3, cmap='gray')
ax4.set_title('15x15')

plt.show()