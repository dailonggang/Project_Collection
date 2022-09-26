import cv2
import matplotlib.pyplot as plt
import numpy as np
# 膨胀
# 作用：跟在侵蚀操作后去噪点，把两个分开的部分连接起来
img = cv2.imread('img/j.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray.copy(), cmap='gray')
# 3x3,1
kernel = np.ones((3, 3), dtype=np.int8)
dilation1 = cv2.dilate(gray.copy(), kernel, iterations=1)

# 5x5,1
kernel = np.ones((5, 5), dtype=np.int8)
dilation2 = cv2.dilate(gray.copy(), kernel, iterations=1)

# 5x5,2
kernel = np.ones((5, 5), dtype=np.int8)
dilation3 = cv2.dilate(gray.copy(), kernel, iterations=2)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray.copy(), cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(dilation1, cmap='gray')
ax2.set_title('3x3,1')

ax3.axis('off')
ax3.imshow(dilation2, cmap='gray')
ax3.set_title('5x5,1')

ax4.axis('off')
ax4.imshow(dilation3, cmap='gray')
ax4.set_title('5x5,2')

plt.show()