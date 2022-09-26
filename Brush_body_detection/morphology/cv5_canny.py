# canny边缘检测算法是一种流行的边缘检测方法，由John F. Canny in发明
# 算法具体的推导超出本系列课程应用，但是他主要的步骤如下
# 1.高斯模糊降噪
# 2.使用Sobel filter计算图片像素梯度
# 3.NMS非最大值抑制计算局部最大值
# 4.Hysteresis thresholding 滞后阈值法过滤

# 其中canny的两个参数T_lower、T_upper就是这里的
# 其实一般我们在使用时，要注意的就是这两个值得选择：

# A 高于阈值 maxVal 所以是真正的边界点，C 虽然低于 maxVal 但高于minVal 并且与 A 相连，所以也被认为是真正的边界点。
# 而 B 就会被抛弃，因为他不仅低于 maxVal 而且不与真正的边界点相连。
# 所以选择合适的 maxVal和 minVal 对于能否得到好的结果非常重要。

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('img/pumpkin.jpg')
img_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_fixed)
"""
edges 为计算得到的边缘图像。
image 为 8 位输入图像。
threshold1 表示处理过程中的第一个阈值。
threshold2 表示处理过程中的第二个阈值。
apertureSize 表示 Sobel 算子的孔径大小。
L2gradient 为计算图像梯度幅度（gradient magnitude）的标识。其默认值为 False。
如果为 True，则使用更精确的 L2 范数进行计算（即两个方向的导数的平方和再开方），否则使用 L1 范数（直接将两个方向导数的绝对值相加）。
"""
edges1 = cv2.Canny(img.copy(), 100, 200)
edges2 = cv2.Canny(img.copy(), 50, 200)
edges3 = cv2.Canny(img.copy(), 50, 100)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img_fixed, cmap='gray')

ax2.axis('off')
ax2.imshow(edges1, cmap='gray')

ax3.axis('off')
ax3.imshow(edges2, cmap='gray')

ax4.axis('off')
ax4.imshow(edges3, cmap='gray')
plt.show()