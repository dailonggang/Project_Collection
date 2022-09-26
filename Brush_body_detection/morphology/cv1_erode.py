import cv2
import numpy as np
import matplotlib.pyplot as plt
# 侵蚀
# 作用：就是去除白色噪点，将两个连起来的形状打散
img = cv2.imread('img/j.png')
# print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为单通道
# 原图
# plt.imshow(gray, cmap='gray')
# plt.show()
# 侵蚀：作用就是去除白色噪点，将两个连起来的形状打散
# 3x3,1
kernel = np.ones((3, 3), dtype=np.int8)
ersion1 = cv2.erode(gray.copy(), kernel, iterations=1)
# plt.imshow(ersion1, cmap='gray')
# plt.show()

# 5x5,1
kernel = np.ones((5, 5), dtype=np.int8)
ersion2 = cv2.erode(gray.copy(), kernel, iterations=1)

# 5x5,2
kernel = np.ones((5, 5), dtype=np.int8)
ersion3 = cv2.erode(gray.copy(), kernel, iterations=2)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10, 4), sharex=True, sharey=True)
"""
nrows,ncols：子图的行列数。
sharex,sharey：
设置为 True 或者 ‘all’ 时，所有子图共享 x 轴或者 y 轴，
设置为 False or ‘none’ 时，所有子图的 x,y 轴均为独立，
设置为 ‘row’ 时，每一行的子图会共享 x 或者 y 轴，
设置为 ‘col’ 时，每一列的子图会共享 x 或者 y 轴。
squeeze：
默认为 True,是设置返回的子图对象的数组格式。
当为 False 时,不论返回的子图是只有一个还是只有一行,都会用二维数组格式返回他的对象。
当为 True 时,如果设置的子图是（nrows=ncols=1）,即子图只有一个,则返回的子图对象是一个标量的形式,如果子图有（N×1）或者（1×N）个,则返回的子图对象是一个一维数组的格式，如果是（N×M）则是返回二位格式。
subplot_kw:
字典格式,传递给 add_subplot(),用于创建子图。
gridspec_kw：
字典格式,传递给 GridSpec 的构造函数,用于创建子图所摆放的网格。
class matplotlib.gridspec.GridSpec(nrows, ncols, figure=None, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None)
如，设置 gridspec_kw={‘height_ratios’: [3, 1]} 则子图在列上的分布比例是3比1。
**fig_kw :
所有其他关键字参数都传递给 figure()调用。
如，设置 figsize=(21, 12) ，则设置了图像大小。
返回值
fig： matplotlib.figure.Figure 对象
ax：子图对象(matplotlib.axes.Axes)或者是他的数组
"""

ax1.axis('off')
ax1.imshow(gray.copy(), cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(ersion1, cmap='gray')
ax2.set_title('3x3,1')

ax3.axis('off')
ax3.imshow(ersion2, cmap='gray')
ax3.set_title('5x5,1')

ax4.axis('off')
ax4.imshow(ersion3, cmap='gray')
ax4.set_title('5x5,2')
plt.show()