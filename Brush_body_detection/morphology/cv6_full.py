# 检测书法文字
# 步骤：
# 1、读取图片，灰度、二值化处理
# 2、侵蚀去噪点
# 3、膨胀连接
# 4、闭合孔洞
# 5、边缘检测
# 6、画检测框

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 可通过rcParams方法人为控制显示图像大小
# plt.rcParams['figure.dpi'] = 150

# 读取
img = cv2.imread('img/shufa.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 显示灰度图
# plt.imshow(gray, cmap='gray')

"""
二值化
src：输入图片
thresh：比较的阈值
maxval：超出阈值被设定的值
type：模式
输出：设定阈值和输出图像
作用：将画面像素与比较阈值对比，小于阈值设为0（黑色），大于阈值设为目标值
"""
r, black_img = cv2.threshold(src=gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
# print(r, black_img)
# plt.imshow(black_img, cmap='gray')

# 边缘检测
edges = cv2.Canny(black_img, 30, 200)
# plt.imshow(edges, cmap='gray')
# plt.show()

# 找轮廓
"""
image-寻找轮廓的图像；
mode-轮廓的检索模式：
    cv2.RETR_EXTERNAL表示只检测外轮廓
    cv2.RETR_LIST检测的轮廓不建立等级关系
    cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE建立一个等级树结构的轮廓。
method-为轮廓的近似办法：
    cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，
例如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
返回值:
cv2.findContours()函数返回两个值，一个是轮廓本身(countours)，还有一个是每条轮廓对应的属性(hierarchy)。
cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。
"""
coutours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print(coutours)
# print('*'*50)
# print(h)

img_copy = img.copy()
for c in coutours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
# plt.imshow(img_copy)
# plt.show()
# 形态学变化
# 先对二值化后的图进行侵蚀，去除噪点
kernel = np.ones((3, 3), dtype=np.int8)
erosion1 = cv2.erode(black_img, kernel, iterations=1)
# plt.imshow(erosion1, cmap='gray')

# 再膨胀
kernel = np.ones((10, 10), dtype=np.int8)
dilation = cv2.dilate(erosion1, kernel, iterations=2)
# plt.imshow(dilation, cmap='gray')

# 闭合
kernel = np.ones((10, 10), dtype=np.int8)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
# plt.imshow(closing, cmap='gray')

# 边缘检测
edges1 = cv2.Canny(closing, 30, 200)
# plt.imshow(edges1, cmap='gray')

# 找轮廓
coutours1, h = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print(coutours1)
img_copy1 = img.copy()
for c in coutours1:
    # print(c)
    x, y, w, h = cv2.boundingRect(c)
    if w > 80:  # 去除过小矩形
        cv2.rectangle(img_copy1, (x, y), (x+w, y+h), (0, 255, 0), 3)
# plt.imshow(img_copy)

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(1, 9, figsize=(14, 5), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(gray, cmap='gray')
ax1.set_title('orginal image')

ax2.axis('off')
ax2.imshow(black_img, cmap='gray')
ax2.set_title('Bina')

ax3.axis('off')
ax3.imshow(edges, cmap='gray')
ax3.set_title('Bin_canny')

ax4.axis('off')
ax4.imshow(img_copy, cmap='gray')
ax4.set_title('contours')

ax5.axis('off')
ax5.imshow(erosion1, cmap='gray')
ax5.set_title('erode')

ax6.axis('off')
ax6.imshow(dilation, cmap='gray')
ax6.set_title('dilation')

ax7.axis('off')
ax7.imshow(closing, cmap='gray')
ax7.set_title('closing')

ax8.axis('off')
ax8.imshow(edges1, cmap='gray')
ax8.set_title('edges1')

ax9.axis('off')
ax9.imshow(img_copy1, cmap='gray')
ax9.set_title('contours')

plt.show()

