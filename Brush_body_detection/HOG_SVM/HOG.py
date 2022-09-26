# 导入必要包
import cv2
import matplotlib.pyplot as plt

# 导入skimage
from skimage.feature import hog
from skimage import data, exposure

img = cv2.imread('../../img/dog.jpg')
img = cv2.resize(img, (500, 370))

img_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fd, hog_image = hog(image=img_gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)
"""
image：输入图像
orientations：把180度分成几份，bin的数量
pixels_per_cell ：元组形式，一个Cell内的像素大小
cells_per_block： 元组形式，一个Block内的Cell大小
visualize： 是否需要可视化，如果True，hog会返回numpy图像
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img_fixed)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# 查看一下HOG特征大小
print(fd.shape)
print(fd)