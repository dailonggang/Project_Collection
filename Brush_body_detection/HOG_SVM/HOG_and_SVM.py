# 项目将书法图片分为五类（篆书、隶书、草书、行书、楷书）
"""
读取数据
提取HOG特征
送至SVM训练
评估模型
保存模型
可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier


# 解决中文路径
def readImg(filepath):
    raw_data = np.fromfile(filepath, dtype=np.uint8)
    img = cv2.imdecode(raw_data, -1)
    return img


img_1 = readImg('images/行书/丙/敬世江_5945d30c02c1e30a89de9dc3920a0011adc9aa46.jpg')
img_transcol = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
# plt.imshow(img_transcol)
# plt.show()


# 缩放照片尺度
def resizeGray(img, new_size):
    img = cv2.resize(img, new_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
image = resizeGray(img_1, (200, 200))


from skimage.feature import hog
from skimage import data, exposure
"""使用HOG提取特征"""
fd, hog_image = hog(image, orientations=4, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# # Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

"""批量读取文件数据，提取HOG特征"""
import os
import glob


# 列出目录下的文件
# def listdir_nohidden(path):
#     return glob.glob(os.path.join(path, '*'))


# 读取文件
def image_reader(file_name, new_size):
    img = readImg(file_name)
    img = resizeGray(img, new_size)
    return img


# 从文件中读取特征和标签
def get_file_hog_label_list_from_disk(selectNum=1000):
    char_styles = ['篆书', '隶书', '草书', '行书', '楷书']
    # 特征列表和标签列表
    fileFeaturesList = []
    fileLabelList = []
    # 遍历各种风格
    for style in char_styles:
        file_list = glob.glob('./images/' + style + '/*/*')
        print('风格：{style}下共有{num}张图片\n'.format(style=style, num=len(file_list)))
        # 打乱顺序
        random.shuffle(file_list)
        # 挑选固定数量文件
        select_files = file_list[:selectNum]

        # 挑选指定数量文件
        for file_item in select_files:
            # 读取文件
            img = image_reader(file_item, (100, 100))
            # 提取特征
            features = hog(img, orientations=4, pixels_per_cell=(6, 6), cells_per_block=(2, 2))
            features = list(features)

            fileFeaturesList.append(features)
            fileLabelList.append(char_styles.index(style))

        print('风格：{style}，共挑选了{num}张图片\n\n'.format(style=style, num=len(select_files)))
    return fileFeaturesList, fileLabelList


from sklearn import svm
from sklearn.model_selection import train_test_split
# 每个字最多挑选1个
fileFeaturesList, fileLabelList = get_file_hog_label_list_from_disk(selectNum=1000)

# 将样本分为训练和测试样本
# 训练样本特征、测试样本特征、训练样本类别标签、测试样本类别标签
x_train, x_test, y_train, y_test = train_test_split(fileFeaturesList, fileLabelList,
                                                  test_size=0.25, random_state=42)
# print(len(x_train), len(x_test), len(y_train), len(y_test))

from sklearn.metrics import accuracy_score
# 统计各种类别数量
from collections import Counter
Counter(fileLabelList)

# SVM分类器
cls = svm.SVC(kernel='rbf')
cls.fit(x_train, y_train)  # 训练
predictLabels = cls.predict(x_test)
print("svm acc:%s" % accuracy_score(y_test, predictLabels))

cls1 = svm.SVC(kernel='linear')
cls1.fit(x_train, y_train)
predictLabels = cls1.predict(x_test)
print("svm 1 acc:%s" % accuracy_score(y_test, predictLabels))


cls2 = svm.SVC(kernel='poly')
cls2.fit(x_train, y_train)
predictLabels = cls2.predict(x_test)
print("svm  2 acc:%s" % accuracy_score(y_test, predictLabels))
#
#

# # KNN
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(x_train, y_train)
# predictLabels = neigh.predict(x_test)
# print("KNN acc:%s" % accuracy_score(y_test,predictLabels))
#
#
# 保存模型
from joblib import dump, load
dump(cls, 'models/svc.joblib')
# dump(neigh, './models/neigh.joblib')


# 加载训练好的模型
cls = load('models/svc.joblib')
predictLabels = cls.predict(x_test)
print("svm acc:%s" % accuracy_score(y_test, predictLabels))


# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictLabels)
print(cm)


import seaborn as sn
import pandas as pd

df_cm = pd.DataFrame(cm, index=[i for i in ['Zhuan', 'Li', 'Cao', 'Xing', 'Kai']],
                  columns=[i for i in ['Zhuan', 'Li', 'Cao', 'Xing', 'Kai']])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap="Greens", fmt="d")
plt.imshow(cm)
plt.show()