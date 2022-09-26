# 步骤
# 1、图片数据预处理
# 2、加载模型
# 3、训练模型
# 4、预测图片
# 5、评估测试数据集
# 6、保存模型
# 7、调用加载模型

import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import glob

img_path = './yalefaces/train/subject01.glasses.gif'

# cap = cv2.VideoCapture(img_path)
# ret, img = cap.read()
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 图片预处理
# img_list:numpy格式图片
# label_list：numpy格式的label
# cls.train(img_list,np.array(label_list))

# 为了减少运算，提高速度，将人脸区域用人脸检测器提取出来
# 构造hog人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()

def getFaceImgLabel(fileName):
    # 读取图片
    cap = cv2.VideoCapture(fileName)
    ret, img = cap.read()
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = hog_face_detector(img, 1)
    # 判断是否有人脸
    if len(detections) > 0:
        # 获取人脸区域坐标
        x = detections[0].left()
        y = detections[0].top()
        r = detections[0].right()
        b = detections[0].bottom()
        # 截取人脸
        img_crop = img[y:b, x:r]
        # 缩放解决冲突
        img_crop = cv2.resize(img_crop, (120, 120))
        # 获取人脸labelid
        label_id = int(fileName.split('/')[-1].split('.')[0].split('subject')[-1])
        # 返回值
        return img_crop, label_id
    else:
        return None, -1


# 测试一张图片
# img, label = getFaceImgLabel(img_path)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
# print(label)

file_list = glob.glob('./yalefaces/train/*')
# 构造两个空列表
img_list = []
label_list = []

for train_file in file_list:
    # 获取每一张图片的对应信息
    img, label = getFaceImgLabel(train_file)

    # 过滤数据
    if label != -1:
        img_list.append(img)
        label_list.append(label)

# print(len(img_list))
# print(len(label_list))

# 构造分类器
face_cls = cv2.face.LBPHFaceRecognizer_create()
# cv2.face.EigenFaceRecognizer_create()
# cv2.face.FisherFaceRecognizer_create()

# 训练
face_cls.train(img_list, np.array(label_list))

"""预测一张图片"""
# test_file = './yalefaces/test/subject03.glasses.gif'
#
# img, label = getFaceImgLabel(test_file)
# 过滤数据
# if label != -1:
#     predict_id, distance = face_cls.predict(img)
#     print(predict_id)

# 评估模型
file_list = glob.glob('./yalefaces/test/*')

true_list = []
predict_list = []

for test_file in file_list:
    # 获取每一张图片的对应信息
    img, label = getFaceImgLabel(test_file)
    # 过滤数据
    if label != -1:
        predict_id, distance = face_cls.predict(img)
        predict_list.append(predict_id)
        true_list.append(label)


# 查看准确率
from sklearn.metrics import accuracy_score

accuracy_score(true_list, predict_list)
print(accuracy_score(true_list, predict_list))
# 获取融合矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_list, predict_list)
plt.imshow(cm)
plt.show()

# 可视化
# import seaborn
#
# seaborn.heatmap(cm, annot=True)
#
# 保存模型
face_cls.save('./weights/LBPH.yml')
# 调用模型
new_cls = cv2.face.LBPHFaceRecognizer_create()
new_cls.read('./weights/LBPH.yml')
# 预测一张图片
test_file = './yalefaces/test/subject03.glasses.gif'

img, label = getFaceImgLabel(test_file)
# 过滤数据
if label != -1:
    predict_id, distance = new_cls.predict(img)
    print(predict_id)


