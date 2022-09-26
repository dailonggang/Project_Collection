# 步骤
# 1、图片数据预处理
# 2、加载模型
# 3、提取图片的特征描述符
# 4、预测图片：找到欧氏距离最近的特征描述符
# 5、评估测试数据集


# 导入包
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
plt.rcParams['figure.dpi'] = 200

# 获取人脸的68个关键点

# 人脸检测模型
hog_face_detector = dlib.get_frontal_face_detector()
# 关键点 检测模型
shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
# 读取一张测试图片
img = cv2.imread('./images/faces2.jpg')
# 检测人脸
detections = hog_face_detector(img, 1)
for face in detections:
    # 人脸框坐标
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    # 获取68个关键点
    points = shape_detector(img, face)

    # 绘制关键点
    for point in points.parts():
        cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), 1)

    # 绘制矩形框
    cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 面部特征描述符

# resnet模型
face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')


# 提取单张图片的特征描述符,label
def getFaceFeatLabel(fileName):
    # 获取人脸labelid
    label_id = int(fileName.split('/')[-1].split('.')[0].split('subject')[-1])
    # 读取图片
    cap = cv2.VideoCapture(fileName)
    ret, img = cap.read()
    # 转为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 人脸检测
    detections = hog_face_detector(img, 1)
    face_descriptor = None

    for face in detections:
        # 获取关键点
        points = shape_detector(img, face)
        # 获取特征描述符
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(img, points)
        # 转为numpy 格式的数组
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = np.reshape(face_descriptor, (1, -1))

    return label_id, face_descriptor


# 测试一张图片
id1, fd1 = getFaceFeatLabel('./yalefaces/train/subject01.leftlight.gif')
# print(id1, fd1.shape)

# 对train文件夹进行处理
import glob

file_list = glob.glob('./yalefaces/train/*')
# 构造两个空列表
label_list = []
feature_list = None

name_list = {}
index = 0
for train_file in file_list:
    # 获取每一张图片的对应信息
    label, feat = getFaceFeatLabel(train_file)

    # 过滤数据
    if feat is not None:
        # 文件名列表
        name_list[index] = train_file

        # label列表
        label_list.append(label)

        if feature_list is None:
            feature_list = feat
        else:
            # 特征列表
            feature_list = np.concatenate((feature_list, feat), axis=0)
        index += 1

# 计算一个特征描述符与所有特征的距离(排除自己)
np.linalg.norm((feature_list[0]-feature_list[1:]), axis=1)
# 寻找最小值索引
np.argmin(np.linalg.norm((feature_list[0]-feature_list[1:]), axis=1))

# 评估测试数据集
file_list = glob.glob('./yalefaces/test/*')
# 构造两个空列表
predict_list = []
label_list = []
# 距离阈值
threshold = 0.5

for test_file in file_list:
    # 获取每一张图片的对应信息
    label, feat = getFaceFeatLabel(test_file)

    # 读取图片
    cap = cv2.VideoCapture(test_file)
    ret, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 过滤数据
    if feat is not None:
        # 计算距离
        distances = np.linalg.norm((feat - feature_list), axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        if min_distance < threshold:
            # 同一人
            predict_id = int(name_list[min_index].split('/')[-1].split('.')[0].split('subject')[-1])
        else:
            predict_id = -1

        predict_list.append(predict_id)
        label_list.append(label)

        cv2.putText(img, 'True:' + str(label), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(img, 'Pred:' + str(predict_id), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.putText(img, 'Dist:' + str(min_distance), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        # 显示
        plt.figure()
        plt.imshow(img)
        plt.show()

from sklearn.metrics import accuracy_score
acc = accuracy_score(label_list, predict_list)
print(acc)