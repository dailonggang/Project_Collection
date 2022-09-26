# 导入相关包
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 加载模型
model = tf.keras.models.load_model('./data/face_mask_model/')
model.summary()

# 挑选测试图片
img = cv2.imread('./images/2.no/0_0_baibaihe_0093.jpg')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 加载SSD模型
face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt',
                                         'weights/res10_300x300_ssd_iter_140000.caffemodel')


# 人脸检测函数
def face_detect(img):
    # 转为Blob
    img_blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123), swapRB=True)
    # 输入
    face_detector.setInput(img_blob)
    # 推理
    detections = face_detector.forward()
    # 获取原图尺寸
    img_h, img_w = img.shape[:2]

    # 人脸数量
    person_count = detections.shape[3]

    for face_index in range(person_count):
        # 置信度
        confidence = detections[0, 0, face_index, 2]
        if confidence > 0.5:
            locations = detections[0, 0, face_index, 3:7] * np.array([img_w, img_h, img_w, img_h])
            # 取证
            l, t, r, b = locations.astype('int')
            # cv2.rectangle(img,(l,t),(r,b),(0,255,0),5)
            return img[t:b, l:r]
    return None

# 转为Blob的函数
def imgBlob(img):
    # 转为Blob
    img_blob = cv2.dnn.blobFromImage(img,1,(100,100),(104,177,123),swapRB=True)
    # 压缩维度
    img_squeeze = np.squeeze(img_blob).T
    # 旋转
    img_rotate = cv2.rotate(img_squeeze,cv2.ROTATE_90_CLOCKWISE)
    # 镜像
    img_flip =  cv2.flip(img_rotate,1)
    # 去除负数，并归一化
    img_blob = np.maximum(img_flip,0) / img_flip.max()
    return img_blob

# 裁剪人脸
img_crop = face_detect(img)

plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
plt.show()

# 转为Blob
img_blob = imgBlob(img_crop)
plt.imshow(img_blob)

img_input = img_blob.reshape(1, 100, 100, 3)
result = model.predict(img_input)

from scipy.special import softmax
result = softmax(result[0])
max_index = result.argmax()
max_value = result[max_index]
import os
labels = os.listdir('./images/')
print(labels[max_index])
