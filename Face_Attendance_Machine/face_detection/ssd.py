# 导入包
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200
# 读取照片
img = cv2.imread('./images/4.jpg')
# deploy.prototxt.txt：https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
# res10_300x300_ssd_iter_140000.caffemodel：https://github.com/Shiva486/facial_recognition/blob/master/res10_300x300_ssd_iter_140000.caffemodel

# 加载模型
face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt',
                                         './weights/res10_300x300_ssd_iter_140000.caffemodel')

# 原图尺寸
img_height = img.shape[0]
img_width = img.shape[1]
# 缩放至模型输入尺寸
img_resize = cv2.resize(img, (500, 300))
# 图像转为blob
img_blob = cv2.dnn.blobFromImage(img_resize, 1.9, (500, 300), (104.0, 177.0, 123.0))
# 输入
face_detector.setInput(img_blob)
# 推理
detections = face_detector.forward()
# print(detections)
# 查看大小
# 查看检测人脸数量
num_of_detections = detections.shape[2]
# 原图复制，一会绘制用
img_copy = img.copy()

for index in range(num_of_detections):
    # 置信度
    detection_confidence = detections[0, 0, index, 2]
    # 挑选置信度
    if detection_confidence > 0.15:
        # 位置
        locations = detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
        # 打印
        print(detection_confidence * 100)

        lx, ly, rx, ry = locations.astype('int')
        # 绘制
        cv2.rectangle(img_copy, (lx, ly), (rx, ry), (0, 255, 0), 5)

# 展示
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.show()