# 导入相关包
import cv2
# 安装DLIB
import dlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
# 读取照片
img = cv2.imread('./images/4.jpg')


# 构造CNN人脸检测器
cnn_face_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
# 检测人脸
detections = cnn_face_detector(img, 1)
# 解析矩形结果
for face in detections:
    x = face.rect.left()
    y = face.rect.top()
    r = face.rect.right()
    b = face.rect.bottom()
    # 置信度
    c = face.confidence
    print(c)

    cv2.rectangle(img, (x, y), (r, b), (255, 0, 255), 3)
# 显示照片
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

