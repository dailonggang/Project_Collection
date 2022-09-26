# 导入包
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

# 读取照片
img = cv2.imread('./images/4.jpg')
# MTCNN需要RGB通道顺序
img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

from mtcnn.mtcnn import MTCNN

# 加载模型
face_detetor = MTCNN()
# 检测人脸
detections = face_detetor.detect_faces(img_cvt)
for face in detections:
    (x, y, w, h) = face['box']
    cv2.rectangle(img_cvt, (x, y), (x + w, y + h), (0, 255, 0), 5)
plt.imshow(img_cvt)
plt.show()