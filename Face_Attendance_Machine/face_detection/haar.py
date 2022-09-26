# 步骤
# 1、读取包含人脸的图片
# 2.使用haar模型识别人脸
# 3.将识别结果用矩形框画出来

# 导入相关包
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

# 读取图片
img = cv2.imread('./images/4.jpg')

# 构造haar检测器
face_detector = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
# 转为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 检测结果
detections = face_detector.detectMultiScale(img_gray, scaleFactor=1.03,
                                            minNeighbors=7, minSize=(10, 10), maxSize=(100, 100))
# print(detections)
 
# 解析检测结果
for (x, y, w, h) in detections:
    # print(w, h)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3)
# 显示绘制结果
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()