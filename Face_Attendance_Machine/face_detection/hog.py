# 导入相关包
import cv2
import matplotlib.pyplot as plt
import dlib
plt.rcParams['figure.dpi'] = 200

# 读取照片
img = cv2.imread('./images/4.jpg')

# 构造HOG人脸检测器
hog_face_detetor = dlib.get_frontal_face_detector()
# 检测人脸
# scale 类似haar的scaleFactor
detections = hog_face_detetor(img, 1)
print(detections)

# 解析矩形结果
for face in detections:
    x = face.left()
    y = face.top()
    r = face.right()
    b = face.bottom()
    cv2.rectangle(img, (x, y), (r, b), (255, 0, 255), 3)
# 显示照片
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

