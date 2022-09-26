"""
视频流检测人脸：
1、构造MTCNN人脸检测器
2、获取视频流
3、检测每一帧画面
4、画人脸框并显示
"""

# 导入包
import cv2
# 导入MTCNN
from mtcnn.mtcnn import MTCNN

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 镜像
    frame = cv2.flip(frame, 1)

    # MTCNN需要RGB通道顺序
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 加载模型
    face_detetor = MTCNN()
    # 检测人脸
    detections = face_detetor.detect_faces(frame)
    for face in detections:
        (x, y, w, h) = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 显示画面
    cv2.imshow('Demo', frame)

    # 退出条件
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()