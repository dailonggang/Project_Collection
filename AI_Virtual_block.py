"""
手势控制图形
1、opencv 读取视频流
2、在视频图像上画一个图形块
3、通过mediapipe库获取手指关节坐标
4、判断手指是否在方块上
5、在方块上通过两指拖拽图形移动
6、完善：通过食指和中指指尖距离确定是否激活移动
7、完善：画面显示FPS等信息
"""

import cv2
import math
import time
# 导入mediapipe：https://google.github.io/mediapipe/solutions/hands
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 读取视频流
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器

fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数

# 获取画面宽度、高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter('hands.mp4', fourcc, fps, (width, height))  # 写入视频

# 方块初始数组
x, y, w, h = 100, 100, 200, 200

L1, L2 = 0, 0

on_block = False
block_color = (0, 255, 0)

while True:
    fps = time.time()
    ret, frame = cap.read()

    # 镜像
    frame = cv2.flip(frame, 1)

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 识别
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 如果有结果
    if results.multi_hand_landmarks:

        # 遍历双手
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制21个关键点
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # 至于hand_landmarks官方也没有给出，所以打印看看，确实对应21个关键点坐标
            # print(hand_landmarks)
            # exit()

            # 保存21个关键点的x,y坐标列表
            x_list = []
            y_list = []
            for landmark in hand_landmarks.landmark:
                # 添加x坐标
                x_list.append(landmark.x)
                # 添加y坐标
                y_list.append(landmark.y)

            # 输出一下长度
            # print(len(x_list))

            # 获取食指指尖坐标，这里需要重新与画面的宽度与高度相乘，因为官方上解释x,y已经除过画面的宽高
            index_finger_x = int(x_list[8] * width)
            index_finger_y = int(y_list[8] * height)
            # 获取画面像素坐标
            # print(index_finger_x, index_finger_y)

            # 获取中指坐标
            middle_finger_x = int(x_list[12] * width)
            middle_finger_y = int(y_list[12] * height)

            # 计算两指距离
            # finger_distance =math.sqrt( (middle_finger_x - index_finger_x)**2 + (middle_finger_y-index_finger_y)**2)
            finger_distance = math.hypot((middle_finger_x - index_finger_x), (middle_finger_y - index_finger_y))

            # 看一下距离
            # print(finger_distance)

            # 把食指和中指指尖画出来
            cv2.circle(frame, (index_finger_x, index_finger_y), 20, (0, 255, 255), -1)
            cv2.circle(frame, (middle_finger_x, middle_finger_y), 20, (0, 255, 255), -1)

            # 判断食指指尖在不在方块上
            if finger_distance < 30:
                # X坐标范围 Y坐标范围
                if (index_finger_x > x and index_finger_x < (x + w)) and (
                        index_finger_y > y and index_finger_y < (y + h)):

                    if on_block == False:
                        L1 = index_finger_x - x
                        L2 = index_finger_y - y
                        block_color = (255, 0, 255)
                        on_block = True
                else:
                    continue

            else:
                # 解除
                on_block = False
                block_color = (0, 255, 0)

            # 更新坐标
            if on_block == True:
                x = index_finger_x - L1
                y = index_finger_y - L2

    # 画一个正方形，需要实心
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),-1)

    # 半透明处理
    overlay = frame.copy()
    cv2.rectangle(frame, (x, y), (x + w, y + h), block_color, -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)

    # 显示刷新率FPS
    now = time.time()
    fps_text = 1 / (now - fps)
    fps = now
    FPS = cv2.putText(frame, "FPS: " + str(int(fps_text)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # 保存
    output.write(frame)
    # 显示画面
    cv2.imshow('demo', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



