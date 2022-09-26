# 加载包
import tensorflow as tf
# 构造转换器
converter = tf.lite.TFLiteConverter.from_saved_model('./data/face_mask_model/')
# 转换
tflite_model = converter.convert()
# 保存lite
with open('./data/face_mask.tflite', 'wb') as f:
    f.write(tflite_model)