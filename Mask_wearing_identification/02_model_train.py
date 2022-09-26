# 步骤
# 1.读取NPZ文件
# 2.onehot 独热编码
# 3.分为train和test数据
# 4.搭建CNN模型
# 5.训练模型
# 6.保存模型

# 1.读取NPZ文件
import numpy as np
arr = np.load('./data/imageData.npz')
img_list = arr['arr_0']
label_list = arr['arr_1']
np.unique(label_list)

from sklearn.preprocessing import OneHotEncoder

# 实例化
onehot = OneHotEncoder()
# 编码
y_onehot =onehot.fit_transform(label_list.reshape(-1,1))
y_onehot_arr = y_onehot.toarray()

# 3.分为train和test数据
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(img_list, y_onehot_arr, test_size=0.2, random_state=42)

# 4.搭建CNN模型
# pip install --upgrade tensorflow
# pip install tensorflow-gpu==版本号  # GPU
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

# 搭建模型

model = Sequential([
    layers.Conv2D(16, 3, padding='same', input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(166, activation='relu'),
    layers.Dense(22, activation='relu'),
    layers.Dense(3, activation='sigmoid')
])
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 预览模型
model.summary()

# 5.训练模型[22]
history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=30, epochs=15)


# 查看训练效果
import pandas as pd
import matplotlib.pyplot as plt

history_pd = pd.DataFrame(history.history)

# 查看损失
plt.plot(history_pd['loss'])
plt.plot(history_pd['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train set', 'test set'], loc='upper right')
plt.show()

# 查看准确率
plt.plot(history_pd['accuracy'])
plt.plot(history_pd['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train set', 'test set'], loc='upper right')
plt.show()

# 6.保存模型
model.save('./data/face_mask_model')