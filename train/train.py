# -*- coding: utf-8 -*-

import numpy as np
import keras as keras
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import os

print(tf.__version__)
image_list = []  # トレーニングimage格納用list
label_list = []  # トレーニングlabel格納用list

train_path = "data/trainingSet"
test_path = "data/testSet"

for label in os.listdir(train_path):
    dir = train_path + "/" + label  # 各ディレクトリ名がラベル名になっているのでそれをトレーニングラベルに利用する。
    for filename in os.listdir(dir):
        label_list.append(label)
        image_path = dir + "/" + filename
        image = np.array(Image.open(image_path).convert(
            "L").resize((28, 28)))  # 画像をグレースケールで28x28のサイズに変換
        image_list.append(image/255.)  # 255で割って正規化

image_list = np.array(image_list)  # リストをnumpy配列に変換する。
label_list = np.array(label_list)  # リストをnumpy配列に変換する。
label_list = keras.utils.np_utils.to_categorical(label_list)

(train_data, test_data, train_label, test_label) = train_test_split(
    image_list, label_list, test_size=0.3, random_state=111)
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

batch_size = 128
epochs = 5
kernel_size = (4, 4)
input_shape = train_data[0].shape

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=kernel_size,
                              input_shape=input_shape, activation="relu"))
model.add(keras.layers.Conv2D(
    filters=64, kernel_size=kernel_size, activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(
    filters=64, kernel_size=kernel_size, activation="relu"))
model.add(keras.layers.Conv2D(
    filters=64, kernel_size=kernel_size, activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

model.compile(
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.fit(train_data, train_label,
          batch_size=batch_size, epochs=epochs)  # 学習させる

scores = model.evaluate(test_data, test_label, verbose=1)

# モデル保存処理
# json_string = model.to_json()
# open('mnist.json', 'w').write(json_string)
# model.save_weights('mnist.h5')
model.save('mnist.h5')

# スコアを標準出力に出力
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# # Proto Buffer 形式で保存
input_keras_model = './conv_mnist.h5'
export_dir = './conv_mnist_pb'

old_session = tf.keras.backend.get_session()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.keras.backend.set_session(sess)
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
signature = tf.saved_model.predict_signature_def(inputs={t.name: t for t in model.inputs},
                                                 outputs={t.name: t for t in model.outputs})
builder.add_meta_graph_and_variables(sess,
                                     tags=[
                                         tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
builder.save(as_text=True)
sess.close()
tf.keras.backend.set_session(old_session)

print('input_node_names:')
for t in model.inputs:
    print(t.name)

print('output_node_names:')
for t in model.outputs:
    print(t.name)
