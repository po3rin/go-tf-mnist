import numpy as np
import keras as keras
import tensorflow as tf
from PIL import Image

model = keras.models.load_model('mnist.h5')
model.summary()

model.compile(
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

img = Image.open('./../testdata/img_10.jpg').convert("L")
img = np.resize(img, (28, 28, 1))
im2arr = np.array(img)
im2arr = im2arr.reshape(1, 28, 28, 1)

print(im2arr)

y_pred = model.predict_classes(im2arr)
print(y_pred)
