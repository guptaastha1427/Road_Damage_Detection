# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!kaggle datasets download -d atulyakumar98/pothole-detection-dataset


import numpy as np 
import pandas as pd


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam


plt.imshow(cv2.imread("/kaggle/input/pothole-detection-dataset/potholes/1.jpg"))
# <matplotlib.image.AxesImage at 0x7af2cd6a3400>

plt.imshow(cv2.imread("/kaggle/input/pothole-detection-dataset/normal/2.jpg"))
# <matplotlib.image.AxesImage at 0x7af2c9da6500>


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True, 
    validation_split=0.2)

training_set = train_datagen.flow_from_directory('/kaggle/input/pothole-detection-dataset', 
    target_size = (64, 64),
    batch_size = 32, 
    class_mode = 'binary', 
    subset="training")
# output -> Found 546 images belonging to 2 classes.


validation_generator = train_datagen.flow_from_directory("/kaggle/input/pothole-detection-dataset",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation')
# output -> Found 135 images belonging to 2 classes.


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4))
# drop from .4 to .5

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# adam = Adam(learning_rate= 0.01) 
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# cnn = tf.keras.models.Sequential()
# cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=8, activation='relu', input_shape=[64, 64, 3]))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=4))
# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu'))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# cnn.add(tf.keras.layers.Flatten())

# cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
# cnn.add(tf.keras.layers.Dropout(rate=0.1))
# cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cnn.fit(x = training_set, validation_data = validation_generator, epochs = 100)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/pothole-detection-dataset/potholes/100.jpg', target_size = (64, 64))
plt.imshow(cv2.imread("../input/pothole-detection-dataset/potholes/100.jpg"))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'pothole'
else:
  prediction = 'normal'


print(prediction)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 748ms/step
pothole

 
