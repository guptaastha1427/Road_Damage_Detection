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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TF Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

tf.random.set_seed(42)
np.random.seed(42)


# TF Version: 2.20.0
# GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
# # Based on your confirmed folder structure

DATA_DIR = "/kaggle/input/datasets/atulyakumar98/pothole-detection-dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 7

for root, dirs, files in os.walk(DATA_DIR):
    print(root, "->", dirs, f"({len(files)} files)")


# /kaggle/input/datasets/atulyakumar98/pothole-detection-dataset -> ['normal', 'potholes'] (0 files)
# /kaggle/input/datasets/atulyakumar98/pothole-detection-dataset/normal -> [] (352 files)
# /kaggle/input/datasets/atulyakumar98/pothole-detection-dataset/potholes -> [] (329 files)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

class_labels = list(train_gen.class_indices.keys())
print("Classes:", class_labels)
print("Training samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)



# Found 546 images belonging to 2 classes.
# Found 135 images belonging to 2 classes.
# Classes: ['normal', 'potholes']
# Training samples: 546
# Validation samples: 135



from tensorflow.keras.applications import MobileNetV2

def build_transfer_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_transfer_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()



# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1784141312.963100      58 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13756 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
# I0000 00:00:1784141312.966531      58 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 13756 MB memory:  -> device: 1, name: Tesla T4, pci bus id: 0000:00:05.0, compute capability: 7.5
# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
# 9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
# │ mobilenetv2_1.00_224            │ (None, 7, 7, 1280)     │     2,257,984 │
# │ (Functional)                    │                        │               │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ global_average_pooling2d        │ (None, 1280)           │             0 │
# │ (GlobalAveragePooling2D)        │                        │               │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense (Dense)                   │ (None, 128)            │       163,968 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dropout (Dropout)               │ (None, 128)            │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_1 (Dense)                 │ (None, 1)              │           129 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 2,422,081 (9.24 MB)
#  Trainable params: 164,097 (641.00 KB)
#  Non-trainable params: 2,257,984 (8.61 MB)



labels = train_gen.classes.astype(int)
classes, counts = np.unique(labels, return_counts=True)

total = len(labels)
n_classes = len(classes)
class_weight_dict = {int(c): total / (n_classes * count) for c, count in zip(classes, counts)}
print("Class weights:", class_weight_dict)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]



# Class weights: {0: np.float64(0.9680851063829787), 1: np.float64(1.0340909090909092)}


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)



fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['accuracy'], label='train_acc')
ax[0].plot(history.history['val_accuracy'], label='val_acc')
ax[0].set_title('Accuracy')
ax[0].legend()

ax[1].plot(history.history['loss'], label='train_loss')
ax[1].plot(history.history['val_loss'], label='val_loss')
ax[1].set_title('Loss')
ax[1].legend()

plt.show()




val_gen.reset()
preds = model.predict(val_gen)
pred_labels = (preds > 0.5).astype(int).flatten()
true_labels = val_gen.classes

print(classification_report(true_labels, pred_labels, target_names=class_labels))

cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# 5/5 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step 
#               precision    recall  f1-score   support

#       normal       0.97      0.96      0.96        70
#     potholes       0.95      0.97      0.96        65

#     accuracy                           0.96       135
#    macro avg       0.96      0.96      0.96       135
# weighted avg       0.96      0.96      0.96       135


from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def check_image(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]
    label = class_labels[1] if pred > 0.5 else class_labels[0]
    confidence = pred if pred > 0.5 else 1 - pred

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

    plt.imshow(img)
    plt.title(f"{label} ({confidence*100:.1f}% confidence)")
    plt.axis('off')
    plt.show()

    return label




check_image("/kaggle/input/datasets/atulyakumar98/pothole-detection-dataset/normal/119.jpg", model, class_labels)
2026-07-15 18:52:09.009628: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-07-15 18:52:09.143831: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
Prediction: normal
Confidence: 99.39%

'normal'
