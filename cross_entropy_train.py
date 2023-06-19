import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import random
import pickle
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet_v2
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet_v3
from tensorflow.keras.applications import ResNet50, InceptionResNetV2, InceptionV3, MobileNetV2, VGG16, VGG19, MobileNetV3Large

BASE_MODEL_TYPE = 'ResNet50'

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def get_preprocess_function(base_model_type):
    if base_model_type == 'ResNet50':
        return preprocess_resnet50
    elif base_model_type == 'InceptionResNetV2':
        return preprocess_inception_resnet_v2
    elif base_model_type == 'InceptionV3':
        return preprocess_inception_v3
    elif base_model_type == 'MobileNetV2':
        return preprocess_mobilenet_v2
    elif base_model_type == 'MobileNetV3':
        return preprocess_mobilenet_v3
    elif base_model_type == 'VGG16':
        return preprocess_vgg16
    elif base_model_type == 'VGG19':
        return preprocess_vgg19
    elif base_model_type == 'ResNet50V2':
        return tf.keras.applications.mobilenet.preprocess_input
    elif base_model_type == 'EfficientNet':
        return tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(
            'Invalid base_model_type.')


preprocess_input = get_preprocess_function(BASE_MODEL_TYPE)

# Setting seed to ensure reproducibility.
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

BASE_PATH = 'data'

# Image Data Generator for training and validation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   validation_split=0.05,
                                   rotation_range=20,
                                   brightness_range=[0.5, 1.5],
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    BASE_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    BASE_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model architecture

if BASE_MODEL_TYPE == 'ResNet50':
    base_cnn = ResNet50(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'InceptionResNetV2':
    base_cnn = InceptionResNetV2(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'InceptionV3':
    base_cnn = InceptionV3(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'MobileNetV2':
    base_cnn = MobileNetV2(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'MobileNetV3':
    base_cnn = MobileNetV3Large(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'VGG16':
    base_cnn = VGG16(weights="imagenet", input_shape=(224, 224, 3),
                     include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'VGG19':
    base_cnn = VGG19(weights="imagenet", input_shape=(224, 224, 3),
                     include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'ResNet50V2':
    base_cnn = tf.keras.applications.resnet_v2.ResNet50V2(weights="imagenet", input_shape=(224, 224, 3),
                                                          include_top=False, pooling='avg')
elif BASE_MODEL_TYPE == 'EfficientNet':
    base_cnn = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet", input_shape=(224, 224, 3),
                                                                 include_top=False, pooling='avg')
else:
    raise ValueError(
        'Invalid base_model_type')

for layer in base_cnn.layers:
    layer.trainable = False

x = base_cnn.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_cnn.input, outputs=predictions)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)

# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x=train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=[early_stopping_callback]
)

with open("cross_entropy_history/" + str(BASE_MODEL_TYPE) + ".pkl", "wb") as f:
    pickle.dump(history.history, f)

# Save the model
model.save("cross_entropy_history/" + str(BASE_MODEL_TYPE) + ".h5")
