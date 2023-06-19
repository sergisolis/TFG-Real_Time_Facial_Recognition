import time
import os
import cv2
import random
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet_v2
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet_v3
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionResNetV2, InceptionV3, MobileNetV2, VGG16, VGG19, MobileNetV3Large
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.keras import metrics
import pickle


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


# Usage in generate_batch_dataset
preprocess_input = get_preprocess_function(BASE_MODEL_TYPE)


def create_generator_dataset(list_dirs, maxfiles=20):
    list_path_images = []
    for dir in list_dirs:
        new_path = os.path.join(path, dir)
        images = os.listdir(new_path)[:maxfiles]
        num_images = len(images)
        if num_images >= 2:
            for i in range(num_images - 1):
                for j in range(i + 1, num_images):

                    anchor = os.path.join(new_path, images[i])
                    positive = os.path.join(new_path, images[j])

                    count = 0
                    while count < 1:
                        negative_dir = dir
                        while negative_dir == dir:
                            negative_dir = random.choice(list_dirs)

                        negative_images = os.listdir(f'{path}/{negative_dir}')
                        negative_image = random.choice(negative_images)
                        negative_dir = os.path.join(path, negative_dir)
                        negative = os.path.join(negative_dir, negative_image)
                        count += 1

                        list_path_images.append([positive, anchor, negative])

    random.shuffle(list_path_images)
    return list_path_images


path = 'data'
list_dirs = os.listdir(path)

triplet_dataset = create_generator_dataset(list_dirs)


def split_data(list_file, ratio):
    if sum(ratio) != 1:
        print('Total ratio must equal 1')
        return
    else:
        train_size = int(len(list_file) * ratio[0])
        test_size = int(len(list_file) * ratio[1])

    return list_file[:train_size], list_file[train_size:(train_size + test_size)]


train, val = split_data(triplet_dataset, [0.95, 0.05])

print(len(train))
print(len(val))


def process_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img


def augment_image(image):
    # Decrease brightness by a factor between 0.7 and 1
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    random_bright = 0.7 + 0.3 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * random_bright
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # cap values at 255
    hsv = np.array(hsv, dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Colorshift
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.int16(image)
    image[:, :, 0] = cv2.add(image[:, :, 0], np.random.randint(0, 75)) % 256
    image[:, :, 1] = cv2.add(image[:, :, 1], np.random.randint(0, 75)) % 256
    image[:, :, 2] = cv2.add(image[:, :, 2], np.random.randint(0, 75)) % 256
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply random rotation within a range of 20 degrees
    angle = np.random.randint(-20, 20)
    M = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return image


def generate_batch_dataset(list_files, batch_size=64, preprocess=True, augment=False):
    num_batch = len(list_files) // batch_size

    for i in range(num_batch + 1):
        anchor = []
        positive = []
        negative = []
        j = i*batch_size

        while j < (i+1) * batch_size and j < len(list_files):
            a, p, n = list_files[j]

            anchor_image = process_image(a)
            positive_image = process_image(p)
            negative_image = process_image(n)

            # Add original images to the list
            anchor.append(anchor_image)
            positive.append(positive_image)
            negative.append(negative_image)

            # If augment is True, add augmented images to the list as well
            if augment:
                for _ in range(4):  # Create 4 augmented versions
                    anchor_image_aug = augment_image(anchor_image)
                    positive_image_aug = augment_image(positive_image)
                    negative_image_aug = augment_image(negative_image)

                    anchor.append(anchor_image_aug)
                    positive.append(positive_image_aug)
                    negative.append(negative_image_aug)

            j += 1

        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([positive, anchor, negative])


train_generator = generate_batch_dataset(train, batch_size=64)
val_generator = generate_batch_dataset(val, batch_size=64)


class Distance(tf.keras.layers.Layer):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def call(self, anchor, positive, negative):
        d_pos = tf.reduce_sum(tf.square(positive - anchor), -1)
        d_neg = tf.reduce_sum(tf.square(negative - anchor), -1)
        return (d_pos, d_neg)


def make_layers_not_trainable(base_cnn):
    for layer in base_cnn.layers:

        layer.trainable = False


def encoder(input_shape, base_model_type='ResNet50'):
    # Choose the base model
    if base_model_type == 'ResNet50':
        base_cnn = ResNet50(
            weights="imagenet", input_shape=input_shape, include_top=False, pooling='avg')
    elif base_model_type == 'InceptionResNetV2':
        base_cnn = InceptionResNetV2(
            weights="imagenet", input_shape=input_shape, include_top=False, pooling='avg')
    elif base_model_type == 'InceptionV3':
        base_cnn = InceptionV3(
            weights="imagenet", input_shape=input_shape, include_top=False, pooling='avg')
    elif base_model_type == 'MobileNetV2':
        base_cnn = MobileNetV2(
            weights="imagenet", input_shape=input_shape, include_top=False, pooling='avg')
    elif base_model_type == 'MobileNetV3':
        base_cnn = MobileNetV3Large(
            weights="imagenet", input_shape=input_shape, include_top=False, pooling='avg')
    elif base_model_type == 'VGG16':
        base_cnn = VGG16(weights="imagenet", input_shape=input_shape,
                         include_top=False, pooling='avg')
    elif base_model_type == 'VGG19':
        base_cnn = VGG19(weights="imagenet", input_shape=input_shape,
                         include_top=False, pooling='avg')
    elif base_model_type == 'ResNet50V2':
        base_cnn = tf.keras.applications.resnet_v2.ResNet50V2(weights="imagenet", input_shape=input_shape,
                                                              include_top=False, pooling='avg')
    elif base_model_type == 'EfficientNet':
        base_cnn = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet", input_shape=input_shape,
                                                                     include_top=False, pooling='avg')
    else:
        raise ValueError(
            'Invalid base_model_type')

    make_layers_not_trainable(base_cnn)

    encode_model = Sequential([
        base_cnn,
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

    return encode_model


def final_model(input_shape=(224, 224, 3)):

    encode = encoder(input_shape, BASE_MODEL_TYPE)

    input_a = Input(input_shape, name='input_anchor')
    input_p = Input(input_shape, name='input_positive')
    input_n = Input(input_shape, name='input_negative')

    feature_a = encode(input_a)
    feature_p = encode(input_p)
    feature_n = encode(input_n)

    distances = Distance()(
        feature_a,
        feature_p,
        feature_n
    )
    model = Model(inputs=[input_a, input_p, input_n], outputs=distances)
    return model


network = final_model()


class FaceNetModel(Model):
    def __init__(self, network, margin=1.0):
        super(FaceNetModel, self).__init__()

        self.margin = margin
        self.network = network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.network(inputs)

    def train_step(self, data):
        # Get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @ property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]


FaceNet_model = FaceNetModel(network)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-01)
FaceNet_model.compile(optimizer=optimizer)


def test_on_triplets(batch_size=64):
    pos_scores, neg_scores = [], []
    test_loss = []

    for data in generate_batch_dataset(val, batch_size=batch_size):
        # , augment=False
        # Calculate test loss
        loss = FaceNet_model.test_on_batch(data)
        test_loss.append(loss)

        # Calculate accuracy
        prediction = FaceNet_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    test_loss = sum(test_loss) / len(test_loss)

    accuracy = np.sum(np.array(pos_scores) < np.array(
        neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Test Loss = {test_loss:.5f}")
    print(f"Accuracy on test = {accuracy:.5f}")
    return(test_loss, accuracy, ap_mean, an_mean, ap_stds, an_stds)


epochs = 50
batch_size = 64

train_loss = []
test_loss = []

min_val_loss = float('inf')
patience = 15
no_improve_epochs = 0

for epoch in range(1, epochs+1):
    t = time.time()

    # Training the model on train data
    epoch_loss = []
    for data in generate_batch_dataset(train, batch_size=batch_size):
        loss = FaceNet_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    train_loss.append(epoch_loss)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time()-t)} sec)")
    print(f"Loss on train    = {epoch_loss:.10f}")

    # Testing the model on test data
    metric = test_on_triplets(batch_size=batch_size)
    test_loss.append(metric[0])
    accuracy = metric[1]

    curr_val_loss = metric[0]

    # Check for improvement in validation loss
    if curr_val_loss < min_val_loss:
        min_val_loss = curr_val_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # If no improvement, stop training
    if no_improve_epochs >= patience:
        print('Early stopping: no improvement in validation loss for {} epochs'.format(
            patience))
        break


def extract_encoder(model):
    ec = encoder((224, 224, 3), BASE_MODEL_TYPE)
    i = 0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        ec.layers[i].set_weights(layer_weight)
        i += 1
    return ec


encode = extract_encoder(FaceNet_model)
encode.save("triplet_loss_trained_models/" + str(BASE_MODEL_TYPE))
encode.summary()

history = {"train_loss": train_loss, "test_loss": test_loss}
with open("triplet_loss_history/" + str(BASE_MODEL_TYPE) + ".pkl", "wb") as f:
    pickle.dump(history, f)
