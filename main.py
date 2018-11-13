import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
from export_module import export_model_for_mobile, simple_save
import os
import sys
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from PIL import Image
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
from keras.optimizers import Adam

np.random.seed(42)


def build_model():
    model = Sequential()

    # relu - функция активации (softplus - более смягченная)
    model.add(Dense(800, input_dim=784, kernel_initializer="normal", activation="relu"))
    # softmax - функция активации: сумма всех выходных нейронов равна 1
    model.add(Dense(10, kernel_initializer="normal", activation="softmax"))

    # SGD - метод обучения: градиентный спуск
    # categorical_crossentropy - мера ошибки
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
    return model


def build_model2(x_train):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model


def save_to_json(model):
    model_json = model.to_json()
    json_file = open("mnist_model.json", "w")
    json_file.write(model_json)
    json_file.close()


def fit_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    # print(x_train.shape)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np_utils.to_categorical(y_train, 10)

    model = build_model2(x_train)

    # print(model.inputs)
    # print(model.outputs)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('./model.h5', save_best_only=True, verbose=1)
    ]
    model.fit(x_train, y_train, batch_size=200, epochs=7, validation_split=0.2, verbose=1, callbacks=callbacks)

    # оценка качества работы сети
    evalute_model(model)

    print(model.summary())

    # save_to_json(model)

    # simple_save('./PlanetModel/1', './model.h5')
    # export_model_for_mobile('mnist_nn', "dense_1_input", "dense_2/Softmax")


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def evalute_model(model):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # оценка качества работы сети
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = np_utils.to_categorical(y_test, 10)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы тестовых данных: %.2f%%" % (score[1] * 100))


def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            # img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28, 28, 1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data


def array_from_image(file_path):
    img = Image.open(file_path).convert("L")
    img = np.resize(img, (28, 28))
    im2arr = np.array(img)
    return im2arr


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    path = get_script_path()
    number_image_path = os.path.join(path, "data", "1.png")
    # print(number_image_path)
    img = image.load_img(path=number_image_path, color_mode="grayscale", target_size=(28, 28, 1))
    img = image.img_to_array(img) / 255.
    # print(img)
    # img = array_from_image(number_image_path)
    test_img = img.reshape(1, 28, 28, 1)
    # model = build_model2(x_train)
    # model.load_weights('./model.h5')
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model = load_model('./model.h5')
    #evalute_model(model)

    print(test_img.shape)
    img_class = model.predict_classes(test_img, verbose=1)
    print(img_class)
    predictions = img_class[0]
    classname = img_class[0]
    print("Class: ", classname)

    img = img.reshape((28, 28))
    plt.imshow(img)
    plt.title(classname)
    plt.show()


if __name__ == '__main__':
    # fit_model()
    main()