import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
from export_module import export_model_for_mobile, simle_save
import os
import sys
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD

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


def build_model2():
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_shape=(784,)))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def fit_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print(x_train.shape)
    x_train = x_train.reshape(60000, 784)
    # print(x_train.shape)
    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = np_utils.to_categorical(y_train, 10)

    model = build_model()

    # print(model.inputs)
    # print(model.outputs)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('./model.h5', verbose=1)
    ]
    model.fit(x_train, y_train, batch_size=200, epochs=100, validation_split=0.2, verbose=1, callbacks=callbacks)

    # оценка качества работы сети
    x_test = x_test.reshape(10000, 784)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = np_utils.to_categorical(y_test, 10)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Точность работы тестовых данных: %.2f%%" % (score[1]*100))

    print(model.summary())

    # simle_save('./PlanetModel/1', './model.h5')
    export_model_for_mobile('mnist_nn', "dense_1_input", "dense_2/Softmax")


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def main():
    path = get_script_path()
    number_image_path = os.path.join(path, "data", "1.png")
    # print(number_image_path)
    img = image.load_img(path=number_image_path, grayscale=True, target_size=(28, 28, 1))
    img = image.img_to_array(img) / 255.
    # print(img)
    test_img = img.reshape((1, 784))
    model = build_model()
    model.load_weights('./model.h5')

    img_class = model.predict_classes(test_img, verbose=1)
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