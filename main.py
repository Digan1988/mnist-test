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
import cv2
from scipy import ndimage

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
    model = load_model('./model.h5')

    path = get_script_path()
    number_image_path = os.path.join(path, "data", "1.png")

    img = image.load_img(path=number_image_path, color_mode="grayscale", target_size=(28, 28, 1))
    img = image.img_to_array(img) / 255.
    test_img = img.reshape(1, 28, 28, 1)

    img_class = model.predict_classes(test_img, verbose=1)

    classname = img_class[0]
    print("Class: ", classname)

    img = img.reshape((28, 28))
    plt.imshow(img)
    plt.title(classname)
    plt.show()


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    print(cy, cx)

    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def detect():
    model = load_model('./model.h5')

    path = get_script_path()
    number_image_path = os.path.join(path, "data", "color_complete.png")
    color_complete = cv2.imread(number_image_path)
    gray_complete = cv2.imread(number_image_path, 0)

    (thresh, gray_complete) = cv2.threshold(255 - gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # gray_complete = cv2.GaussianBlur(gray_complete, (3, 3), 0)
    cv2.imwrite(os.path.join(path, "data", "compl.png"), gray_complete)

    digit_image = -np.ones(gray_complete.shape)

    height, width = gray_complete.shape

    predSet_ret = []

    for cropped_width in range(100, 300, 20):
        for cropped_height in range(100, 300, 20):
            for shift_x in range(0, width - cropped_width, int(cropped_width/4)):
                for shift_y in range(0, height - cropped_height, int(cropped_height/4)):

                    # кусок изображения
                    gray = gray_complete[shift_y:shift_y + cropped_height, shift_x:shift_x + cropped_width]
                    # если в куске почти ничего нет, то следующий шаг
                    if np.count_nonzero(gray) <= 20:
                        continue

                    # если только кусок числа, то следующий шаг
                    if (np.sum(gray[0]) != 0) or (np.sum(gray[:, 0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:, -1]) != 0):
                        continue

                    # верхнее левое положение прямоугольника
                    top_left = np.array([shift_y, shift_x])
                    # нижнее правое положение прямоугольника
                    bottom_right = np.array([shift_y + cropped_height, shift_x + cropped_width])

                    # сужение рамок так, чтобы попало только число
                    while np.sum(gray[0]) == 0:
                        top_left[0] += 1
                        gray = gray[1:]

                    while np.sum(gray[:, 0]) == 0:
                        top_left[1] += 1
                        gray = np.delete(gray, 0, 1)

                    while np.sum(gray[-1]) == 0:
                        bottom_right[0] -= 1
                        gray = gray[:-1]

                    while np.sum(gray[:, -1]) == 0:
                        bottom_right[1] -= 1
                        gray = np.delete(gray, -1, 1)

                    actual_w_h = bottom_right - top_left
                    # есть ли число внутри прямоугольника
                    if np.count_nonzero(digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] + 1) > 0.2 * actual_w_h[0] * actual_w_h[1]:
                        continue

                    # предварительная обработка одного числа с сохранением соотношения сторон
                    rows, cols = gray.shape
                    compl_dif = abs(rows - cols)
                    half_Sm = int(compl_dif / 2)
                    half_Big = half_Sm if half_Sm * 2 == compl_dif else half_Sm + 1
                    if rows > cols:
                        gray = np.lib.pad(gray, ((0, 0), (half_Sm, half_Big)), 'constant')
                    else:
                        gray = np.lib.pad(gray, ((half_Sm, half_Big), (0, 0)), 'constant')

                    # размер куска изображения меняется на 20x20
                    gray = cv2.resize(gray, (20, 20))
                    # добавляются черные линии, чтобы размер получился 28x28
                    gray = np.lib.pad(gray, ((4, 4), (4, 4)), 'constant')

                    # сдвиг изображения с использованием центра масс
                    shiftx, shifty = getBestShift(gray)
                    shifted = shift(gray, shiftx, shifty)
                    gray = shifted

                    flatten = gray.flatten() / 255.0

                    test_img = flatten.reshape(1, 28, 28, 1)

                    img_class = model.predict_classes(test_img, verbose=1)

                    classname = img_class[0]
                    # print("Class: ", classname)

                    # cv2.imwrite(os.path.join(path, "data", "color_complete" + "_" + str(shift_x) + "_" + str(shift_y) + ".png"), gray)

                    # добавление зеленой рамки на изображении
                    cv2.rectangle(color_complete, tuple(top_left[::-1]), tuple(bottom_right[::-1]), color=(0, 255, 0), thickness=5)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(color_complete, str(classname), (top_left[1], bottom_right[0] + 50), font, fontScale=1.4, color=(0, 255, 0), thickness=4)

    # сохранение изображения с рамками на диск
    cv2.imwrite(os.path.join(path, "data", "color_complete_digitized_image.png"), color_complete)

if __name__ == '__main__':
    # fit_model()
    # main()
    detect()