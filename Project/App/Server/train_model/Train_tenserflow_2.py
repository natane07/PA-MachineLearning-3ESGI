from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_core.python import keras, layers

path_bas = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Bas/image_scrap/image_resize/"
path_chaussure= "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_scrap/image_resize/"
path_haut = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Haut/image_scrap/image_resize/"

def load_image_dataset(path, label):
    fichiers = [f for f in listdir(path) if isfile(join(path, f))]
    img = []
    y = []
    for i in fichiers:
        im = Image.open(path + (i))
        im_arr1 = np.array(im) / 255
        if im.width is 30 and im.height is 30 and len(im_arr1.shape) is 3:
            print(path + (i))
            im_arr1 = np.reshape(im_arr1, (30 * 30 * 3))
            img.append(im_arr1)
            y.append(label)
    # y = np.reshape(y, 3 * len(y))
    return np.array(img), y

def load_one_image_dataset(path):
    img = []
    im = Image.open(path)
    im_arr1 = np.array(im) / 255
    if im.width is 30 and im.height is 30 and len(im_arr1.shape) is 3:
        print(path)
        im_arr1 = np.reshape(im_arr1, (30 * 30 * 3))
        img.append(im_arr1)
    return np.array(img)


image_bas, y_bas = load_image_dataset(path_bas, 0)
image_chaussure, y_chaussure = load_image_dataset(path_chaussure, 1)
image_haut, y_haut = load_image_dataset(path_haut, 2)

x_train = np.concatenate((image_bas, image_chaussure, image_haut))
y_image = np.concatenate((y_bas, y_chaussure, y_haut))
y_train = tf.keras.utils.to_categorical(y_image, 3) * 2.0 - 1

model = keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(150, activation=keras.activations.tanh))
model.add(tf.keras.layers.Dense(150, activation=keras.activations.tanh))
model.add(tf.keras.layers.Dense(150, activation=keras.activations.tanh))
model.add(tf.keras.layers.Dense(3, activation=keras.activations.tanh))

model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=[keras.metrics.categorical_accuracy])

model.fit(x_train, y_train, epochs=100,
          callbacks=[keras.callbacks.TensorBoard(log_dir='logs')])

path_bas_one_image = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Bas/image_not_scrap/image_resize/b2.jpg"
test = load_one_image_dataset(path_bas_one_image)
print(model.predict(np.array([test[0]])))

path_chaussure_one_image = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_not_scrap/image_resize/c1.jpg"
test = load_one_image_dataset(path_chaussure_one_image)
print(model.predict(np.array([test[0]])))

path_haut_one_image = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Haut/image_not_scrap/image_resize/h2.jpg"
test = load_one_image_dataset(path_haut_one_image)
print(model.predict(np.array([test[0]])))
