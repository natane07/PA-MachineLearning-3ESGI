import numpy as np
from ctypes import *
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

print("Chargement de la DLL et definition des signatures des méthodes")
my_dll_path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Lib/MachineLearning/target/debug/machine_learning_c.dll"
my_lib = CDLL(my_dll_path)
# Creation du modele PMC
# Creation du modele PMC
my_lib.create_mlp.argtypes = [
    POINTER(c_int64),
    c_int
]
my_lib.create_mlp.restype = c_void_p
# Prediction
my_lib.mlp_classification.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_classification.restype = c_void_p
my_lib.mlp_classification_image.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_classification_image.restype = c_int
my_lib.mlp_classification_max_value.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_classification_max_value.restype = c_double

my_lib.mlp_regression.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_regression.restype = c_void_p
my_lib.mlp_regression_max_value.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_regression_max_value.restype = c_double
# Entrainement du PMC
my_lib.mlp_train_classification.argtypes = [c_void_p,
                                            c_int,
                                            POINTER(c_double),
                                            c_int,
                                            POINTER(c_double),
                                            c_int,
                                            c_int,
                                            c_double]
my_lib.mlp_train_classification.restype = None
my_lib.mlp_train_regression.argtypes = [c_void_p,
                                            c_int,
                                            POINTER(c_double),
                                            c_int,
                                            POINTER(c_double),
                                            c_int,
                                            c_int,
                                            c_double]
my_lib.mlp_train_regression.restype = None
my_lib.serialized_mlp.argtypes = [
    c_void_p
]
my_lib.serialized_mlp.restype = None

my_lib.deserialized_mlp.restype = c_void_p

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

path_bas = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Bas/image_scrap/image_resize/"
path_chaussure= "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_scrap/image_resize/"
path_haut = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Haut/image_scrap/image_resize/"

image_bas, y_bas = load_image_dataset(path_bas, [1.0, -1.0, -1.0])
image_chaussure, y_chaussure = load_image_dataset(path_chaussure, [-1.0, 1.0, -1.0])
image_haut, y_haut = load_image_dataset(path_haut, [-1.0, -1.0, 1.0])

x_train = np.concatenate((image_bas, image_chaussure, image_haut))
y_train = np.concatenate((y_bas, y_chaussure, y_haut))

npl = list([2700, 20, 20, 20, 3])
npl_pointer = (c_int64 * 5)(*npl)
mlp = my_lib.create_mlp(npl_pointer, len(npl))

flattened_X = x_train.flatten()
flattened_Y = y_train.flatten()
x_train_pointer = (c_double * len(flattened_X))(*flattened_X)
y_train_pointer = (c_double * len(flattened_Y))(*flattened_Y)

# Entrainement du model MLP
my_lib.mlp_train_classification(mlp, len(x_train), x_train_pointer, len(flattened_X), y_train_pointer,
                                len(flattened_Y), 10000, 0.001)

# Predicte
good_reponse = 0
for i in range(len(x_train)):
    x_pointer = (c_double * len(x_train[i]))(*x_train[i])
    test = my_lib.mlp_classification_image(mlp, x_pointer, len(x_train[i]))
    print(y_train[i])
    if y_train[i][test] == 1:
        good_reponse = good_reponse + 1
    print(test)

print(good_reponse)
my_lib.serialized_mlp(mlp)
print("finish")