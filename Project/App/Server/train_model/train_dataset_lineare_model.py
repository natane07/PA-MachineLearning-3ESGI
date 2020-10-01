import numpy as np
from ctypes import *
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

print("Chargement de la DLL et definition des signatures des méthodes")
my_dll_path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Lib/MachineLearning/target/debug/machine_learning_c.dll"
my_lib = CDLL(my_dll_path)
# Création d'un model
my_lib.create_linear_model.restype = c_void_p
my_lib.create_linear_model.argtypes = [c_int]
# Prédiction d'un model pour la classification
my_lib.predict_linear_classification.argtypes = [
    c_void_p,
    POINTER(c_double),
    c_int
]
my_lib.predict_linear_classification.restype = c_double
# Entrainement du model pour la classification
my_lib.train_linear_model_classification_python.argtypes = [
    c_void_p,
    POINTER(c_double),
    POINTER(c_double),
    c_int,
    c_int,
    c_int,
    c_double
]
my_lib.train_linear_model_classification_python.restype = None
my_lib.serialized_lineare_modele.argtypes = [
    c_void_p,
    c_int
]
my_lib.serialized_lineare_modele.restype = None

my_lib.deserialized_lineare_model.restype = c_void_p

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

image_bas, y_bas = load_image_dataset(path_chaussure, 1.0)
image_chaussure, y_chaussure = load_image_dataset(path_haut, -1.0)
x_train = np.concatenate((image_bas, image_chaussure))
y_train = np.concatenate((y_bas, y_chaussure))

flattened_X = x_train.flatten()
flattened_Y = y_train.flatten()
# print(x_train)
print(x_train.shape)
# print(flattened_X)
# print(flattened_Y)

#Création du model
model = my_lib.create_linear_model(c_int(x_train.shape[1]))

#Entrainement du model
my_lib.train_linear_model_classification_python(
    model,
    flattened_X.ctypes.data_as(POINTER(c_double)),
    y_train.ctypes.data_as(POINTER(c_double)),
    x_train.shape[0],
    x_train.shape[1],
    100000,
    0.001
)

# Prediction du model
accuracy = 0
print("After Training")
for i, inputs_k in enumerate(x_train):
    predict_value = my_lib.predict_linear_classification(model, inputs_k.ctypes.data_as(POINTER(c_double)), len(inputs_k))
    if predict_value == y_train[i]:
        accuracy = accuracy + 1
print(accuracy)

# Serialisation (Fonctionnelle)
my_lib.serialized_lineare_modele(model, c_int(x_train.shape[1]))

#Deserialisation
# model_deser = my_lib.deserialized_lineare_model()
print("FIN")