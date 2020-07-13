from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from ctypes import *

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
#MLP
my_lib.create_mlp.argtypes = [
    POINTER(c_int64),
    c_int
]
my_lib.create_mlp.restype = c_void_p
my_lib.serialized_mlp.argtypes = [
    c_void_p
]
my_lib.serialized_mlp.restype = None
my_lib.deserialized_mlp.restype = c_void_p
my_lib.mlp_classification.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_classification.restype = c_void_p
my_lib.mlp_classification_image.argtypes = [c_void_p, POINTER(c_double), c_int]
my_lib.mlp_classification_image.restype = c_int

my_lib.mlp_train_classification.argtypes = [c_void_p,
                                            c_int,
                                            POINTER(c_double),
                                            c_int,
                                            POINTER(c_double),
                                            c_int,
                                            c_int,
                                            c_double]
my_lib.mlp_train_classification.restype = None

def predict_image_classification(new_img):
    my_lib.mlp_classification_image.argtypes = [c_void_p, POINTER(c_double), c_int]
    my_lib.mlp_classification_image.restype = c_int
    # Deserialisation du model
    mlp = my_lib.deserialized_mlp()
    image_numpy = np.array(new_img)
    x_pointer = (c_double * len(image_numpy[0]))(*image_numpy[0])
    print("RESULT")
    test = my_lib.mlp_classification_image(mlp, x_pointer, len(image_numpy[0]))
    if test == 0:
        return "Bas"
    elif test == 1:
        return "Chaussure"
    else:
        return "Haut"

def predict_image_classification_linear_model(new_img):
    # Deserialisation du model
    model_deser = my_lib.deserialized_lineare_model()
    image_numpy = np.array(new_img)
    print("RESULT LINEARE MODEL")

    predict_value = my_lib.predict_linear_classification(model_deser, image_numpy[0].ctypes.data_as(POINTER(c_double)), len(image_numpy[0]))
    return predict_value