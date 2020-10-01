from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from ctypes import *

my_dll_path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Lib/MachineLearning/target/debug/machine_learning_c.dll"
my_lib = CDLL(my_dll_path)
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

def convert_image_numpy_array(path, limite):
    fichiers = [f for f in listdir(path) if isfile(join(path, f))]
    img = []
    nb_img = 0
    for i in fichiers:
        if nb_img >= limite:
            break
        im = Image.open(path + (i))
        im_arr1 = np.array(im) / 255
        if im.width is 30 and len(im_arr1.shape) is 3:
            nb_img = nb_img + 1
            img.append(im_arr1)
    return np.array(img)

def save_image_numpy_as_file_npy(array, name_file):
    np.save(name_file, array)

def load_file_numpy_to_array(name_file):
    return np.load(name_file)

def flatten_img_array_numpy(arr_numpy, nb_img):
    return np.reshape(arr_numpy, (nb_img * 30 * 30 * 3))

def predict_image_classification(new_img):
    my_lib.mlp_classification_image.argtypes = [c_void_p, POINTER(c_double), c_int]
    my_lib.mlp_classification_image.restype = c_int
    # Deserialisation du model
    mlp = my_lib.deserialized_mlp()
    # path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Bas/image_not_scrap/image_resize/b26.jpg"
    # new_img = []
    # print(path)
    # im = Image.open(path)
    # im_arr1 = np.array(im) / 255
    # print(len(im_arr1.shape))
    # if im.width is 30 and len(im_arr1.shape) is 3:
    #     new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))

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


if __name__ == "__main__":

    path_bas = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Bas/image_scrap/image_resize/"
    path_chaussure= "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_scrap/image_resize/"
    path_haut = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Haut/image_scrap/image_resize/"
    path_numpy_array = "../../im_array_final.npy"

    print(predict_image_classification())
    image_numpy_bas = convert_image_numpy_array(path_bas, 90)
    image_numpy_chaussure = convert_image_numpy_array(path_chaussure, 90)
    image_numpy_haut = convert_image_numpy_array(path_haut, 90)
    image_numpy_final = np.vstack((image_numpy_bas, image_numpy_chaussure))
    image_numpy_final = np.vstack((image_numpy_final, image_numpy_haut))
    save_image_numpy_as_file_npy(image_numpy_final, path_numpy_array)

    print("print image",image_numpy_final, image_numpy_final.shape[0])

    new_arr = load_file_numpy_to_array(path_numpy_array)
    new_arr_flattened = flatten_img_array_numpy(new_arr, image_numpy_final.shape[0])
    print(new_arr_flattened.shape)