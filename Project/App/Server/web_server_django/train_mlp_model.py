from ctypes import *
from typing import List
import numpy as np
from PIL import Image

def transform(l: List[List[float]]):
    result =[]
    for i in l:
        for j in i:
            result.append(j)
    return result



# # # Chargement de la lib
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

def predict_image_classification():
    # Deserialisation du model
    mlp = my_lib.deserialized_mlp()
    path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_not_scrap/image_resize/c1.jpg"
    new_img = []
    print(path)
    im = Image.open(path)
    im_arr1 = np.array(im) / 255
    print(len(im_arr1.shape))
    if im.width is 30 and len(im_arr1.shape) is 3:
        new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))

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

    # predict_image_classification()

    # Deserialisation du mlp.json
    # mlp = my_lib.deserialized_mlp()
    #
    # path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_not_scrap/image_resize/c3.jpg"
    # new_img = []
    #
    # print(path)
    # im = Image.open(path)
    # im_arr1 = np.array(im) / 255
    # print(len(im_arr1.shape))
    # if im.width is 30 and len(im_arr1.shape) is 3:
    #     new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))
    #
    # image_numpy = np.array(new_img)
    # x_pointer = (c_double * len(image_numpy[0]))(*image_numpy[0])
    # print("RESULT")
    # test = my_lib.mlp_classification(mlp, x_pointer, len(image_numpy[0]))


    # Paramétrage du nombre d'image utilisé par classe
    path_numpy_array = "im_array_final.npy"
    nb_img_bas = 90
    nb_img_chaussure = 90
    nb_img_haut = 90

    # Chargement du tableau numpy des images
    new_arr = np.load(path_numpy_array)
    image_arr = []
    image_arr_result = []
    for number, arr in enumerate(new_arr):
        print(number, arr)
        image_arr.append(np.reshape(arr, (30 * 30 * 3)))
        if number < nb_img_bas:
            image_arr_result.append([1, -1, -1])
        elif number >= nb_img_bas and number < nb_img_chaussure + nb_img_bas:
            image_arr_result.append([-1, 1, -1])
        else:
            image_arr_result.append([-1, -1, 1])

    # Creation du réseau du model MLP
    npl = list([2700, 10, 3])
    npl_pointer = (c_int64 * 3)(*npl)
    mlp = my_lib.create_mlp(npl_pointer, len(npl))

    x_flatten = transform(image_arr)
    y_flatten = transform(image_arr_result)
    x_train_pointer = (c_double * len(x_flatten))(*x_flatten)
    y_train_pointer = (c_double * len(y_flatten))(*y_flatten)


    # Entrainement du model MLP
    my_lib.mlp_train_classification(mlp, len(image_arr), x_train_pointer, len(x_flatten), y_train_pointer,
                                    len(y_flatten), 100, 0.001)

    # le nombre d'image correctement predit
    good_reponse = 0
    for i in range(len(image_arr)):
        x_pointer = (c_double * len(image_arr[i]))(*image_arr[i])
        test = my_lib.mlp_classification_image(mlp, x_pointer, len(image_arr[i]))
        print(image_arr_result[i])
        if image_arr_result[i][test] == 1:
            good_reponse = good_reponse + 1
        print(test)
    print(good_reponse)
    print("finish")
    # Serialisation des donnée
    my_lib.serialized_mlp(mlp)

    # Deserialisation du mlp.json
    # mlp = my_lib.deserialized_mlp()
    #
    # path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_not_scrap/image_resize/c3.jpg"
    # new_img = []
    #
    # print(path)
    # im = Image.open(path)
    # im_arr1 = np.array(im) / 255
    # print(len(im_arr1.shape))
    # if im.width is 30 and len(im_arr1.shape) is 3:
    #     new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))
    #
    # image_numpy = np.array(new_img)
    # x_pointer = (c_double * len(image_numpy[0]))(*image_numpy[0])
    # print("RESULT")
    # test = my_lib.mlp_classification(mlp, x_pointer, len(image_numpy[0]))