from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

def convert_image_numpy_array(path):
    fichiers = [f for f in listdir(path) if isfile(join(path, f))]
    img = []
    for i in fichiers:
        im = Image.open(path + (i))
        im_arr1 = np.array(im) / 255
        if im.width is 30 and len(im_arr1.shape) is 3:
            img.append(im_arr1)
    return np.array(img)

def save_image_numpy_as_file_npy(array, name_file):
    np.save(name_file, array)

def load_file_numpy_to_array(name_file):
    return np.load(name_file)

def flatten_img_array_numpy(arr_numpy, nb_img):
    return np.reshape(arr_numpy, (nb_img * 30 * 30 * 3))

path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/Dataset/Raw/Chaussure/image_scrap/image_resize/"
