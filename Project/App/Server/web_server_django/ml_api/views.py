from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
import numpy as np
from PIL import Image
from .controller import lib_ml

# path = "C:/Users/MOI/dev/PA-MachineLearning-3ESGI/Project/App/Server/web_server_django/ml_api/controller/"
def index(request):
    message = "Salut tout le monde !"
    return HttpResponse(message)

@csrf_exempt
def save_mlp_json(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        response_data = {}
        if data:
            received_json_data = json.loads(data)
            handle1 = open('mlp.json', 'w+')
            json_str = json.dumps(received_json_data, indent=4)
            handle1.write(json_str)
            handle1.close()
            response_data['success'] = True
            response_data['message'] = 'Le model MLP est sauvegardé'
        else:
            response_data['success'] = False
            response_data['message'] = 'Not body json'
    return JsonResponse(response_data)

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        path = request.FILES["image"]
        new_img = []
        print(path)
        im = Image.open(path)
        im_arr1 = np.array(im) / 255
        print(len(im_arr1.shape))
        if im.width is 30 and len(im_arr1.shape) is 3:
            new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))
        result = lib_ml.predict_image_classification(new_img)

        image_numpy = np.array(new_img)
        print(image_numpy[0])
        response_data = {}
        response_data['success'] = True
        response_data['message'] = result
    return JsonResponse(response_data)

@csrf_exempt
def json_load_modele_lineaire(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        response_data = {}
        if data:
            received_json_data = json.loads(data)
            handle1 = open('lineare_model.json', 'w+')
            json_str = json.dumps(received_json_data, indent=4)
            handle1.write(json_str)
            handle1.close()
            response_data['success'] = True
            response_data['message'] = 'Le model lineaire est sauvegarde'
        else:
            response_data['success'] = False
            response_data['message'] = 'Not body json'
    return JsonResponse(response_data)

@csrf_exempt
def lineare_model(request):
    if request.method == 'POST':
        path = request.FILES["image"]
        new_img = []
        print(path)
        im = Image.open(path)
        im_arr1 = np.array(im) / 255
        print(len(im_arr1.shape))
        if im.width is 30 and len(im_arr1.shape) is 3:
            new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))
        result = lib_ml.predict_image_classification_linear_model(new_img)
        response_data = {}
        response_data['success'] = True
        response_data['message'] = result
    return JsonResponse(response_data)

@csrf_exempt
def tf_predict(request):
    if request.method == 'POST':
        path = request.FILES["image"]
        new_img = []
        print(path)
        im = Image.open(path)
        im_arr1 = np.array(im) / 255
        print(len(im_arr1.shape))
        if im.width is 30 and len(im_arr1.shape) is 3:
            new_img.append(np.reshape(im_arr1, (30 * 30 * 3)))
        result = lib_ml.predict_image_tenserflow(new_img)
        response_data = {}
        response_data['success'] = True
        response_data['message'] = result
    return JsonResponse(response_data)