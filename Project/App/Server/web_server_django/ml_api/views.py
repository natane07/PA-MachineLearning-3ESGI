from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse


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