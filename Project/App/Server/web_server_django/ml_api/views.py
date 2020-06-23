from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

def index(request):
    message = "Salut tout le monde !"
    return HttpResponse(message)

@csrf_exempt
def save_events_json(request):
    if request.method == 'POST':
        print('Raw Data: "%s"' % request.body)
        data = request.body.decode('utf-8')
        received_json_data = json.loads(data)
        print(received_json_data['data'])
        for i in received_json_data['data']:
            print(i['titi'])
    return HttpResponse("OK")

# JSON EXEMPLE
# {
#     "data": [
#         {
#         "titi": 1
#         },
#         {
#         "titi": 2
#         }
#     ]
# }