from django.conf.urls import url

from . import views # import views so we can use them in urls.


urlpatterns = [
    url(r'^$', views.index), # "/store" will call the method "index" in "views.py"
    url(r'save_mlp', views.save_mlp_json),  # "/store" will call the method "index" in "views.py"
    url(r'predict_image', views.predict_image),
    url(r'lineare_model', views.lineare_model),
    url(r'json_load_modele_lineaire', views.json_load_modele_lineaire),
]