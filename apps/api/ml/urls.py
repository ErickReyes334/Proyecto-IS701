from django.urls import path
from .views import PrediccionImagenView  

urlpatterns = [
    path('predict-image/', PrediccionImagenView.as_view(), name='predict-image'),
]
