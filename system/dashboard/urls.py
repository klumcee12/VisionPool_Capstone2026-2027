from django.urls import path
from . import views

urlpatterns = [
    path('',                  views.dashboard,      name='dashboard'),
    path('api/violations/',   views.violations_api, name='violations_api'),
    path('api/status/',       views.status_api,     name='status_api'),
]
