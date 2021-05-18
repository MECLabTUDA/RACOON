"""racoon URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from django.contrib.auth.views import LoginView
from django.conf.urls import url

from . import views

from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'locations', views.LocationViewSet, basename='Locations')
router.register(r'data', views.DailyDataViewSet)
router.register(r'measures', views.MeasuresViewSet)
router.register(r'measureData', views.MeasureDataViewSet)
router.register(r'plotData', views.PlotDataViewSet)
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

urlpatterns = [
    path('', views.home_light, name='home-light'),
    path('dark/', views.home_dark, name='home-dark'),
    path('api/', include(router.urls)),
    path('api/meta/', views.MetaDataView, name='meta-data'),
    path('api/avg/', views.AverageDataView, name='avg-data'),
    path('api/export/', views.ExportDataView, name='data-export'),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^user_login/$', views.user_login, name='user_login'),
    url(r'^user_logout/$', views.user_logout, name='user_logout'),
]
