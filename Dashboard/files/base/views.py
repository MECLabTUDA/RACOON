from django.shortcuts import render

from rest_framework import viewsets
from rest_framework import permissions
from rest_framework import generics
from rest_framework.authtoken.models import Token

from .models import Locations, DailyData, MeasureData, Measures, PlotData
from .serializers import LocationsSerializer, DailyDataSerializer, MeasureDataSerializer, MeasuresSerializer, PlotMetaSerializer, PlotDataSerializer, UserSerializer, GroupSerializer, PlotDataPatchSerializer, MeasureDataPatchSerializer

from django.contrib.auth.models import User, Group
from django.db.models import Min, Max
from django.http import JsonResponse
from django.http import HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import ensure_csrf_cookie
from django.db.models import Prefetch
from django.db.models import Max, Min, Avg

import csv
import codecs

from django.contrib.auth import get_user_model
from django.views import View

from datetime import datetime, timedelta

import json
import re
from .internationalization import i18nText

User = get_user_model()

@ensure_csrf_cookie
def home_light(request):
    lang = request.GET.get('lang', 'de').lower()
    lang = lang if lang in ['de', 'en'] else 'de'
    logged_in = request.user.is_authenticated
    username = request.user.username
    token = ''
    if logged_in:
        try:
            token = Token.objects.get(user=request.user)
        except Token.DoesNotExist:
            token = i18nText[lang]['no_token']

    return render(request, 'index.html', { "theme": 'LIGHT', "lang": lang, 'i18n': i18nText[lang], "logged_in":logged_in, "username": username, "api_token": token })

@ensure_csrf_cookie
def home_dark(request):
    lang = request.GET.get('lang', 'de').lower()
    lang = lang if lang in ['de', 'en'] else 'de'
    logged_in = request.user.is_authenticated
    username = request.user.username
    token = ''
    if logged_in:
        try:
            token = Token.objects.get(user=request.user)
        except Token.DoesNotExist:
            token = 'Ask admin'

    return render(request, 'index.html', { "theme": 'DARK', "lang": lang, 'i18n': i18nText[lang], "logged_in":logged_in, "username": username, "api_token": token })

class LocationViewSet(viewsets.ModelViewSet):   
    queryset = Locations.objects.all()     
    serializer_class = LocationsSerializer
    filterset_fields = ['location_id']
    
    def get_permissions(self):
        if self.request.method == 'GET':
           return [permissions.AllowAny()]
        else:
            user = self.request.user
            isApiUser = False
            if hasattr(user, 'racoonpermissions'):
                isApiUser = user.racoonpermissions.api_post_permission
            if isApiUser:
                return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]

class DailyDataViewSet(viewsets.ModelViewSet):
    queryset = DailyData.objects.none()
    serializer_class = DailyDataSerializer
    filterset_fields = {
        'location':['exact', 'in'], 
        'date':['gte', 'lte', 'exact'],
    }

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, 'racoonpermissions'):
            if user.is_staff or user.racoonpermissions.all_locations_visible:
                queryset = DailyData.objects.all().prefetch_related(Prefetch('plotdata_set', queryset=PlotData.objects.all()))
            else:
                # Show all public measures and exclude private ones where no permission exists
                permission_list = [loc.location_id for loc in user.racoonpermissions.visible_locations.all()]
                exclusion_list = [loc.location_id for loc in Locations.objects.all() if loc.location_id not in permission_list]
                queryset = DailyData.objects.all().prefetch_related(
                    Prefetch('measuredata_set', queryset=MeasureData.objects.exclude(daily_data__location__location_id__in=exclusion_list, measure__public_visible=False)),
                    Prefetch('plotdata_set', queryset=PlotData.objects.exclude(daily_data__location__location_id__in=exclusion_list, public_visible=False))
                )
        else:
            # only show the public measures
            queryset = DailyData.objects.all().prefetch_related(
                Prefetch('measuredata_set', queryset=MeasureData.objects.filter(measure__public_visible=True)),
                Prefetch('plotdata_set', queryset=PlotData.objects.filter(public_visible=True))
            )
        # Plot Data should not be passed for performance reasons. It can be obtained from the PlotData View.
        return queryset

    def patch(self, request):
        if not isinstance(request.data, dict):
            return JsonResponse(status=400, data={"non_field_errors": [f"Invalid data. Expected a dictionary, but got {type(request.data).__name__}."]}, safe=False)
        allSerializers = {}
        errors = {}

        if 'location' not in request.data:
            return JsonResponse(status=400, data={"location": ["This field is required."]}, safe=False) 
        if 'date' not in request.data:
            return JsonResponse(status=400, data={"date": ["This field is required."]}, safe=False)

        # Obtain DailyData Object
        location_id = request.data['location']
        date = request.data['date']
        try:
            dailyDataObject = DailyData.objects.get(location__location_id=location_id, date=date)
            # Process new Measure Data
            if 'measureData' in request.data:
                measureData = request.data['measureData']
                for mD in measureData:
                    mD['daily_data'] = dailyDataObject.pk
                
                mDserializer = MeasureDataPatchSerializer(data=measureData, many=True)
                if not mDserializer.is_valid():
                    errors['measureData'] = mDserializer.errors
                allSerializers['measureData'] = mDserializer

            # Process new Plots
            if 'plotData' in request.data:
                plotData = request.data['plotData']
                for pD in plotData:
                    pD['daily_data'] = dailyDataObject.pk

                pDserializer = PlotDataPatchSerializer(data=plotData, many=True)
                allSerializers['plotData'] = pDserializer

                if not pDserializer.is_valid():
                    errors['plotData'] = pDserializer.errors

            # Serialize and Check Data
            if errors:
                return JsonResponse(status=400, data=errors, safe=False)
            else:
                resp = {}
                resp['location'] = location_id
                resp['date'] = date
                for name, serializer in allSerializers.items():
                    serializer.save()
                    resp[name] = serializer.data
                return JsonResponse(status=201, data=resp, safe=False)

        except DailyData.DoesNotExist:
            return JsonResponse(status=400, data={"non_field_errors": [f"data object '{date} - {location_id}' does not exist"]}, safe=False)

    
    def get_permissions(self):
        if self.request.method == 'GET':
            return [permissions.AllowAny()]
        else:
            user = self.request.user
            isApiUser = False
            if hasattr(user, 'racoonpermissions'):
                isApiUser = user.racoonpermissions.api_post_permission
            if isApiUser:
                return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]

class MeasureDataViewSet(viewsets.ModelViewSet):
    queryset = MeasureData.objects.all()
    serializer_class = MeasureDataSerializer
    filterset_fields = ['measure_id', 'daily_data__date', 'daily_data__location']

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, 'racoonpermissions'):
            if user.is_staff or user.racoonpermissions.all_locations_visible:
                queryset = MeasureData.objects.all()
            else:
                # Show all public measureData and exclude private ones where no permission exists
                permission_list = [loc.location_id for loc in user.racoonpermissions.visible_locations.all()]
                exclusion_list = [loc.location_id for loc in Locations.objects.all() if loc.location_id not in permission_list]
                queryset = MeasureData.objects.exclude(daily_data__location__location_id__in=exclusion_list, measure__public_visible=False)
        else:
            queryset = MeasureData.objects.filter(measure__public_visible=True)
        return queryset
    
    def get_permissions(self):
        if self.request.method == 'GET':
           return [permissions.AllowAny()]
        else:
            user = self.request.user
            isApiUser = False
            if hasattr(user, 'racoonpermissions'):
                isApiUser = user.racoonpermissions.api_post_permission
            if isApiUser:
                return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]



class PlotDataViewSet(viewsets.ModelViewSet):
    queryset = PlotData.objects.all()
    serializer_class = PlotDataSerializer
    filterset_fields = ['plot_id', 'daily_data__date', 'daily_data__location']

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, 'racoonpermissions'):
            if user.is_staff or user.racoonpermissions.all_locations_visible:
                queryset = PlotData.objects.all()
            else:
                # Show all public measureData and exclude private ones where no permission exists
                permission_list = [loc.location_id for loc in user.racoonpermissions.visible_locations.all()]
                exclusion_list = [loc.location_id for loc in Locations.objects.all() if loc.location_id not in permission_list]
                queryset = PlotData.objects.exclude(daily_data__location__location_id__in=exclusion_list, public_visible=False)
        else:
            queryset = PlotData.objects.filter(public_visible=True)
        return queryset
    
    def get_permissions(self):
        if self.request.method == 'GET':
           return [permissions.AllowAny()]
        else:
            user = self.request.user
            isApiUser = False
            if hasattr(user, 'racoonpermissions'):
                isApiUser = user.racoonpermissions.api_post_permission
            if isApiUser:
                return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]

class MeasuresViewSet(viewsets.ModelViewSet):
    queryset = Measures.objects.all()
    serializer_class = MeasuresSerializer
    #permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        user = self.request.user
        if user.is_authenticated:
            queryset = Measures.objects.all()

        if hasattr(user, 'racoonpermissions'):
            if user.is_staff or user.racoonpermissions.all_locations_visible:
                queryset = Measures.objects.all()
            else:
                # Only show a list of all measures if a user has permissions for the measures of at least one location
                permission_list = [loc.location_id for loc in user.racoonpermissions.visible_locations.all()]
                if len(permission_list) > 0 :                    
                    queryset = Measures.objects.all()
                else:
                    queryset = Measures.objects.exclude(**{'public_visible': False})
        else:
            queryset = Measures.objects.exclude(**{'public_visible': False})

        return queryset
    
    def get_permissions(self):
        if self.request.method == 'GET':
           return [permissions.AllowAny()]
        else:
            user = self.request.user
            isApiUser = False
            if hasattr(user, 'racoonpermissions'):
                isApiUser = user.racoonpermissions.api_post_permission
            if isApiUser:
                return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]
        
def AverageDataView(request):
    user = request.user
    if hasattr(user, 'racoonpermissions'):
        if user.is_staff or user.racoonpermissions.all_locations_visible:
            prefiltered_queryset = MeasureData.objects.all()
        else:
            # Show all public measureData and exclude private ones where no permission exists
            permission_list = [loc.location_id for loc in user.racoonpermissions.visible_locations.all()]
            exclusion_list = [loc.location_id for loc in Locations.objects.all() if loc.location_id not in permission_list]
            prefiltered_queryset = MeasureData.objects.exclude(daily_data__location__location_id__in=exclusion_list, measure__public_visible=False)
    else:
        prefiltered_queryset = MeasureData.objects.filter(measure__public_visible=True)

    if 'date' in request.GET:        
        result = prefiltered_queryset.filter(daily_data__date=request.GET['date__lte'])
    elif 'date__gte' in request.GET and 'date__lte' in request.GET:
        result = prefiltered_queryset.filter(daily_data__date__gte=request.GET['date__gte'], daily_data__date__lte=request.GET['date__lte'])
    elif 'date__gte' in request.GET:
        result = prefiltered_queryset.filter(daily_data__date__gte=request.GET['date__gte'])
    elif 'date__lte' in request.GET:
        result = prefiltered_queryset.filter(daily_data__date__lte=request.GET['date__lte'])
    else:
        result = prefiltered_queryset.filter(daily_data__date=datetime.now().strftime("%Y-%m-%d"))

    if 'location__in' in request.GET:
        result = result.filter(daily_data__location__in=request.GET['location__in'].split(','))

    result = result.values('daily_data__date', 'measure__measure_id').annotate(total=Avg('value'))

    final_result = {}

    for res in result:
        date = res['daily_data__date'].strftime("%Y-%m-%d")
        avg = res['total']
        measure = res['measure__measure_id']
        if not measure in final_result.keys():
            final_result[measure] = {}
        final_result[measure][date] = avg

    return JsonResponse(final_result, safe=False)

def ExportDataView(request):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="data.csv"'
    
    # Force response to be UTF-8 - This is where the magic happens
    response.write(codecs.BOM_UTF8)

    if 'date__gte' in request.GET and 'date__lte' in request.GET:
        dataObject = MeasureData.objects.filter(daily_data__date__gte=request.GET['date__gte'], daily_data__date__lte=request.GET['date__lte'])
    else:
        dataObject = MeasureData.objects.filter(daily_data__date=datetime.now().strftime("%Y-%m-%d"))
    
    if 'location__in' in request.GET:
        dataObject = dataObject.filter(daily_data__location__in=request.GET['location__in'].split(','))

    user = request.user
    if hasattr(user, 'racoonpermissions'):
        if user.is_staff or user.racoonpermissions.all_locations_visible:
            dataObject = dataObject.all()
        else:
            # Show all public measures and exclude private ones where no permission exists
            permission_list = [loc.location_id for loc in user.racoonpermissions.visible_locations.all()]
            exclusion_list = [loc.location_id for loc in Locations.objects.all() if loc.location_id not in permission_list]
            dataObject = dataObject.exclude(daily_data__location__location_id__in=exclusion_list, measure__public_visible=False)
    else:
        dataObject = dataObject.filter(measure__public_visible=True)


    users = User.objects.all()
    header = [
        "location",
        "date",
        "measure",
        "value",
    ]
    
    writer = csv.DictWriter(response, fieldnames=header)
    writer.writeheader()
    
    for entry in dataObject:
        writer.writerow(
            {
                "location": entry.daily_data.location_id,
                "date": entry.daily_data.date,
                "measure": entry.measure.measure_id,
                "value": entry.value,
            }
        )
        
    return response

def MetaDataView(request):
    result = MeasureData.objects

    measure_ids = [(measure.measure_id, measure.public_visible) for measure in Measures.objects.all()]

    aggs = []
    aggs.append(Min('value'))
    aggs.append(Max('value'))

    final_result = []
    for measure_id, is_public in measure_ids:
        result = MeasureData.objects.filter(measure=measure_id)
        result = result.aggregate(*aggs)

        res = {
            'measure_id': measure_id, 
            'min_value': result['value__min'],
            'max_value': result['value__max']
        }

        if not result['value__min']:
            res['min_value'] = 0
        
        if not result['value__max']:
            res['max_value'] = 0

        if not request.user.is_authenticated:
            if is_public:
                final_result.append(res)
        else:
            final_result.append(res)

    return JsonResponse(final_result, safe=False)

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

def user_login(request):
    if request.method == "POST":
        if request.is_ajax():
            username = request.POST['username']
            password = request.POST['password']

            user = authenticate(username=username, password=password)
            if user is not None:
                if user.is_active:
                    login(request,user)                    
                    lang = request.GET.get('lang', 'de').lower()
                    lang = lang if lang in ['de', 'en'] else 'de'
                    try:
                        token = Token.objects.get(user=user)
                    except Token.DoesNotExist:
                        token = i18nText[lang]['no_token']
                    data = {'success': True, 'token':str(token)}
                    
                else:
                    data = {'success': False, 'error': 'User is not active'}
            else:
                data = {'success': False, 'error': 'Wrong username and/or password'}
        return HttpResponse(json.dumps(data), content_type='application/json')

    return HttpResponseBadRequest()

def user_logout(request):
    if request.method == "POST":
        logout(request)
        data = {'success': True}
        return HttpResponse(json.dumps(data), content_type='application/json')

    return HttpResponseBadRequest()