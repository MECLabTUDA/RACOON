from .models import Locations, DailyData, MeasureData, Measures, PlotData
from rest_framework import serializers
from django.contrib.auth.models import User, Group
from rest_framework.exceptions import PermissionDenied

class LocationsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Locations
        fields = ['location_id', 'latitude', 'longitude', 'name_de', 'name_en', 'description_de', 'description_en']

class MeasuresSerializer(serializers.ModelSerializer):
    class Meta:
        model = Measures
        fields = ['measure_id', 'public_visible', 'is_main', 'is_color_default', 'is_size_default', 'is_open_ended', 'lower_bound', 'upper_bound', 'name_de', 'name_en', 'description_de', 'description_en']

class MeasureDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MeasureData
        fields = ['measure', 'value']

class PlotDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlotData
        fields = ['plot_id', 'name_de', 'name_en', 'public_visible', 'plot_data']

class PlotMetaSerializer(serializers.ModelSerializer):
    # Difference to PlotDataSerializer: It hides the field plot_data for GET requests (performance reasons)
    class Meta:
        model = PlotData
        fields = ['plot_id', 'name_de', 'name_en', 'public_visible', 'plot_data']
        extra_kwargs = {
            'plot_data': {'write_only': True},
        }

class MeasureDataPatchSerializer(serializers.ModelSerializer):
    # Used for PATCH request. Contains the field daily_data
    class Meta:
        validators = []
        model = MeasureData
        fields = ['measure', 'value', 'daily_data']
        extra_kwargs = {
            'daily_data': {'write_only': True},
        }

    # This is called on PATCH requests to /api/data and overwrites values of existing entries
    def create(self, validated_data):
        # check if MeasueData exists. If yes: update with new value, If no: create new MeasureData
        try:
            item = MeasureData.objects.get(measure=validated_data['measure'], daily_data=validated_data['daily_data'])
        except MeasureData.DoesNotExist:
            MeasureData.objects.create(measure=validated_data['measure'], daily_data=validated_data['daily_data'], value = validated_data['value'])
        else:            
            item = MeasureData.objects.filter(measure=validated_data['measure'], daily_data=validated_data['daily_data']).update(value = validated_data['value'])

        item = MeasureData.objects.get(measure=validated_data['measure'], daily_data=validated_data['daily_data'])
        return item

class PlotDataPatchSerializer(serializers.ModelSerializer):
    # Used for PATCH request. Contains the field daily_data
    class Meta:
        model = PlotData
        fields = ['plot_id', 'name_de', 'name_en', 'public_visible', 'plot_data', 'daily_data']
        extra_kwargs = {
            'daily_data': {'write_only': True},
        }

class DailyDataSerializer(serializers.ModelSerializer):
    measureData = MeasureDataSerializer(many=True, source='measuredata_set')
    # PlotMetaSerializer hides the actual plot data (write_only). This is for performance reasons
    # PlotData should be obtained by requesting it from the PlotData View
    plotData = PlotMetaSerializer(many=True, source='plotdata_set')

    class Meta:
        model = DailyData
        fields = ['location', 'date', 'measureData', 'plotData']

    def create(self, validated_data):
        measureDataList = validated_data.pop('measuredata_set', [])
        plotDataLisit = validated_data.pop('plotdata_set', [])

        # Create the actual DailyData Entry
        instance = DailyData.objects.create(**validated_data)

        # Create the MeasureData Entries
        for mD in measureDataList:
            mD.update({'daily_data_id': instance.id})
            measureDataInstance = MeasureData.objects.create(**mD)

        # Create the PlotData Entries. This also takes plot_data.
        for pD in plotDataLisit:
            pD.update({'daily_data_id': instance.id})
            plotDataInstance = PlotData.objects.create(**pD)

        return instance


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']