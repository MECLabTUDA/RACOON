from django.db import models
from django.contrib.auth.models import User

# These models describe the structure of the database which will be created automatically in the migration step
class Locations(models.Model):

    location_id = models.CharField(max_length=255, primary_key=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=8, blank=False, null=False) # -90 to 90
    longitude = models.DecimalField(max_digits=11, decimal_places=8, blank=False, null=False) # -180 to 180
    name_de = models.CharField(max_length=255, default='', blank=False, null=False)
    name_en = models.CharField(max_length=255, default='', blank=False, null=False)
    description_de = models.TextField(default='', blank=True, null=False)
    description_en = models.TextField(default='', blank=True, null=False)

    class Meta:
        verbose_name_plural = "Locations"
        db_table = 'locations'

    def __str__(self):
        return self.name_en

class Measures(models.Model):
    measure_id = models.CharField(max_length=255, primary_key=True)
    public_visible = models.BooleanField(default=False, help_text="Select if this measure should be shown on the public dashboard page")
    is_main = models.BooleanField(default=False, help_text="Main Measures are measures shown on the map that can be chosen in the configuration menu")
    is_color_default = models.BooleanField(default=False, help_text="Determines the Color measure which is shown by default. Also requires a selection at 'is_main'")
    is_size_default = models.BooleanField(default=False, help_text="Determines the Size measure which is shown by default. Also requires a selection at 'is_main'")
    is_open_ended = models.BooleanField(default=True, help_text="Select if there is no upper limit of this measure. Otherwise determine the limits in 'lower_bound' and 'upper_bound'")
    name_de = models.CharField(max_length=255, blank=False, null=False)
    name_en = models.CharField(max_length=255, blank=False, null=False)
    description_de = models.TextField(default='', blank=True, null=False)
    description_en = models.TextField(default='', blank=True, null=False)
    lower_bound = models.FloatField(default=0.0, null=False, help_text="The lower bound of the measure. Only respected if 'is_open_ended' is not ticked")
    upper_bound = models.FloatField(default=0.0, null=False, help_text="The upper bound of the measure. Only respected if 'is_open_ended' is not ticked")

    class Meta:
        verbose_name_plural = "Measures"
        db_table = 'measures'

    def __str__(self):
        return self.name_en

class DailyData(models.Model):
	# Django automatically adds _id postfix since location manages a model instance named location.
    location = models.ForeignKey("Locations", on_delete=models.CASCADE, blank=False, null=False)
    date = models.DateField(blank=False, null=False)

    class Meta:
        # Gives the proper plural name for admin
        verbose_name_plural = "Daily Data"
        db_table = 'daily_data'
        unique_together = (("location", "date"),) # There will be only one enrty every day for each location
        indexes = [
           models.Index(fields=['date', 'location'])
        ]

    def __str__(self):
        return str(self.date) + ' - ' + str(self.location)

class MeasureData(models.Model):
    daily_data = models.ForeignKey("DailyData", on_delete=models.CASCADE, blank=False, null=False)
    measure = models.ForeignKey("Measures", on_delete=models.CASCADE, blank=False, null=False)
    value = models.FloatField(blank=False, null=False)

    class Meta:
        verbose_name_plural = "Measure Data"
        db_table = 'measure_data'
        unique_together = (("daily_data", "measure"),) # There will be only one enrty every day for each location
        indexes = [
           models.Index(fields=['daily_data', 'measure'])
        ]

    def __str__(self):
        return str(self.daily_data) + ' - ' + str(self.measure)

class PlotData(models.Model):
    plot_id = models.AutoField(primary_key=True)
    daily_data = models.ForeignKey("DailyData", on_delete=models.CASCADE, blank=False, null=False)
    public_visible = models.BooleanField(default=False, help_text="Select if this plot should be shown on the public dashboard page")
    name_de = models.CharField(max_length=255, blank=False, null=False)
    name_en = models.CharField(max_length=255, blank=False, null=False)
    plot_data = models.JSONField(null=False, blank=False, help_text="The actual data of the plot in form of a serialized JSON. Represents the 'options' of an apache echarts plot")

    class Meta:
        verbose_name_plural = "Plot Data"
        db_table = 'plot_data'

    def __str__(self):
        return str(self.daily_data) + ' - ' + str(self.name_en)

# Extension to user in admin panel:
class RacoonPermissions(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    visible_locations = models.ManyToManyField(Locations, blank=True, help_text="Select all locations from which the user should be allowe to view the non-public measures and plots on the dashboard.")
    all_locations_visible = models.BooleanField(default=False, help_text="Check if non-public measure from all locations should be visible for this user. If checked, the selection in 'visible locations' is ignored.")
    api_post_permission = models.BooleanField(default=False, help_text="Allow user to send POST requests to backend API for data manipulation")