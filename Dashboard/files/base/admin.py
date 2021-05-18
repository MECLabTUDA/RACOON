from django.contrib import admin
from .models import Locations, DailyData, Measures, MeasureData, PlotData, RacoonPermissions
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.contrib import admin

class AdminLocations(admin.ModelAdmin):
    model = Locations
    list_display = [field.name for field in Locations._meta.fields]

class AdminDailyData(admin.ModelAdmin):
    model = DailyData
    list_display = [field.name for field in DailyData._meta.fields]
    list_filter = ['location', 'date']

class AdminMeasures(admin.ModelAdmin):
    model = Measures
    list_display = [field.name for field in Measures._meta.fields]
    list_filter = ['is_main', 'is_open_ended', 'public_visible']

class AdminMeasureData(admin.ModelAdmin):
    model = MeasureData
    list_display = [field.name for field in MeasureData._meta.fields]
    list_filter = ['measure', 'daily_data__date', 'daily_data__location']

class AdminPlotData(admin.ModelAdmin):
    model = PlotData
    list_display = ['plot_id', 'daily_data', 'name_de', 'name_en', 'public_visible']
    list_filter = ['public_visible', 'daily_data__date', 'daily_data__location']

# Register your models here.
admin.site.register(Locations, AdminLocations)
admin.site.register(DailyData, AdminDailyData)
admin.site.register(Measures, AdminMeasures)
admin.site.register(MeasureData, AdminMeasureData)
admin.site.register(PlotData, AdminPlotData)

# -------------------------------------------------------------
# Additional Permissions for RACOON users
# -------------------------------------------------------------
# Define an inline admin descriptor for RacoonUser model
# which acts a bit like a singleton
class RacoonUserInline(admin.StackedInline):
    model = RacoonPermissions
    can_delete = False
    verbose_name_plural = 'Racoon Permissions'
    filter_horizontal=('visible_locations', )

# Define a new User admin
class RacoonUser(UserAdmin):
    inlines = (RacoonUserInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'get_racoon_api', 'get_racoon_all_locations_visible', 'get_racoon_visible_locations')
    list_select_related = ('racoonpermissions', )
    list_filter = ('is_staff', 'racoonpermissions__api_post_permission', 'racoonpermissions__all_locations_visible',)

    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super(RacoonUser, self).get_inline_instances(request, obj)

    def get_racoon_visible_locations(self, instance):
        locs = []
        for loc in instance.racoonpermissions.visible_locations.all():
            locs.append(str(loc))
        return ', '.join(locs)
    get_racoon_visible_locations.short_description = 'Locations'

    def get_racoon_all_locations_visible(self, instance):
        return instance.racoonpermissions.all_locations_visible
    get_racoon_all_locations_visible.short_description = 'All Locations Visible'
    get_racoon_all_locations_visible.boolean = True

    def get_racoon_api(self, instance):
        return instance.racoonpermissions.api_post_permission
    get_racoon_api.short_description = 'API Post Permission'
    get_racoon_api.boolean = True

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, RacoonUser)