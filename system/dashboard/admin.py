from django.contrib import admin
from .models import ViolationIncident


@admin.register(ViolationIncident)
class ViolationIncidentAdmin(admin.ModelAdmin):
    list_display  = ('violation_type', 'confidence_pct', 'timestamp', 'camera_id')
    list_filter   = ('violation_type', 'camera_id')
    search_fields = ('violation_type', 'notes')
    ordering      = ('-timestamp',)
    readonly_fields = ('timestamp',)

    def confidence_pct(self, obj):
        return f"{obj.confidence * 100:.0f}%"
    confidence_pct.short_description = 'Confidence'
