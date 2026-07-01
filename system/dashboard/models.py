from django.db import models


class ViolationIncident(models.Model):
    TYPES = [
        ('running',   'Running on deck'),
        ('diving',    'Diving in restricted zone'),
        ('attire',    'Improper attire'),
        ('object',    'Prohibited object'),
        ('wristband', 'Missing wristband'),
    ]

    violation_type  = models.CharField(max_length=50, choices=TYPES)
    confidence      = models.FloatField()
    timestamp       = models.DateTimeField(auto_now_add=True)
    screenshot_path = models.CharField(max_length=500, blank=True)
    camera_id       = models.CharField(max_length=50, default='CAM-01')
    notes           = models.TextField(blank=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.get_violation_type_display()} @ {self.timestamp:%H:%M:%S}"
