import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from .models import ViolationIncident


def dashboard(request):
    last = ViolationIncident.objects.order_by('-id').first()
    today = timezone.now().date()
    context = {
        'last_violation_id': last.id if last else 0,
        'total_today': ViolationIncident.objects.filter(timestamp__date=today).count(),
    }
    return render(request, 'dashboard/index.html', context)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def violations_api(request):
    if request.method == 'GET':
        since_id = int(request.GET.get('since_id', 0))
        qs = ViolationIncident.objects.filter(id__gt=since_id).order_by('id')[:50]
        today = timezone.now().date()
        data = [
            {
                'id':        v.id,
                'type':      v.violation_type,
                'label':     v.get_violation_type_display(),
                'confidence': round(v.confidence, 2),
                'timestamp': v.timestamp.strftime('%H:%M:%S'),
                'camera_id': v.camera_id,
            }
            for v in qs
        ]
        return JsonResponse({
            'violations': data,
            'total': ViolationIncident.objects.filter(timestamp__date=today).count(),
        })

    body = json.loads(request.body)
    incident = ViolationIncident.objects.create(
        violation_type=body.get('type', 'running'),
        confidence=float(body.get('confidence', 0.0)),
        screenshot_path=body.get('screenshot_path', ''),
        camera_id=body.get('camera_id', 'CAM-01'),
    )
    return JsonResponse({'id': incident.id, 'status': 'created'}, status=201)


def status_api(request):
    today = timezone.now().date()
    return JsonResponse({
        'status': 'running',
        'model':  'YOLOv8n',
        'total_today': ViolationIncident.objects.filter(timestamp__date=today).count(),
        'total_all':   ViolationIncident.objects.count(),
    })
