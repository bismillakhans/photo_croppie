import time
import base64
from io import BytesIO
import os
from django.http import JsonResponse
from django.shortcuts import render

from django.conf import settings
# Create your views here.
from django.views import View

from uploadApp.forms import PhotoForm
from uploadApp.models import Photo
from file_upload_ajax.face_crop import crop_image


class DragAndDropUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'uploadApp/upload.html', {'photos': photos_list})

    def post(self, request):
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            result_image = crop_image(os.path.join(settings.MEDIA_ROOT, photo.file.name))
            buffered = BytesIO()
            result_image.save(buffered, format='PNG')
            buffered.seek(0)
            img_byte = buffered.getvalue()
            image_string = "data:image/png;base64," + base64.b64encode(img_byte).decode()

            data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url, 'image_string': image_string}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)


def index(request):
    return render(request, "uploadApp/index.html")
