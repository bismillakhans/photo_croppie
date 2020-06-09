import time

from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views import View

from uploadApp.forms import PhotoForm
from uploadApp.models import Photo





class DragAndDropUploadView(View):
    def get(self, request):
        photos_list = Photo.objects.all()
        return render(self.request, 'uploadApp/upload.html', {'photos': photos_list})

    def post(self, request):
        time.sleep(1)
        form = PhotoForm(self.request.POST, self.request.FILES)
        if form.is_valid():
            photo = form.save()
            data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url}
        else:
            data = {'is_valid': False}
        return JsonResponse(data)

def index(request):
    return render(request,"uploadApp/index.html")


