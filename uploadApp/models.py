
from django.db import models


class Photo(models.Model):
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Face(models.Model):
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='photo')
    file = models.FileField(upload_to='faces/')  

