from django import forms

from uploadApp.models import Photo


class PhotoForm(forms.ModelForm):
    class Meta:
        model=Photo
        fields=('file',)
