from django.shortcuts import render #creada por django
from django.http import StreamingHttpResponse
import cv2
import numpy as np
from django.http import JsonResponse
import random
import os
from django.conf import settings

def home(request):
    return render(request, 'home.html')

