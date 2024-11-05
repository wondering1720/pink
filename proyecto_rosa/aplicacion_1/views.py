from django.shortcuts import render #creada por django
from django.http import StreamingHttpResponse
import cv2
import numpy as np
from django.http import JsonResponse
import random
import os
from django.conf import settings

# Rutas absolutas para los archivos Haar Cascade
ruta_full_body = os.path.join(settings.BASE_DIR, "aplicacion_1", "static", "haarcascade", "haarcascade_fullbody.xml")
ruta_mouth = os.path.join(settings.BASE_DIR, "aplicacion_1", "static", "haarcascade", "haarcascade_mcs_mouth.xml")

if not os.path.exists(ruta_full_body):
    raise FileNotFoundError(f"El archivo {ruta_full_body} no se encuentra.")
if not os.path.exists(ruta_mouth):
    raise FileNotFoundError(f"El archivo {ruta_mouth} no se encuentra.")

def home(request):
    return render(request, 'home.html')

def camara(request):
    return render(request, 'camara.html')

def extras(request):
    return render(request, 'extras.html')

modo_actual = 'cara'

def deformar_imagen(frame):
    global modo_actual  # Usar la variable global para almacenar el modo

    # Cargar los clasificadores Haar Cascade
    rostro_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ojo_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    full_body = cv2.CascadeClassifier(ruta_full_body)
    boca_cascade = cv2.CascadeClassifier(ruta_mouth)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if modo_actual == 'cara':
        # Detectar rostros
        rostros = rostro_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Aplicar desenfoque en las áreas de los rostros detectados
        for (x, y, w, h) in rostros:
            # Extraer la región de la cara en el marco
            rostro_region = frame[y:y+h, x:x+w]
            # Aplicar un desenfoque gaussiano en la región del rostro
            rostro_difuminado = cv2.GaussianBlur(rostro_region, (35, 35), 0)
            # Colocar la región desenfocada de vuelta en el marco original
            frame[y:y+h, x:x+w] = rostro_difuminado
    
    elif modo_actual == 'ojos':
        # Detectar ojos
        ojos = ojo_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (ex, ey, ew, eh) in ojos:
            # Extraer la región del ojo y aplicar un desenfoque Gaussiano
            ojo_region = frame[ey:ey+eh, ex:ex+ew]
            ojo_difuminado = cv2.GaussianBlur(ojo_region, (15, 15), 0)
            
            # Colocar la región difuminada de vuelta en el fotograma original
            frame[ey:ey+eh, ex:ex+ew] = ojo_difuminado
            
            # Opcional: dibujar un rectángulo alrededor del ojo difuminado
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    elif modo_actual == 'full_body':
        # Detectar cuerpo completo y desenfocar la región detectada
        cuerpos = full_body.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (cx, cy, cw, ch) in cuerpos:
            cuerpo_region = frame[cy:cy+ch, cx:cx+cw]
            cuerpo_difuminado = cv2.GaussianBlur(cuerpo_region, (35, 35), 0)  # Aumenta el desenfoque para una región más grande
            frame[cy:cy+ch, cx:cx+cw] = cuerpo_difuminado
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)  # Color rojo para la región de cuerpo completo

    elif modo_actual == 'boca':
        # Detectar boca y desenfocarla
        bocas = boca_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (mx, my, mw, mh) in bocas:
            boca_region = frame[my:my+mh, mx:mx+mw]
            boca_difuminada = cv2.GaussianBlur(boca_region, (15, 15), 0)
            frame[my:my+mh, mx:mx+mw] = boca_difuminada
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)  # Color cian para la boca
    
    elif modo_actual == 'color':
        # Cambiar el color de toda la imagen
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertir a HSV
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float32) + random.randint(0, 180)) % 180  # Cambiar tono aleatorio
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)  # Convertir de vuelta a BGR

    return frame

def generar_video():
    cap = cv2.VideoCapture(0)  # Accede a la cámara
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = deformar_imagen(frame)  # Deformar el frame capturado
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

def video(request):
    return StreamingHttpResponse(generar_video(), content_type='multipart/x-mixed-replace; boundary=frame')

def cambiar_modo(request):
    global modo_actual
    modo = request.GET.get('modo', 'cara')  # Por defecto es 'cara'
    modo_actual = modo
    return JsonResponse({'modo': modo_actual})
