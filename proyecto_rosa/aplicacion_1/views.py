from django.shortcuts import render #creada por django
from django.http import StreamingHttpResponse
import cv2

def home(request):
    return render(request, 'home.html')

def camara(request):
    return render(request, 'camara.html')

def extras(request):
    return render(request, 'extras.html')

def deformar_imagen(frame):
    # Aquí puedes aplicar las transformaciones que desees; por ejemplo, cambiar colores o distorsionar la imagen.
    # Vamos a aplicar una simple distorsión.
    height, width = frame.shape[:2]
    # Deformación: ejemplo de escala.
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1)  # Rotación de 15 grados
    frame_deformado = cv2.warpAffine(frame, matrix, (width, height))
    return frame_deformado

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