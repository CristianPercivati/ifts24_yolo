# Importar la biblioteca OpenCV
import cv2

# Abrir la cámara predeterminada (índice 0), si incluyes otras como una webcam externa puedes cambiarlo a (1)
cap = cv2.VideoCapture(0)

# Establecer el ancho del fotograma en 640 píxeles
cap.set(3, 640)
# Establecer la altura del fotograma en 480 píxeles
cap.set(4, 480)

# Bucle infinito para capturar continuamente fotogramas de la cámara
while True:
    # Leer un fotograma de la cámara
    ret, img = cap.read()
    # Mostrar el fotograma capturado en una ventana llamada "Cam"
    cv2.imshow('Cam', img)
    # Esperar una pulsación de tecla durante 1 milisegundo
    # Si la tecla presionada es 'q', salir del bucle
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara
cap.release()
# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()