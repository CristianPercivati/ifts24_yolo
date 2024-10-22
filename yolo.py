# Importar librerías
from ultralytics import YOLO # Importar el modelo YOLO de Ultralytics
import cv2 # Importar la biblioteca OpenCV
import math # Importar el módulo math para operaciones matemáticas
# Iniciar la webcam
cap = cv2.VideoCapture(0) # Abrir la cámara predeterminada (índice 0)
cap.set(3, 640) # Establecer el ancho del fotograma en 640 píxeles
cap.set(4, 480) # Establecer la altura del fotograma en 480 píxeles
# Cargar el modelo YOLO
model = YOLO("yolo-Weights/yolov8n.pt") # Cargar el modelo YOLOv8 con pesos pre-entrenados
# Definir las clases de objetos para la detección
classNames = ["persona", "bicicleta", "coche", "motocicleta", "avión", "autobús", "tren", "camión", "bote",
"semáforo", "boca de incendios", "señal de stop", "parquímetro", "banco", "pájaro", "gato",
"perro", "caballo", "oveja", "vaca", "elefante", "oso", "cebra", "jirafa", "mochila", "paraguas",
"bolso", "corbata", "maleta", "frisbee", "esquís", "snowboard", "balón deportivo", "cometa", "bate de béisbol",
"guante de béisbol", "monopatín", "tabla de surf", "raqueta de tenis", "botella", "copa de vino",
"taza",
"tenedor", "cuchillo", "cuchara", "cuenco", "banana", "manzana", "sándwich", "naranja",
"brócoli",
"zanahoria", "perrito caliente", "pizza", "donut", "pastel", "silla", "sofá", "planta en maceta",
"cama",
"mesa de comedor", "retrete", "monitor de TV", "ordenador portátil", "ratón", "control remoto",
"teclado", "teléfono móvil",
"microondas", "horno", "tostadora", "fregadero", "refrigerador", "libro", "reloj", "florero",
"tijeras",
"oso de peluche", "secador de pelo", "cepillo de dientes"]

# Bucle infinito para capturar continuamente fotogramas de la cámara
while True:
    # Leer un fotograma de la cámara
    success, img = cap.read()
    # Realizar la detección de objetos utilizando el modelo YOLO en el fotograma capturado
    results = model(img, stream=True)
    # Iterar a través de los resultados de la detección de objetos
    for r in results:
        boxes = r.boxes # Extraer las cajas delimitadoras de los objetos detectados
    # Iterar a través de cada caja delimitadora
        for box in boxes:
    # Extraer coordenadas de la caja delimitadora
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convertir a valores enteros
            # Dibujar la caja delimitadora en el fotograma
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Calcular e imprimir la puntuación de confianza de la detección
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confianza --->", confidence)
            # Determinar e imprimir el nombre de la clase del objeto detectado
            cls = int(box.cls[0])
            print("Nombre de la clase -->", classNames[cls])
            # Dibujar texto indicando el nombre de la clase en el fotograma
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            # Mostrar el fotograma con los objetos detectados en una ventana llamada "Cam"
    cv2.imshow('Cam', img)
    # Comprobar si se presionó la tecla 'q' para salir del bucle
    if cv2.waitKey(1) == ord('q'):
        break
# Liberar la cámara
cap.release()
# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()