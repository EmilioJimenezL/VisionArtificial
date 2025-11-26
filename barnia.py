import cv2
import numpy as np

# -----------------------------------------
# Parámetros de segmentación de color rojo
# -----------------------------------------
lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

lower_red2 = np.array([170, 100, 80], dtype=np.uint8)
upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

# Elemento estructurante para morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Parámetros para filtrar pelotas
AREA_MIN = 300        # área mínima del contorno
CIRC_MIN = 0.70       # circularidad mínima

# Parámetros de "control" (ganancias P muy sencillas)
Kx = 0.002   # ganancia para movimiento lateral / yaw
Ky = 0.002   # ganancia para altura
Ka = 0.0002  # ganancia para avance/retroceso según área
AREA_REF = 8000  # área deseada de la pelota (cuanto más grande, más cerca)


def segmentar_rojo(hsv):
    """Regresa una máscara binaria del color rojo en HSV."""
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Morfología para limpiar ruido
    mask_open = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask_clean


def detectar_pelota(mask, frame):
    """
    Busca la pelota roja principal en la máscara.
    Regresa:
        cx, cy: centro de la pelota (en pixeles)
        area: área del contorno
        frame_vis: frame con dibujos
    Si no encuentra pelota, regresa cx = cy = None.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    frame_vis = frame.copy()

    mejor_cnt = None
    mejor_area = 0
    mejor_circ = 0
    mejor_bbox = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue

        circularidad = (4 * np.pi * area) / (perim ** 2)

        if circularidad < CIRC_MIN:
            continue

        # Nos quedamos con el contorno de mayor área que cumpla criterios
        if area > mejor_area:
            mejor_area = area
            mejor_circ = circularidad
            mejor_cnt = cnt
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            mejor_bbox = (x, y, w_box, h_box)

    if mejor_cnt is None:
        # No se encontró pelota
        return None, None, 0, frame_vis

    # Dibujar la mejor pelota encontrada
    x, y, w_box, h_box = mejor_bbox
    cx = x + w_box // 2
    cy = y + h_box // 2

    cv2.rectangle(frame_vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    cv2.circle(frame_vis, (cx, cy), 4, (255, 0, 0), -1)
    cv2.putText(frame_vis, f"Ball area={int(mejor_area)} circ={mejor_circ:.2f}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # También dibujar el centro de la imagen
    cx_img = w // 2
    cy_img = h // 2
    cv2.circle(frame_vis, (cx_img, cy_img), 5, (0, 255, 255), -1)
    cv2.line(frame_vis, (cx_img - 20, cy_img), (cx_img + 20, cy_img), (0, 255, 255), 1)
    cv2.line(frame_vis, (cx_img, cy_img - 20), (cx_img, cy_img + 20), (0, 255, 255), 1)

    return cx, cy, mejor_area, frame_vis


def calcular_comandos(cx, cy, area, frame_shape):
    """
    Calcula comandos simples de control a partir del centro de la pelota.
    Regresa (cmd_yaw, cmd_altura, cmd_avance).
    """
    h, w = frame_shape[:2]
    cx_img = w // 2
    cy_img = h // 2

    # Errores en pixeles (posición de la pelota - centro de la imagen)
    ex = cx - cx_img   # izquierda/derecha
    ey = cy - cy_img   # arriba/abajo
    ea = AREA_REF - area  # área pequeña: necesitamos acercarnos

    # Control proporcional simple
    cmd_yaw = -Kx * ex     # si la pelota está a la derecha (ex>0), girar a la derecha (ajustar signo según tu dron)
    cmd_alt = -Ky * ey     # si la pelota está arriba (ey<0), subir, etc.
    cmd_avance = Ka * ea   # si el área es pequeña (ea>0), avanzar hacia la pelota

    # Saturar comandos para que no sean enormes
    cmd_yaw = max(min(cmd_yaw, 1.0), -1.0)
    cmd_alt = max(min(cmd_alt, 1.0), -1.0)
    cmd_avance = max(min(cmd_avance, 1.0), -1.0)

    return cmd_yaw, cmd_alt, cmd_avance


# -----------------------------------------
# Main loop de cámara
# -----------------------------------------
# Para PC: cap = cv2.VideoCapture(0)
# Para dron: aquí cambias por la fuente de video de tu dron (URL RTSP, UDP, etc.)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
else:
    print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame")
        break

    # Puedes redimensionar para ir más rápido
    # frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = segmentar_rojo(hsv)
    cx, cy, area, frame_vis = detectar_pelota(mask, frame)

    if cx is not None and cy is not None:
        cmd_yaw, cmd_alt, cmd_avance = calcular_comandos(cx, cy, area, frame.shape)

        # Mostrar comandos en pantalla
        texto_cmd = f"yaw={cmd_yaw:.2f} alt={cmd_alt:.2f} fwd={cmd_avance:.2f}"
        cv2.putText(frame_vis, texto_cmd, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # -----------------------------
        # AQUÍ es donde conectas con tu dron
        # Ejemplo (pseudo-código):
        # drone.set_yaw(cmd_yaw)
        # drone.set_vertical_speed(cmd_alt)
        # drone.set_forward_speed(cmd_avance)
        # -----------------------------
        print("Comandos dron:", texto_cmd)
    else:
        cv2.putText(frame_vis, "Pelota NO detectada", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print("Pelota no detectada")

    cv2.imshow("Camara - Deteccion pelota roja", frame_vis)
    cv2.imshow("Mascara rojo", mask)

    # Tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
